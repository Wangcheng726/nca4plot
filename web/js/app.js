import { NCAEngine } from './nca.js';
import { SkyRenderer } from './renderer.js';

const canvas = document.querySelector('#sky');
const cursor = document.querySelector('#cursor');
const intro = document.querySelector('#intro');
const enterButton = document.querySelector('#enterButton');
const enterLabel = document.querySelector('#enterLabel');
const loadingDetail = document.querySelector('#loadingDetail');
const statusPill = document.querySelector('#statusPill');
const statusText = document.querySelector('#statusText');
const stepCounter = document.querySelector('#stepCounter');
const pauseButton = document.querySelector('#pauseButton');
const resetButton = document.querySelector('#resetButton');
const qualitySelect = document.querySelector('#qualitySelect');
const aboutButton = document.querySelector('#aboutButton');
const aboutPanel = document.querySelector('#about');
const aboutClose = document.querySelector('#aboutClose');
const fullscreenButton = document.querySelector('#fullscreenButton');
const gestureNote = document.querySelector('#gestureNote');
const gestureStrong = gestureNote.querySelector('strong');
const gestureSpan = gestureNote.querySelector('p span');
const toast = document.querySelector('#toast');
const modeButtons = Array.from(document.querySelectorAll('.mode-button'));

const params = new URLSearchParams(location.search);
const captureMode = params.has('capture');
const autostart = captureMode || params.has('autostart') || params.get('enter') === '1';
const requestedQuality = Number(params.get('quality'));
const initialSize = [128, 256, 512].includes(requestedQuality) ? requestedQuality : 256;
const MODES = ['regenerate', 'growing', 'persistent'];
const requestedMode = params.get('mode');
let mode = MODES.includes(requestedMode) ? requestedMode : 'regenerate';

const MODE_COPY = {
  regenerate: {
    gesture: ['拖动，撕开夜空', '松手之后，它会重新长回原来的画。'],
    toast: '再生模式：撕开的地方，会长回《星空》本来的样子。',
  },
  growing: {
    gesture: ['拖动，撕开夜空', '散落的记忆星点，自组织成整幅天空。'],
    toast: '生长模式：散落的记忆星点，长回整幅《星空》。',
  },
  persistent: {
    gesture: ['拖动，撕开夜空', '松手之后，它会自己重新组织。'],
    toast: '永动模式：没有目标图，天空永远流动、永不重复。',
  },
};

const renderer = new SkyRenderer(canvas);
let engine = null;
let entered = false;
let paused = false;
let busy = false;
let switching = false;
let frameHandle = 0;
let lastFrame = performance.now();
let lastInference = 0;
let inferenceDelay = 1000 / 30;
let woundUntil = 0;
let woundCount = 0;
let lastStatusKey = '';
let activePointerId = null;
let lastPoint = null;
let toastTimer = 0;
let rendererReady = false;
let sessionInfo = { backend: '—', loadMs: 0 };
const targetMetaCache = new Map();

if (captureMode) document.body.classList.add('capture');

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function padSteps(value) {
  return String(Math.floor(value)).padStart(6, '0');
}

function setStatus(label, modeClass = '') {
  const key = `${label}|${modeClass}`;
  if (key === lastStatusKey) return;
  lastStatusKey = key;
  statusText.textContent = label;
  statusPill.classList.toggle('running', modeClass === 'running');
  statusPill.classList.toggle('wounded', modeClass === 'wounded');
}

function updateStepCounter() {
  stepCounter.textContent = padSteps(engine?.steps ?? 0);
}

function updateStatus() {
  if (!engine) return;
  const now = performance.now();
  if (switching) {
    setStatus('正在切换天空的记忆…');
  } else if (paused) {
    setStatus('已暂停 · 局部规则静候');
  } else if (now < woundUntil) {
    setStatus(engine.kind === 'target' ? '伤口正在长回原画' : '伤口边界失效 · 正在再生', 'wounded');
  } else if (!entered) {
    setStatus('正在唤醒局部规则');
  } else if (engine.kind === 'target') {
    if (mode === 'growing' && engine.steps < 90) {
      setStatus('从记忆星群出发 · 自组织中', 'running');
    } else if (mode === 'growing') {
      setStatus('已长成 · 记忆中的星空', 'running');
    } else if (engine.steps < 150) {
      setStatus('记忆完好 · 天空正在苏醒', 'running');
    } else {
      setStatus('记忆完好 · 撕开后会重新长回', 'running');
    }
  } else if (engine.steps < 280) {
    setStatus('从一颗零出发 · 生长中', 'running');
  } else {
    setStatus('持续流动 · never settling', 'running');
  }
  updateStepCounter();
}

function showToast(message, duration = 2400) {
  toast.textContent = message;
  toast.classList.add('visible');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('visible'), duration);
}

function updateModeButtons() {
  for (const button of modeButtons) {
    const active = button.dataset.mode === mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', active ? 'true' : 'false');
  }
}

function setModeButtonsDisabled(disabled) {
  for (const button of modeButtons) button.disabled = disabled;
}

function updateGestureCopy() {
  const copy = MODE_COPY[mode] ?? MODE_COPY.regenerate;
  gestureStrong.textContent = copy.gesture[0];
  gestureSpan.textContent = copy.gesture[1];
}

async function loadTargetMeta(sizeKey = '') {
  if (!targetMetaCache.has(sizeKey)) {
    const name = sizeKey ? `nca_target_meta_${sizeKey}.json` : 'nca_target_meta.json';
    let response = await fetch(new URL(`../models/${name}`, import.meta.url).href);
    if (!response.ok && sizeKey) {
      response = await fetch(new URL('../models/nca_target_meta.json', import.meta.url).href);
    }
    if (!response.ok) throw new Error(`target meta missing: ${response.status}`);
    targetMetaCache.set(sizeKey, await response.json());
  }
  return targetMetaCache.get(sizeKey);
}

function warmPlan(nextMode) {
  if (nextMode === 'persistent') return captureMode ? 680 : 280;
  if (nextMode === 'regenerate') return captureMode ? 640 : 460;
  return 0; // growing: the visitor watches the memory constellation coalesce
}

async function buildEngine(nextMode, quality, warmSteps, budgetMs) {
  let next;
  if (nextMode === 'persistent') {
    next = new NCAEngine({ kind: 'texture', width: quality, height: quality });
  } else {
    const meta = await loadTargetMeta();
    // Growing shares the painting weights but boots from a sparse memory
    // constellation and runs on a slower clock so the self-organisation is
    // watchable (~14 CA steps/s instead of ~60).
    next = new NCAEngine({
      kind: 'target', width: meta.width, height: meta.height,
      stepsPerRun: nextMode === 'growing' ? 1 : undefined,
    });
  }
  let info;
  try {
    info = await next.load();
    if (nextMode === 'growing') next.plantConstellation();
    if (warmSteps > 0) {
      await next.warmup(warmSteps, budgetMs, (steps) => {
        if (!entered) {
          enterLabel.textContent = `天空生长至 ${steps} 步`;
          loadingDetail.textContent = `${info.backend === 'webgpu' ? 'WebGPU' : 'WASM'} · 局部规则正在协商全局形态`;
        }
      });
    }
  } catch (error) {
    next.dispose();
    throw error;
  }
  sessionInfo = { ...info, size: next.width };
  return next;
}

async function waitUntilIdle(timeoutMs = 2000) {
  const started = performance.now();
  while (busy && performance.now() - started < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

async function applyMode(nextMode, { silent = false } = {}) {
  if (!MODES.includes(nextMode) || switching) return;
  const nextKind = nextMode === 'persistent' ? 'texture' : 'target';
  if (engine && nextMode === mode && engine.kind === nextKind) return;
  switching = true;
  setModeButtonsDisabled(true);
  const previous = engine;
  const previousMode = mode;
  try {
    await waitUntilIdle();
    const quality = Number(qualitySelect.value) || initialSize;
    const next = await buildEngine(nextMode, quality, warmPlan(nextMode), captureMode ? 16000 : 8000);
    previous?.dispose();
    engine = next;
    mode = nextMode;
    paused = false;
    pauseButton.classList.remove('paused');
    pauseButton.setAttribute('aria-label', '暂停');
    renderer.setKind(engine.kind);
    renderer.setSimulationSize(engine.width, engine.height);
    renderer.resetActivity();
    renderer.uploadState(engine.stateData, engine.width, engine.height);
    woundUntil = 0;
    lastInference = 0;
    updateModeButtons();
    updateGestureCopy();
    updateStatus();
    if (!silent) showToast(MODE_COPY[mode].toast, 2800);
  } catch (error) {
    console.error('[nca4plot] mode switch failed', error);
    mode = previousMode;
    updateModeButtons();
    showToast('切换失败，已保留原来的天空。', 3400);
  } finally {
    switching = false;
    setModeButtonsDisabled(false);
    qualitySelect.disabled = mode !== 'persistent';
  }
}

async function boot() {
  try {
    await renderer.init();
    renderer.setKind(mode === 'persistent' ? 'texture' : 'target');
    renderer.setSimulationSize(initialSize);
    rendererReady = true;
    renderLoop(performance.now());

    qualitySelect.value = String(initialSize);
    loadingDetail.textContent = '加载 5,856 参数的神经细胞自动机…';
    updateModeButtons();
    updateGestureCopy();
    await applyMode(mode, { silent: true });
    if (!engine) throw new Error('engine was not built');
    updateStepCounter();

    if (engine.steps < 20) {
      loadingDetail.textContent = '模型已就绪，天空将从一片记忆星群开始生长。';
    } else {
      loadingDetail.textContent = `${engine.backend === 'webgpu' ? 'WebGPU' : 'WASM'} · 已生长 ${engine.steps} 步 · 5,856 参数`;
    }
    enterButton.disabled = false;
    enterLabel.textContent = captureMode ? '进入天空' : '进入这片天空';
    setStatus('已就绪 · 等待进入');
    updateStepCounter();

    if (autostart) {
      enterSky();
    }
  } catch (error) {
    console.error('[nca4plot] boot failed', error);
    enterLabel.textContent = '启动失败';
    loadingDetail.textContent = `${error.message} 请使用支持 WebGL2 的现代浏览器。`;
    setStatus('无法启动');
  }
}

function enterSky() {
  if (entered || !engine) return;
  entered = true;
  document.body.classList.add('entered');
  intro.setAttribute('aria-hidden', 'true');
  woundUntil = 0;
  lastInference = 0;
  updateStatus();
}

function renderLoop(now) {
  frameHandle = requestAnimationFrame(renderLoop);
  const dt = Math.min(now - lastFrame, 100);
  lastFrame = now;

  if (rendererReady) {
    renderer.render(now * 0.001);
    renderer.timeScale = paused ? 0 : 1;
  }

  if (!entered || paused || busy || switching || !engine) {
    updateStatus();
    return;
  }

  if (now - lastInference >= inferenceDelay) {
    lastInference = now;
    busy = true;
    const started = performance.now();
    engine.run()
      .then((advanced) => {
        if (advanced) {
          renderer.uploadState(engine.stateData, engine.width, engine.height);
          updateStepCounter();
        }
      })
      .catch((error) => {
        console.error('[nca4plot] inference failed', error);
        paused = true;
        pauseButton.classList.add('paused');
        showToast('推理中断，已暂停以保护当前天空。', 4200);
      })
      .finally(() => {
        const elapsed = performance.now() - started;
        const floor = mode === 'growing' ? 71 : 16.7;
      inferenceDelay = clamp(Math.max(floor, elapsed * 1.06), floor, 130);
        busy = false;
      });
  }

  updateStatus();
}

function simulationRadiusFromScreen(screenRadius) {
  if (!engine) return 10;
  const r = renderer.caScreenRect();
  return Math.max(1.2, screenRadius * engine.width / Math.max(r.width, 1));
}

function screenBrushRadius() {
  const rect = canvas.getBoundingClientRect();
  return clamp(Math.min(rect.width, rect.height) * 0.058, 27, 78);
}

function updateCursorSize() {
  const radius = screenBrushRadius();
  cursor.style.setProperty('--cursor-size', `${Math.round(radius * 2)}px`);
}

function mapPointerToState(clientX, clientY) {
  return renderer.screenToCa(clientX, clientY);
}

function paintWound(point, previous = null) {
  if (!engine || !point || (paused && !entered)) return;
  const radius = simulationRadiusFromScreen(screenBrushRadius());
  if (previous) {
    engine.eraseSegment(previous.x, previous.y, point.x, point.y, radius, 1.5);
  } else {
    engine.eraseCircle(point.x, point.y, radius, 1.5);
  }
  renderer.markWound(point.x, point.y, radius);
  if (previous) renderer.markWound(previous.x, previous.y, radius);
  renderer.uploadState(engine.stateData, engine.width, engine.height);
  woundUntil = performance.now() + 1050;
  updateStatus();
}

function moveCursor(event) {
  cursor.style.left = `${event.clientX}px`;
  cursor.style.top = `${event.clientY}px`;
  document.body.classList.add('pointer-active');
}

canvas.addEventListener('pointerenter', (event) => {
  if (!entered) return;
  updateCursorSize();
  moveCursor(event);
});

canvas.addEventListener('pointermove', (event) => {
  if (!entered) return;
  moveCursor(event);
  if (activePointerId !== event.pointerId) return;
  const point = mapPointerToState(event.clientX, event.clientY);
  if (point) {
    paintWound(point, lastPoint);
    lastPoint = point;
  } else {
    lastPoint = null;
  }
});

canvas.addEventListener('pointerdown', (event) => {
  if (!entered || event.button > 0) return;
  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  document.body.classList.add('erasing', 'has-drawn');
  updateCursorSize();
  moveCursor(event);
  lastPoint = mapPointerToState(event.clientX, event.clientY);
  if (lastPoint) {
    woundCount += 1;
    paintWound(lastPoint);
  }
  event.preventDefault();
});

function finishStroke(event) {
  if (activePointerId === null) return;
  if (event && event.pointerId !== undefined && event.pointerId !== activePointerId) return;
  activePointerId = null;
  lastPoint = null;
  document.body.classList.remove('erasing');
  woundUntil = performance.now() + 900;
  updateStatus();
}

canvas.addEventListener('pointerup', finishStroke);
canvas.addEventListener('pointercancel', finishStroke);
window.addEventListener('pointerup', finishStroke);
canvas.addEventListener('pointerleave', () => {
  if (activePointerId === null) document.body.classList.remove('pointer-active');
});
window.addEventListener('resize', () => {
  renderer.resize();
  updateCursorSize();
  if (engine) renderer.uploadState(engine.stateData, engine.width, engine.height);
});

pauseButton.addEventListener('click', () => {
  if (!engine) return;
  paused = !paused;
  renderer.timeScale = paused ? 0 : 1;
  pauseButton.classList.toggle('paused', paused);
  pauseButton.setAttribute('aria-label', paused ? '继续' : '暂停');
  showToast(paused ? '天空停在此刻；伤口也会等待。' : '局部规则重新开始流动。', 1600);
  updateStatus();
});

resetButton.addEventListener('click', () => {
  if (!engine || busy || switching) return;
  if (mode === 'growing') engine.plantConstellation();
  else engine.reset();
  paused = false;
  pauseButton.classList.remove('paused');
  pauseButton.setAttribute('aria-label', '暂停');
  renderer.resetActivity();
  renderer.uploadState(engine.stateData, engine.width, engine.height);
  document.body.classList.remove('has-drawn');
  woundUntil = 0;
  lastInference = 0;
  showToast(mode === 'persistent'
    ? '一切归零。新的天空正在长出来。'
    : mode === 'growing'
      ? '回到一片记忆星群。天空会重新长出来。'
      : '记忆复原。天空回到《星空》本来的样子。', 2400);
  updateStatus();
});

for (const button of modeButtons) {
  button.addEventListener('click', () => {
    if (!engine || switching) return;
    applyMode(button.dataset.mode);
  });
}

qualitySelect.addEventListener('change', async () => {
  const size = Number(qualitySelect.value);
  if (!engine || mode !== 'persistent' || size === engine.width) return;
  qualitySelect.disabled = true;
  const wasPaused = paused;
  const previousInfo = sessionInfo;
  paused = true;
  setStatus(`重新编织 ${size}px 天空物质…`);
  showToast(`切换至 ${size}px 模拟网格，正在重新生长。`, 3200);
  try {
    await waitUntilIdle();
    const next = await buildEngine('persistent', size, size === 512 ? 180 : 240, 6500);
    engine.dispose();
    engine = next;
    renderer.setSimulationSize(size, size);
    renderer.uploadState(engine.stateData, size, size);
    paused = wasPaused;
    pauseButton.classList.toggle('paused', paused);
    updateStepCounter();
    updateStatus();
    showToast(`${size}px 天空已就绪 · ${sessionInfo.backend.toUpperCase()}`, 2200);
  } catch (error) {
    console.error('[nca4plot] quality switch failed', error);
    sessionInfo = previousInfo;
    showToast('切换失败，已保留原来的天空。', 3400);
    qualitySelect.value = String(engine.width);
    paused = wasPaused;
    pauseButton.classList.toggle('paused', paused);
    updateStatus();
  } finally {
    qualitySelect.disabled = mode !== 'persistent';
  }
});

function openAbout() {
  aboutPanel.hidden = false;
  aboutButton.setAttribute('aria-expanded', 'true');
  requestAnimationFrame(() => aboutClose.focus({ preventScroll: true }));
}

function closeAbout() {
  aboutPanel.hidden = true;
  aboutButton.setAttribute('aria-expanded', 'false');
  aboutButton.focus({ preventScroll: true });
}

aboutButton.addEventListener('click', openAbout);
aboutClose.addEventListener('click', closeAbout);
aboutPanel.querySelector('[data-close-about]').addEventListener('click', closeAbout);
window.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && !aboutPanel.hidden) closeAbout();
  if (event.key === ' ' && entered && aboutPanel.hidden) {
    event.preventDefault();
    pauseButton.click();
  }
});

fullscreenButton.addEventListener('click', async () => {
  try {
    if (!document.fullscreenElement) {
      await document.documentElement.requestFullscreen();
    } else {
      await document.exitFullscreen();
    }
  } catch (error) {
    console.warn('[nca4plot] fullscreen unavailable', error);
    showToast('当前浏览器未允许全屏。', 2200);
  }
});

enterButton.addEventListener('click', enterSky);

updateCursorSize();
updateStatus();
boot();

// Small public surface for testing, documentation screenshots, and live debugging.
window.nca4plot = {
  get engine() { return engine; },
  get renderer() { return renderer; },
  get entered() { return entered; },
  get paused() { return paused; },
  get steps() { return engine?.steps ?? 0; },
  get backend() { return sessionInfo.backend; },
  get woundCount() { return woundCount; },
  get mode() { return mode; },
  enter: enterSky,
  reset: () => resetButton.click(),
  pause: () => pauseButton.click(),
  setMode: (next) => applyMode(next),
  eraseNormalized(u, v, radius = 10) {
    if (!engine) return;
    const x = u * engine.width;
    const y = (1 - v) * engine.height;
    engine.eraseCircle(x, y, radius, 1.5);
    renderer.markWound(x, y, radius);
    renderer.uploadState(engine.stateData, engine.width, engine.height);
    woundUntil = performance.now() + 1200;
    document.body.classList.add('has-drawn');
    updateStatus();
  },
  stats() {
    if (!engine) return null;
    const data = engine.stateData;
    let sum = 0;
    let sumSq = 0;
    let min = Infinity;
    let max = -Infinity;
    const stride = Math.max(1, Math.floor(data.length / 40000));
    let count = 0;
    for (let i = 0; i < data.length; i += stride) {
      const value = data[i];
      sum += value;
      sumSq += value * value;
      min = Math.min(min, value);
      max = Math.max(max, value);
      count += 1;
    }
    const mean = sum / count;
    return {
      size: engine.width,
      width: engine.width,
      height: engine.height,
      kind: engine.kind,
      mode,
      steps: engine.steps,
      backend: sessionInfo.backend,
      mean,
      std: Math.sqrt(Math.max(0, sumSq / count - mean * mean)),
      min,
      max,
    };
  },
};

window.addEventListener('beforeunload', () => {
  cancelAnimationFrame(frameHandle);
});