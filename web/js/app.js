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
const toast = document.querySelector('#toast');

const params = new URLSearchParams(location.search);
const captureMode = params.has('capture');
const autostart = captureMode || params.has('autostart') || params.get('enter') === '1';
const requestedQuality = Number(params.get('quality'));
const initialSize = [128, 256, 512].includes(requestedQuality) ? requestedQuality : 256;

const renderer = new SkyRenderer(canvas);
let engine = null;
let entered = false;
let paused = false;
let busy = false;
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

if (captureMode) document.body.classList.add('capture');

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function padSteps(value) {
  return String(Math.floor(value)).padStart(6, '0');
}

function setStatus(label, mode = '') {
  const key = `${label}|${mode}`;
  if (key === lastStatusKey) return;
  lastStatusKey = key;
  statusText.textContent = label;
  statusPill.classList.toggle('running', mode === 'running');
  statusPill.classList.toggle('wounded', mode === 'wounded');
}

function updateStepCounter() {
  stepCounter.textContent = padSteps(engine?.steps ?? 0);
}

function updateStatus() {
  if (!engine) return;
  const now = performance.now();
  if (paused) {
    setStatus('已暂停 · 局部规则静候');
  } else if (now < woundUntil) {
    setStatus('伤口边界失效 · 正在再生', 'wounded');
  } else if (!entered) {
    setStatus('正在唤醒局部规则');
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

async function makeEngine(size, warmSteps = 0, budgetMs = 5000) {
  const next = new NCAEngine(size);
  let info;
  try {
    info = await next.load();
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
  sessionInfo = { ...info, size };
  return next;
}

async function waitUntilIdle(timeoutMs = 2000) {
  const started = performance.now();
  while (busy && performance.now() - started < timeoutMs) {
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
}

async function boot() {
  try {
    await renderer.init();
    renderer.setSimulationSize(initialSize);
    rendererReady = true;
    renderLoop(performance.now());

    loadingDetail.textContent = '加载 5,856 参数的神经细胞自动机…';
    const warmTarget = captureMode ? 680 : 280;
    const warmBudget = captureMode ? 16000 : 5600;
    engine = await makeEngine(initialSize, warmTarget, warmBudget);
    renderer.uploadState(engine.stateData, engine.size);
    updateStepCounter();

    if (engine.steps < 20) {
      loadingDetail.textContent = '模型已就绪，天空将从零开始生长。';
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

  if (!entered || paused || busy || !engine) {
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
          renderer.uploadState(engine.stateData, engine.size);
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
        inferenceDelay = clamp(Math.max(16.7, elapsed * 1.06), 16.7, 130);
        busy = false;
      });
  }

  updateStatus();
}

function simulationRadiusFromScreen(screenRadius) {
  if (!engine) return 10;
  const rect = canvas.getBoundingClientRect();
  return Math.max(1.2, screenRadius * engine.size / Math.max(rect.width, 1));
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
  const rect = canvas.getBoundingClientRect();
  const sx = clamp((clientX - rect.left) / Math.max(rect.width, 1), 0, 1);
  const sy = clamp(1 - (clientY - rect.top) / Math.max(rect.height, 1), 0, 1);
  const aspect = rect.width / Math.max(rect.height, 1);
  let ux = sx;
  let uy = sy;
  if (aspect > 1) {
    uy = 0.5 + (sy - 0.5) / aspect;
  } else {
    ux = 0.5 + (sx - 0.5) * aspect;
  }
  return {
    x: ux * engine.size,
    y: (1 - uy) * engine.size,
  };
}

function paintWound(point, previous = null) {
  if (!engine || paused && !entered) return;
  const radius = simulationRadiusFromScreen(screenBrushRadius());
  if (previous) {
    engine.eraseSegment(previous.x, previous.y, point.x, point.y, radius, 1.5);
  } else {
    engine.eraseCircle(point.x, point.y, radius, 1.5);
  }
  renderer.markWound(point.x, point.y, radius);
  if (previous) renderer.markWound(previous.x, previous.y, radius);
  renderer.uploadState(engine.stateData, engine.size);
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
  paintWound(point, lastPoint);
  lastPoint = point;
});

canvas.addEventListener('pointerdown', (event) => {
  if (!entered || event.button > 0) return;
  activePointerId = event.pointerId;
  canvas.setPointerCapture(event.pointerId);
  document.body.classList.add('erasing', 'has-drawn');
  updateCursorSize();
  moveCursor(event);
  lastPoint = mapPointerToState(event.clientX, event.clientY);
  woundCount += 1;
  paintWound(lastPoint);
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
  if (engine) renderer.uploadState(engine.stateData, engine.size);
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
  if (!engine || busy) return;
  engine.reset();
  paused = false;
  pauseButton.classList.remove('paused');
  pauseButton.setAttribute('aria-label', '暂停');
  renderer.resetActivity();
  renderer.uploadState(engine.stateData, engine.size);
  document.body.classList.remove('has-drawn');
  woundUntil = 0;
  lastInference = 0;
  showToast('一切归零。新的天空正在长出来。', 2400);
  updateStatus();
});

qualitySelect.addEventListener('change', async () => {
  const size = Number(qualitySelect.value);
  if (!engine || size === engine.size) return;
  qualitySelect.disabled = true;
  const wasPaused = paused;
  const previousInfo = sessionInfo;
  paused = true;
  setStatus(`重新编织 ${size}px 天空物质…`);
  showToast(`切换至 ${size}px 模拟网格，正在重新生长。`, 3200);
  try {
    await waitUntilIdle();
    const next = await makeEngine(size, size === 512 ? 180 : 240, 6500);
    engine.dispose();
    engine = next;
    renderer.setSimulationSize(size);
    renderer.uploadState(engine.stateData, size);
    paused = wasPaused;
    pauseButton.classList.toggle('paused', paused);
    updateStepCounter();
    updateStatus();
    showToast(`${size}px 天空已就绪 · ${sessionInfo.backend.toUpperCase()}`, 2200);
  } catch (error) {
    console.error('[nca4plot] quality switch failed', error);
    sessionInfo = previousInfo;
    showToast('切换失败，已保留原来的天空。', 3400);
    qualitySelect.value = String(engine.size);
    paused = wasPaused;
    pauseButton.classList.toggle('paused', paused);
    updateStatus();
  } finally {
    qualitySelect.disabled = false;
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
  enter: enterSky,
  reset: () => resetButton.click(),
  pause: () => pauseButton.click(),
  eraseNormalized(u, v, radius = 10) {
    if (!engine) return;
    engine.eraseCircle(u * engine.size, (1 - v) * engine.size, radius, 1.5);
    renderer.markWound(u * engine.size, (1 - v) * engine.size, radius);
    renderer.uploadState(engine.stateData, engine.size);
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
      size: engine.size,
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
