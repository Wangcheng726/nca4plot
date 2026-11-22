// 复原力 · minimal demo: one seed grows into a plant; erase and it regrows.
// All visible motion comes from the exported NCA rule (no painting guards).
import { NCAEngine } from './nca.js';

const VIEW_PX = 480;
const engine = new NCAEngine({
  kind: 'target', width: 80, height: 80,
  modelFile: 'plant_80.onnx', metaFile: 'plant_meta_80.json',
  lockWounds: false, stepsPerRun: 4,
});

const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const off = document.createElement('canvas');
const offCtx = off.getContext('2d');
const statusEl = document.getElementById('status');
const btnPause = document.getElementById('btnPause');
const brushEl = document.getElementById('brushSize');

let paused = false;
let busy = false;
let brush = 4;
let frame = 0;
let targetRGB = null;   // Uint8Array w*h*3, white-composited target
let blankMSE = 1;       // MSE of empty white canvas vs target (recovery scale)
let recovery = 0;

function composite() {
  const { width: w, height: h, stateData, chn } = engine;
  const n2 = w * h;
  const img = offCtx.createImageData(w, h);
  const d = img.data;
  for (let i = 0; i < n2; i += 1) {
    const a = stateData[3 * n2 + i];
    for (let c = 0; c < 3; c += 1) {
      let v = 1 - a + stateData[c * n2 + i];
      v = v < 0 ? 0 : (v > 1 ? 1 : v);
      d[i * 4 + c] = v * 255;
    }
    d[i * 4 + 3] = 255;
  }
  return img;
}

function render() {
  offCtx.putImageData(composite(), 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
}

function updateRecovery() {
  if (!targetRGB) return;
  const img = composite();
  const d = img.data;
  let se = 0;
  for (let i = 0; i < targetRGB.length; i += 1) {
    const diff = d[(i / 3 | 0) * 4 + (i % 3)] - targetRGB[i];
    se += diff * diff;
  }
  const mse = se / targetRGB.length / 65025;
  recovery = Math.max(0, Math.min(1, 1 - mse / blankMSE));
}

function updateStatus() {
  const pct = (recovery * 100).toFixed(1);
  statusEl.textContent =
    `步骤 ${engine.steps} · 恢复度 ${pct}% · ${engine.backend} · ${engine.parameterCount?.toLocaleString()} 参数` +
    (paused ? ' · 已暂停' : '');
}

async function loadTarget() {
  const meta = await (await fetch(new URL('../models/plant_meta_80.json', import.meta.url))).json();
  const img = new Image();
  img.src = new URL(`../models/${meta.target_image}`, import.meta.url).href;
  await img.decode();
  document.getElementById('targetImg').src = img.src;
  const w = engine.width, h = engine.height;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(img, 0, 0, w, h);
  const d = cx.getImageData(0, 0, w, h).data;
  targetRGB = new Uint8Array(w * h * 3);
  let se = 0;
  for (let i = 0; i < w * h; i += 1) {
    for (let k = 0; k < 3; k += 1) {
      targetRGB[i * 3 + k] = d[i * 4 + k];
      const diff = 255 - d[i * 4 + k];
      se += diff * diff;
    }
  }
  blankMSE = Math.max(se / targetRGB.length / 65025, 1e-9);
}

// ---- interaction -----------------------------------------------------------
function toGrid(evt) {
  const r = canvas.getBoundingClientRect();
  return {
    x: (evt.clientX - r.left) / r.width * engine.width,
    y: (evt.clientY - r.top) / r.height * engine.height,
  };
}

let erasing = false;
let last = null;
canvas.addEventListener('pointerdown', (evt) => {
  evt.preventDefault();
  canvas.setPointerCapture(evt.pointerId);
  erasing = true;
  last = toGrid(evt);
  engine.eraseCircle(last.x, last.y, brush, 1.2);
  render(); updateRecovery(); updateStatus();
});
canvas.addEventListener('pointermove', (evt) => {
  if (!erasing) return;
  const p = toGrid(evt);
  engine.eraseSegment(last.x, last.y, p.x, p.y, brush, 1.2);
  last = p;
  render(); updateRecovery(); updateStatus();
});
const stopErase = () => { erasing = false; last = null; };
canvas.addEventListener('pointerup', stopErase);
canvas.addEventListener('pointercancel', stopErase);

function setPaused(v) {
  paused = v;
  btnPause.textContent = paused ? '继续' : '暂停';
  updateStatus();
}
function reset() {
  engine.reset();
  recovery = 0;
  render(); updateStatus();
}
function damage(fx, fy, fr) {
  // Standard test discs (same geometry as the offline probes), scaled to grid.
  const s = engine.width / 80;
  engine.eraseCircle(fx * s, fy * s, fr * s, 0.5);
  render(); updateRecovery(); updateStatus();
}

document.getElementById('btnReset').addEventListener('click', reset);
btnPause.addEventListener('click', () => setPaused(!paused));
document.getElementById('btnBrushDown').addEventListener('click', () => {
  brush = Math.max(2, brush - 1); brushEl.textContent = brush;
});
document.getElementById('btnBrushUp').addEventListener('click', () => {
  brush = Math.min(14, brush + 1); brushEl.textContent = brush;
});
document.getElementById('btnDmgSmall').addEventListener('click', () => damage(45, 50, 6));
document.getElementById('btnDmgBig').addEventListener('click', () => damage(40, 44, 16));
window.addEventListener('keydown', (evt) => {
  if (evt.code === 'Space') { evt.preventDefault(); setPaused(!paused); }
  else if (evt.code === 'KeyR') reset();
});

// ---- main loop -------------------------------------------------------------
async function main() {
  const info = await engine.load();
  off.width = engine.width; off.height = engine.height;
  const zoom = Math.max(1, Math.round(VIEW_PX / engine.width));
  canvas.width = engine.width * zoom;
  canvas.height = engine.height * zoom;
  await loadTarget();
  render(); updateStatus();
  console.log('[plant] loaded', info);
  // Demo pace: 4 CA steps per 100 ms (40 steps/s) so growth is watchable
  // on any display refresh rate (headless rAF is uncapped).
  let lastRun = 0;
  const tick = async (ts) => {
    if (!paused && !busy && ts - lastRun >= 100) {
      busy = true;
      lastRun = ts;
      try { await engine.run(); } catch (err) { console.error(err); }
      busy = false;
      frame += 1;
      if (frame % 12 === 0) updateRecovery();
      render(); updateStatus();
    }
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// Headless-verification / demo handle (read-only helpers + engine).
window.plant = { engine, setPaused, reset, damage, render, updateRecovery, updateStatus, get recovery() { return recovery; }, get paused() { return paused; } };

main().catch((err) => {
  statusEl.textContent = `加载失败：${err.message}`;
  console.error(err);
});
