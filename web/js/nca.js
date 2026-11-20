import * as ort from '../assets/ort/ort.webgpu.bundle.min.mjs';

const STEPS_PER_RUN = { 128: 4, 256: 4, 512: 2 };

// Persistent-mode damage lock: cells erased past this keep fraction are pinned
// to zero forever -- Distill "persistent" semantics, where damage never heals.
// This is a boundary condition on the state, not learned weights: every visible
// motion still comes from the NCA rule itself; the lock only refuses to let the
// wound core regrow. The feathered rim keeps less than full erase and stays
// alive, so the sky reweaves around an open wound.
const LOCK_KEEP_THRESHOLD = 0.35;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

async function loadSeedPixels(url, width, height) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`target seed image missing: ${response.status}`);
  const bitmap = await createImageBitmap(await response.blob());
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  const pixels = ctx.getImageData(0, 0, width, height).data;
  const n2 = width * height;
  const out = new Float32Array(4 * n2);
  for (let i = 0; i < n2; i += 1) {
    out[i] = pixels[i * 4] / 255;
    out[n2 + i] = pixels[i * 4 + 1] / 255;
    out[2 * n2 + i] = pixels[i * 4 + 2] / 255;
    out[3 * n2 + i] = pixels[i * 4 + 3] / 255;
  }
  return out;
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
}

// Two flavours of the same 5,856-parameter rule:
//  - texture: square grid, 12ch (RGB+0.5 offset, 9 hidden), infinite drift;
//             erased wound cores are damage-locked open (persistent mode).
//  - target:  rectangular grid, 12ch (RGBA in [0,1], 8 hidden), regrows a
//             fixed painting. Zero state = transparent = dead matter.
export class NCAEngine {
  constructor(options = {}) {
    if (typeof options === 'number') options = { width: options, height: options };
    this.kind = options.kind ?? 'texture';
    this.width = options.width ?? 256;
    this.height = options.height ?? this.width;
    this.modelFile = options.modelFile ?? (this.kind === 'texture'
      ? `nca_starry_${this.width}.onnx`
      : `nca_target_${this.width}x${this.height}.onnx`);
    this.stepsPerRun = options.stepsPerRun
      ?? (this.kind === 'texture' ? (STEPS_PER_RUN[this.width] ?? 4) : 2);
    this.seedRGBA = null;
    this.seedFull = null;
    this.session = null;
    this.backend = 'uninitialised';
    this.stateData = null;
    this.masksData = null;
    this.masksTensor = null;
    this.lockMask = null;
    this.lockedPixels = [];
    this.steps = 0;
    this.version = 0;
    this.loadMs = 0;
  }

  // Legacy square-grid alias.
  get size() { return this.width; }
  get n2() { return this.width * this.height; }

  async load() {
    const started = performance.now();
    ort.env.logLevel = 'error';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = new URL('../assets/ort/', import.meta.url).href;
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance';

    if (this.kind === 'target') {
      // Per-grid meta first; the legacy un-suffixed file remains the
      // 256x144 fallback for both target modes.
      const sizedUrl = new URL(`../models/nca_target_meta_${this.width}x${this.height}.json`, import.meta.url).href;
      const legacyUrl = new URL('../models/nca_target_meta.json', import.meta.url).href;
      let response = await fetch(sizedUrl);
      if (!response.ok) response = await fetch(legacyUrl);
      if (!response.ok) throw new Error(`target meta missing: ${response.status}`);
      const meta = await response.json();
      this.seedRGBA = meta.seed_rgba;
      this.stepsPerRun = meta.steps_per_inference ?? this.stepsPerRun;
      if (meta.width !== this.width || meta.height !== this.height) {
        throw new Error(`target meta ${meta.width}x${meta.height} != engine ${this.width}x${this.height}`);
      }
      if (meta.seed_image) {
        const seedUrl = new URL(`../models/${meta.seed_image}`, import.meta.url).href;
        this.seedFull = await loadSeedPixels(seedUrl, this.width, this.height);
      }
    }

    const modelUrl = new URL(`../models/${this.modelFile}`, import.meta.url).href;
    const common = {
      graphOptimizationLevel: 'all',
      executionMode: 'sequential',
    };

    let session = null;
    let backend = 'wasm';
    if ('gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
        if (adapter) {
          session = await ort.InferenceSession.create(modelUrl, {
            ...common,
            executionProviders: ['webgpu'],
          });
          backend = 'webgpu';
        }
      } catch (error) {
        console.warn('[nca] WebGPU unavailable, falling back to WASM.', error);
      }
    }

    if (!session) {
      session = await ort.InferenceSession.create(modelUrl, {
        ...common,
        executionProviders: ['wasm'],
      });
    }

    this.session = session;
    this.backend = backend;
    this.loadMs = performance.now() - started;
    this._allocate();
    this.reset();
    return { backend, loadMs: this.loadMs };
  }

  _allocate() {
    const n2 = this.n2;
    this.stateData = new Float32Array(12 * n2);
    this.masksData = new Float32Array(this.stepsPerRun * n2);
    this.masksTensor = new ort.Tensor(
      'float32',
      this.masksData,
      [this.stepsPerRun, 1, this.height, this.width],
    );
    this.lockMask = this.kind === 'texture' ? new Uint8Array(n2) : null;
    this.lockedPixels = [];
    this.steps = 0;
    this.version += 1;
  }

  reset() {
    this.stateData.fill(0);
    if (this.lockMask) this.lockMask.fill(0);
    this.lockedPixels.length = 0;
    if (this.kind === 'target' && this.seedFull) {
      // Regenerate boots as the living painting itself: RGBA = the memory of
      // the sky, hidden channels = 0. Stability on this state is trained.
      this.stateData.set(this.seedFull, 0);
    }
    this.steps = 0;
    this.version += 1;
  }

  plantConstellation(density = 0.6) {
    // Growing mode: the sky begins as a sparse constellation of memory --
    // scattered living cells carrying the painting's own pigment (hidden 0).
    // Every dead cell then sits beside living memory, so the trained regrowth
    // dynamics coalesce the constellation into the whole painting. A lone
    // seed cannot carry composition memory across 100+ cells at 5,856 params
    // (see checkpoints/*_training_log*.jsonl diagnostics); the constellation
    // is the honest initial condition that grows back the original sky.
    this.stateData.fill(0);
    if (this.kind === 'target' && this.seedFull) {
      const n2 = this.n2;
      for (let i = 0; i < n2; i += 1) {
        if (Math.random() >= density) continue;
        for (let channel = 0; channel < 4; channel += 1) {
          this.stateData[channel * n2 + i] = this.seedFull[channel * n2 + i];
        }
      }
    }
    this.steps = 0;
    this.version += 1;
  }

  _fillMasks() {
    const data = this.masksData;
    for (let i = 0; i < data.length; i += 1) {
      data[i] = Math.random() < 0.5 ? 0 : 1;
    }
  }

  async run() {
    if (!this.session) throw new Error('NCA session is not loaded.');
    const versionAtStart = this.version;
    this._fillMasks();
    const stateTensor = new ort.Tensor(
      'float32',
      this.stateData,
      [1, 12, this.height, this.width],
    );

    const result = await this.session.run({
      state: stateTensor,
      masks: this.masksTensor,
    });

    // A reset or a fresh wound arrived while inference was in flight. The result
    // belongs to the old state and must not resurrect erased cells.
    if (versionAtStart !== this.version) return false;

    this.stateData = result.state_out.data;
    if (this.lockedPixels.length) {
      const n2 = this.n2;
      const data = this.stateData;
      for (let k = 0; k < this.lockedPixels.length; k += 1) {
        const pixel = this.lockedPixels[k];
        for (let channel = 0; channel < 12; channel += 1) {
          data[channel * n2 + pixel] = 0;
        }
      }
    }
    this.steps += this.stepsPerRun;
    return true;
  }

  async warmup(targetSteps, maxMs = 5200, onProgress = null) {
    const started = performance.now();
    while (this.steps < targetSteps) {
      if (performance.now() - started > maxMs) break;
      const advanced = await this.run();
      if (!advanced) break;
      if (onProgress) onProgress(this.steps, performance.now() - started);
    }
    return this.steps;
  }

  eraseCircle(cx, cy, radius, feather = 1.6) {
    radius = clamp(radius, 0.75, Math.min(this.width, this.height) * 0.4);
    feather = Math.max(feather, 0.4);
    const w = this.width;
    const h = this.height;
    const n2 = this.n2;
    const x0 = Math.max(0, Math.floor(cx - radius - feather));
    const x1 = Math.min(w - 1, Math.ceil(cx + radius + feather));
    const y0 = Math.max(0, Math.floor(cy - radius - feather));
    const y1 = Math.min(h - 1, Math.ceil(cy + radius + feather));
    this.version += 1;

    for (let y = y0; y <= y1; y += 1) {
      const dy = y - cy;
      for (let x = x0; x <= x1; x += 1) {
        const dx = x - cx;
        const distance = Math.hypot(dx, dy);
        if (distance > radius + feather) continue;
        const erase = 1 - smoothstep(radius - feather, radius + feather, distance);
        if (erase <= 0) continue;
        const keep = 1 - erase;
        const pixel = y * w + x;
        if (this.kind === 'texture' && keep < LOCK_KEEP_THRESHOLD) {
          // Wound core: permanent damage. Zero every channel now and record the
          // cell so run() re-pins it after each inference -- the texture rule
          // can never heal it, while the feathered rim keeps drifting.
          if (this.lockMask && !this.lockMask[pixel]) {
            this.lockMask[pixel] = 1;
            this.lockedPixels.push(pixel);
          }
          for (let channel = 0; channel < 12; channel += 1) {
            this.stateData[channel * n2 + pixel] = 0;
          }
          continue;
        }
        for (let channel = 0; channel < 12; channel += 1) {
          this.stateData[channel * n2 + pixel] *= keep;
        }
      }
    }
  }

  eraseSegment(x0, y0, x1, y1, radius, feather = 1.7) {
    const dx = x1 - x0;
    const dy = y1 - y0;
    const distance = Math.hypot(dx, dy);
    const spacing = Math.max(radius * 0.42, 0.8);
    const count = Math.max(1, Math.ceil(distance / spacing));
    for (let i = 0; i <= count; i += 1) {
      const t = i / count;
      this.eraseCircle(x0 + dx * t, y0 + dy * t, radius, feather);
    }
  }

  dispose() {
    this.session?.release?.();
    this.session = null;
    this.stateData = null;
    this.masksData = null;
    this.masksTensor = null;
    this.lockMask = null;
    this.lockedPixels = [];
  }

  get memoryBytes() {
    return this.stateData ? this.stateData.byteLength : 0;
  }
}

export { ort };