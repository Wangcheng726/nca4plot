import * as ort from '../assets/ort/ort.webgpu.bundle.min.mjs';

const STEPS_PER_RUN = { 128: 4, 256: 4, 512: 2 };

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp((x - edge0) / Math.max(edge1 - edge0, 1e-6), 0, 1);
  return t * t * (3 - 2 * t);
}

export class NCAEngine {
  constructor(size = 256) {
    this.size = size;
    this.stepsPerRun = STEPS_PER_RUN[size] ?? 4;
    this.session = null;
    this.backend = 'uninitialised';
    this.stateData = null;
    this.masksData = null;
    this.masksTensor = null;
    this.steps = 0;
    this.version = 0;
    this.loadMs = 0;
  }

  async load() {
    const started = performance.now();
    ort.env.logLevel = 'error';
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    ort.env.wasm.wasmPaths = new URL('../assets/ort/', import.meta.url).href;
    if (ort.env.webgpu) ort.env.webgpu.powerPreference = 'high-performance';

    const modelUrl = new URL(`../models/nca_starry_${this.size}.onnx`, import.meta.url).href;
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
    return { backend, loadMs: this.loadMs };
  }

  _allocate() {
    const n = this.size * this.size;
    this.stateData = new Float32Array(12 * n);
    this.masksData = new Float32Array(this.stepsPerRun * n);
    this.masksTensor = new ort.Tensor(
      'float32',
      this.masksData,
      [this.stepsPerRun, 1, this.size, this.size],
    );
    this.steps = 0;
    this.version += 1;
  }

  reset() {
    this.stateData.fill(0);
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
      [1, 12, this.size, this.size],
    );

    const result = await this.session.run({
      state: stateTensor,
      masks: this.masksTensor,
    });

    // A reset or a fresh wound arrived while inference was in flight. The result
    // belongs to the old state and must not resurrect erased cells.
    if (versionAtStart !== this.version) return false;

    this.stateData = result.state_out.data;
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
    radius = clamp(radius, 0.75, this.size * 0.4);
    feather = Math.max(feather, 0.4);
    const n = this.size;
    const x0 = Math.max(0, Math.floor(cx - radius - feather));
    const x1 = Math.min(n - 1, Math.ceil(cx + radius + feather));
    const y0 = Math.max(0, Math.floor(cy - radius - feather));
    const y1 = Math.min(n - 1, Math.ceil(cy + radius + feather));
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
        const pixel = y * n + x;
        for (let channel = 0; channel < 12; channel += 1) {
          const index = channel * n * n + pixel;
          this.stateData[index] *= keep;
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
  }

  get memoryBytes() {
    return this.stateData ? this.stateData.byteLength : 0;
  }
}

export { ort };

