const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

// Stable impasto material pass. Every term here is either static in screen/UV
// space or driven by the (temporally smoothed) CA state, so consecutive frames
// never shimmer: all visible motion comes from the NCA itself.
const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSky;
uniform sampler2D uBrush;
uniform vec2 uResolution;
uniform vec2 uCaSize;
uniform float uContain;
uniform float uContrast;
uniform float uSat;
uniform float uRelief;
uniform float uSpec;
uniform float uEdge;
uniform float uBlueShift;

float luminance(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Screen UV -> CA UV. "cover" fills the window (square texture mode);
// "contain" fits the whole painting inside the window (target modes).
vec2 mapUv(vec2 uv) {
  float sa = uResolution.x / max(uResolution.y, 1.0);
  float ca = uCaSize.x / max(uCaSize.y, 1.0);
  vec2 p = uv - 0.5;
  if (uContain > 0.5) {
    if (sa > ca) p.x *= sa / ca; else p.y *= ca / sa;
  } else {
    if (sa > ca) p.y *= ca / sa; else p.x *= sa / ca;
  }
  return p + 0.5;
}

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

float vnoise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

float potential(vec2 p) {
  return vnoise(p) + 0.5 * vnoise(p * 2.03 + 17.7);
}

// Static, divergence-free stroke direction field (curl of a noise potential).
// It never changes with time, so brush fibres do not rotate frame to frame.
vec2 strokeDir(vec2 uv) {
  float e = 0.22;
  vec2 q = uv * 2.6 + 3.1;
  vec2 curlA = vec2(
    potential(q + vec2(0.0, e)) - potential(q - vec2(0.0, e)),
    -(potential(q + vec2(e, 0.0)) - potential(q - vec2(e, 0.0)))
  );
  vec2 q2 = uv * 6.5 + 41.7;
  vec2 curlB = vec2(
    potential(q2 + vec2(0.0, e)) - potential(q2 - vec2(0.0, e)),
    -(potential(q2 + vec2(e, 0.0)) - potential(q2 - vec2(e, 0.0)))
  );
  vec2 dir = curlA * 0.72 + curlB * 0.28;
  return normalize(dir + vec2(1e-4, 1e-4));
}

void main() {
  vec2 uv = mapUv(vUv);

  // Letterbox around the contained painting: deep night, static grain only.
  if (uContain > 0.5 && (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0)) {
    vec3 night = vec3(0.004, 0.008, 0.016);
    night += (hash21(gl_FragCoord.xy) - 0.5) * 0.012;
    outColor = vec4(night, 1.0);
    return;
  }

  vec4 sky = texture(uSky, uv);
  vec3 raw = clamp(sky.rgb, 0.0, 1.0);
  float matter = sky.a;

  vec2 dir = strokeDir(uv);
  vec2 perp = vec2(-dir.y, dir.x);
  vec2 brushA = vec2(dot(uv, dir) * 3.6, dot(uv, perp) * 9.5);
  vec2 brushB = vec2(dot(uv, dir) * 6.5, dot(uv, perp) * 14.0) + 0.37;
  vec4 brushLow = texture(uBrush, brushA);
  vec4 brushHigh = texture(uBrush, brushB);

  float grain = brushLow.r * 0.62 + brushHigh.r * 0.38;
  float ridge = brushLow.g * 0.58 + brushHigh.g * 0.42;
  float bristle = brushLow.b * 0.54 + brushHigh.b * 0.46;
  float relief = ridge * 0.72 + grain * 0.18 + bristle * 0.10;

  // Gentle grade that keeps the trained palette (cobalt / ultramarine / cream).
  vec3 col = (raw - 0.5) * uContrast + 0.5;
  float luma = luminance(col);
  col = mix(vec3(luma), col, uSat);
  col = mix(col, col * vec3(0.86, 0.95, 1.12), smoothstep(0.45, 0.12, luma) * uBlueShift);
  col += vec3(0.045, 0.032, 0.004) * smoothstep(0.62, 0.95, luma);

  // Impasto thickness: static fibre relief modulates pigment density.
  col *= 1.0 - uRelief * 0.5 + uRelief * relief;

  // Stable relief lighting (relief is static in UV, so derivatives are stable).
  float hx = dFdx(relief) * 18.0;
  float hy = dFdy(relief) * 18.0;
  vec3 n = normalize(vec3(-hx, -hy, 0.75));
  vec3 lightDir = normalize(vec3(-0.42, 0.62, 0.78));
  float diffuse = 0.90 + 0.10 * max(dot(n, lightDir), 0.0);
  float specular = pow(max(dot(reflect(-lightDir, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
  col *= diffuse;
  col += vec3(0.95, 0.86, 0.62) * specular * uSpec * (0.4 + 0.6 * smoothstep(0.4, 0.9, luma));

  // Dark contours between impasto strokes, driven only by the live CA field.
  vec2 tx = 1.0 / uCaSize;
  float lc = luminance(texture(uSky, uv + vec2(tx.x, 0.0)).rgb);
  float lw = luminance(texture(uSky, uv - vec2(tx.x, 0.0)).rgb);
  float ln = luminance(texture(uSky, uv + vec2(0.0, tx.y)).rgb);
  float ls = luminance(texture(uSky, uv - vec2(0.0, tx.y)).rgb);
  float edge = smoothstep(0.03, 0.16, length(vec2(lc - lw, ln - ls)));
  col *= 1.0 - uEdge * edge;

  // Wounds (zeroed cells) read as deep night; smoothed activity avoids blinking.
  col = mix(vec3(0.004, 0.008, 0.016), col, smoothstep(0.02, 0.22, matter));

  float vignetteDistance = length((vUv - 0.5) * vec2(1.0, 0.78));
  float vignette = 1.0 - smoothstep(0.34, 0.92, vignetteDistance);
  col *= mix(0.82, 1.0, vignette);

  // Static film grain: identical every frame, no temporal shimmer.
  col += (hash21(gl_FragCoord.xy) - 0.5) * 0.012;
  outColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}`;

function compileShader(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader);
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed: ${log}`);
  }
  return shader;
}

function createProgram(gl, vertexSource, fragmentSource) {
  const program = gl.createProgram();
  const vertex = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program);
    gl.deleteProgram(program);
    throw new Error(`Program link failed: ${log}`);
  }
  return program;
}

function createTexture(gl) {
  const texture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return texture;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.decoding = 'async';
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Could not load image: ${url}`));
    image.src = url;
  });
}

const GRADE_PRESETS = {
  // The texture model invents its own palette; push it toward cobalt/cream.
  texture: { contrast: 1.18, sat: 1.26, relief: 0.10, spec: 0.03, edge: 0.26, blueShift: 0.45 },
  // The target model already IS the painting; grade it gently, keep it legible.
  target: { contrast: 1.06, sat: 1.12, relief: 0.10, spec: 0.03, edge: 0.18, blueShift: 0.22 },
};

export class SkyRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.program = null;
    this.skyTexture = null;
    this.brushTexture = null;
    this.vao = null;
    this.pixels = null;
    this.activity = null;
    this.scar = null;
    this.lastUpload = 0;
    this.timeScale = 1;
    this.kind = 'texture';
    this.grade = { ...GRADE_PRESETS.texture };
    this.caWidth = 256;
    this.caHeight = 256;
    this.size = 256;
    this.width = 1;
    this.height = 1;
    this.locations = {};
  }

  async init() {
    const gl = this.canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;
    this.program = createProgram(gl, VERTEX_SHADER, FRAGMENT_SHADER);
    gl.useProgram(this.program);

    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    this.locations = {
      sky: gl.getUniformLocation(this.program, 'uSky'),
      brush: gl.getUniformLocation(this.program, 'uBrush'),
      resolution: gl.getUniformLocation(this.program, 'uResolution'),
      caSize: gl.getUniformLocation(this.program, 'uCaSize'),
      contain: gl.getUniformLocation(this.program, 'uContain'),
      contrast: gl.getUniformLocation(this.program, 'uContrast'),
      sat: gl.getUniformLocation(this.program, 'uSat'),
      relief: gl.getUniformLocation(this.program, 'uRelief'),
      spec: gl.getUniformLocation(this.program, 'uSpec'),
      edge: gl.getUniformLocation(this.program, 'uEdge'),
      blueShift: gl.getUniformLocation(this.program, 'uBlueShift'),
    };

    this.skyTexture = createTexture(gl);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([3, 8, 13, 255]),
    );

    this.brushTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    const brushImage = await loadImage(new URL('../assets/textures/brush.png', import.meta.url).href);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, brushImage);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.generateMipmap(gl.TEXTURE_2D);

    gl.useProgram(this.program);
    gl.uniform1i(this.locations.sky, 0);
    gl.uniform1i(this.locations.brush, 1);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const maxPixels = 4_200_000;
    let dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rawPixels = Math.max(rect.width, 1) * Math.max(rect.height, 1) * dpr * dpr;
    if (rawPixels > maxPixels) dpr *= Math.sqrt(maxPixels / rawPixels);
    this.width = Math.max(1, Math.round(rect.width * dpr));
    this.height = Math.max(1, Math.round(rect.height * dpr));
    if (this.canvas.width !== this.width || this.canvas.height !== this.height) {
      this.canvas.width = this.width;
      this.canvas.height = this.height;
    }
    this.gl.viewport(0, 0, this.width, this.height);
  }

  setKind(kind) {
    this.kind = kind === 'target' ? 'target' : 'texture';
    this.grade = { ...GRADE_PRESETS[this.kind] };
  }

  setSimulationSize(width, height = width) {
    this.caWidth = width;
    this.caHeight = height;
    this.size = width;
    this.pixels = new Uint8ClampedArray(width * height * 4);
    this.activity = new Float32Array(width * height);
    this.scar = new Float32Array(width * height);
    this.lastUpload = 0;
  }

  resetActivity() {
    if (this.activity) this.activity.fill(0);
    if (this.scar) this.scar.fill(0);
    this.lastUpload = 0;
  }

  // The on-screen rectangle occupied by the CA image, in client pixels.
  // Single source of truth for both rendering and pointer mapping.
  caScreenRect() {
    const rect = this.canvas.getBoundingClientRect();
    const screenAspect = rect.width / Math.max(rect.height, 1);
    const caAspect = this.caWidth / Math.max(this.caHeight, 1);
    const contain = this.kind === 'target';
    let w;
    let h;
    if (contain ? screenAspect > caAspect : screenAspect < caAspect) {
      h = rect.height;
      w = h * caAspect;
    } else {
      w = rect.width;
      h = w / caAspect;
    }
    return {
      left: rect.left + (rect.width - w) / 2,
      top: rect.top + (rect.height - h) / 2,
      width: w,
      height: h,
    };
  }

  // Client point -> CA cell coordinates (y down = state row). Null when the
  // point falls in the letterbox around a contained painting.
  screenToCa(clientX, clientY) {
    const r = this.caScreenRect();
    const u = (clientX - r.left) / Math.max(r.width, 1);
    const v = (clientY - r.top) / Math.max(r.height, 1);
    if (this.kind === 'target' && (u < 0 || u > 1 || v < 0 || v > 1)) return null;
    return { x: u * this.caWidth, y: v * this.caHeight };
  }

  // Paint a decaying scar over freshly erased cells. This is a material-layer
  // memory of the wound (not motion): it holds the tear in deep night for a
  // moment so the NCA's primer tissue reads as darkness, then fades while
  // real pigment regrows. Decay is frozen while the sky is paused.
  markWound(cx, cy, radius, soft = 1.5) {
    if (!this.scar) return;
    const w = this.caWidth;
    const h = this.caHeight;
    const outer = radius + soft;
    const x0 = Math.max(0, Math.floor(cx - outer));
    const x1 = Math.min(w - 1, Math.ceil(cx + outer));
    const y0 = Math.max(0, Math.floor(cy - outer));
    const y1 = Math.min(h - 1, Math.ceil(cy + outer));
    for (let y = y0; y <= y1; y += 1) {
      for (let x = x0; x <= x1; x += 1) {
        const d = Math.hypot(x - cx, y - cy);
        if (d > outer) continue;
        const fall = d <= radius ? 1 : 1 - (d - radius) / soft;
        const i = y * w + x;
        if (fall > this.scar[i]) this.scar[i] = fall;
      }
    }
  }

  uploadState(stateData, width, height = width) {
    if (width !== this.caWidth || height !== this.caHeight || !this.pixels) {
      this.setSimulationSize(width, height);
    }
    const gl = this.gl;
    const pixels = this.pixels;
    const activity = this.activity;
    const scar = this.scar;
    const now = performance.now();
    const dt = this.lastUpload ? Math.min(100, now - this.lastUpload) : 0;
    this.lastUpload = now;
    const decay = Math.exp((-dt / 900) * (this.timeScale ?? 1));
    const n2 = width * height;
    const isTarget = this.kind === 'target';
    for (let i = 0; i < n2; i += 1) {
      const p = i * 4;
      let r;
      let g;
      let b;
      let target;
      if (isTarget) {
        // RGBA lives directly in [0,1]; alpha IS the living-matter field.
        r = stateData[i];
        g = stateData[n2 + i];
        b = stateData[n2 * 2 + i];
        target = Math.min(1, Math.max(0, stateData[n2 * 3 + i]));
      } else {
        r = stateData[i] + 0.5;
        g = stateData[n2 + i] + 0.5;
        b = stateData[n2 * 2 + i] + 0.5;
        // Activity = how far a cell is from the erased zero state. RGB distance
        // from flat grey plus hidden-channel magnitude; temporally smoothed so
        // asynchronous CA updates can never blink cells on/off between frames.
        const luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const sat = Math.max(r, g, b) - Math.min(r, g, b);
        const hmag = Math.max(
          Math.abs(stateData[n2 * 3 + i]),
          Math.abs(stateData[n2 * 4 + i]),
          Math.abs(stateData[n2 * 5 + i]),
        );
        target = Math.min(1, Math.max(hmag * 1.7, sat * 1.6 + Math.abs(luma - 0.5) * 2.2));
      }
      pixels[p] = Math.min(1, Math.max(0, r)) * 255;
      pixels[p + 1] = Math.min(1, Math.max(0, g)) * 255;
      pixels[p + 2] = Math.min(1, Math.max(0, b)) * 255;
      activity[i] += (target - activity[i]) * 0.45;
      scar[i] *= decay;
      pixels[p + 3] = activity[i] * (1 - scar[i]) * 255;
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.skyTexture);
    // Flip on upload so state row 0 appears at the TOP of the screen, matching
    // the pointer mapping (screenToCa: y down = state row).
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  }

  render() {
    const gl = this.gl;
    if (!gl) return;
    gl.clearColor(0.01, 0.03, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.skyTexture);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.brushTexture);

    gl.uniform2f(this.locations.resolution, this.width, this.height);
    gl.uniform2f(this.locations.caSize, this.caWidth, this.caHeight);
    gl.uniform1f(this.locations.contain, this.kind === 'target' ? 1 : 0);
    gl.uniform1f(this.locations.contrast, this.grade.contrast);
    gl.uniform1f(this.locations.sat, this.grade.sat);
    gl.uniform1f(this.locations.relief, this.grade.relief);
    gl.uniform1f(this.locations.spec, this.grade.spec);
    gl.uniform1f(this.locations.edge, this.grade.edge);
    gl.uniform1f(this.locations.blueShift, this.grade.blueShift ?? 0.45);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}