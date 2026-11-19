const VERTEX_SHADER = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPosition;
out vec2 vUv;
void main() {
  vUv = aPosition * 0.5 + 0.5;
  gl_Position = vec4(aPosition, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler2D;

in vec2 vUv;
out vec4 outColor;

uniform sampler2D uSky;
uniform sampler2D uBrush;
uniform vec2 uResolution;
uniform float uSize;
uniform float uTime;

float luminance(vec3 color) {
  return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

vec2 coverUv(vec2 uv) {
  float aspect = uResolution.x / max(uResolution.y, 1.0);
  vec2 p = uv;
  if (aspect > 1.0) {
    p.y = 0.5 + (uv.y - 0.5) / aspect;
  } else {
    p.x = 0.5 + (uv.x - 0.5) * aspect;
  }
  return p;
}

float hash21(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}

void main() {
  vec2 uv = coverUv(vUv);
  vec2 texel = 1.0 / vec2(uSize);
  vec3 raw = clamp(texture(uSky, uv).rgb, 0.0, 1.0);
  float matter = texture(uSky, uv).a;
  float micro = matter;

  float l00 = luminance(texture(uSky, uv + vec2(-1.0, -1.0) * texel).rgb);
  float l01 = luminance(texture(uSky, uv + vec2( 0.0, -1.0) * texel).rgb);
  float l02 = luminance(texture(uSky, uv + vec2( 1.0, -1.0) * texel).rgb);
  float l10 = luminance(texture(uSky, uv + vec2(-1.0,  0.0) * texel).rgb);
  float l11 = luminance(raw);
  float l12 = luminance(texture(uSky, uv + vec2( 1.0,  0.0) * texel).rgb);
  float l20 = luminance(texture(uSky, uv + vec2(-1.0,  1.0) * texel).rgb);
  float l21 = luminance(texture(uSky, uv + vec2( 0.0,  1.0) * texel).rgb);
  float l22 = luminance(texture(uSky, uv + vec2( 1.0,  1.0) * texel).rgb);

  vec2 grad = vec2(
    (l10 + 2.0 * l11 + l12) - (l00 + 2.0 * l01 + l02),
    (l01 + 2.0 * l11 + l21) - (l00 + 2.0 * l10 + l20)
  ) * 0.125;
  float gradLength = length(grad);
  vec2 tangent = normalize(vec2(-grad.y, grad.x) + vec2(1e-5, 0.0));
  vec2 normal = vec2(-tangent.y, tangent.x);
  float flow = smoothstep(0.008, 0.15, gradLength);

  vec2 brushCoord = uv - 0.5;
  vec2 brushA = vec2(dot(brushCoord, tangent) * 3.8, dot(brushCoord, normal) * 10.0) + 0.5;
  vec2 brushB = vec2(dot(brushCoord, tangent) * 11.0, dot(brushCoord, normal) * 25.0) + 0.5;
  vec4 brushLow = texture(uBrush, brushA);
  vec4 brushHigh = texture(uBrush, brushB);

  float grain = brushLow.r * 0.62 + brushHigh.r * 0.38;
  float ridge = brushLow.g * 0.58 + brushHigh.g * 0.42;
  float bristle = brushLow.b * 0.54 + brushHigh.b * 0.46;
  float mottle = brushLow.a * 0.7 + brushHigh.a * 0.3;

  vec3 col = pow(raw, vec3(0.72));
  col = (col - 0.5) * 1.13 + 0.5;
  float luma = luminance(col);

  vec3 cool = col * vec3(0.76, 0.93, 1.14);
  vec3 warm = col * vec3(1.17, 1.05, 0.77) + vec3(0.025, 0.018, 0.0);
  col = mix(cool, warm, smoothstep(0.34, 0.82, luma));

  float relief = (ridge * 0.62 + micro * 0.38);
  col *= 0.86 + 0.28 * relief + 0.11 * grain + 0.08 * bristle;
  col += vec3(0.07, 0.10, 0.08) * pow(relief, 2.2) * (0.35 + flow * 0.65);

  float hx = dFdx(relief) * 36.0;
  float hy = dFdy(relief) * 36.0;
  vec3 n = normalize(vec3(-hx, -hy, 0.62));
  vec3 lightDir = normalize(vec3(-0.42, 0.62, 0.78));
  float diffuse = 0.78 + 0.22 * max(dot(n, lightDir), 0.0);
  float specular = pow(max(dot(reflect(-lightDir, n), vec3(0.0, 0.0, 1.0)), 0.0), 22.0);
  col *= diffuse;
  col += vec3(0.95, 0.84, 0.58) * specular * (0.07 + 0.17 * smoothstep(0.35, 0.88, luma));

  col = mix(vec3(0.002, 0.006, 0.011), col, smoothstep(0.015, 0.20, matter));

  float edgeGlow = smoothstep(0.006, 0.09, gradLength) * smoothstep(0.13, 0.45, luma);
  col += vec3(0.10, 0.16, 0.18) * edgeGlow * 0.25;
  col = mix(vec3(luma), col, 1.17);

  float vignetteDistance = length((vUv - 0.5) * vec2(1.0, 0.78));
  float vignette = 1.0 - smoothstep(0.28, 0.84, vignetteDistance);
  col *= mix(0.62, 1.0, vignette);
  col += vec3(0.018, 0.028, 0.038) * (1.0 - smoothstep(0.0, 0.65, luma));

  float grainNoise = hash21(gl_FragCoord.xy + fract(uTime * 0.07) * 397.0) - 0.5;
  col += grainNoise * 0.022;
  col = pow(max(col, 0.0), vec3(0.96));
  col = clamp(col, 0.0, 1.0);

  outColor = vec4(col, 1.0);
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

export class SkyRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = null;
    this.program = null;
    this.skyTexture = null;
    this.brushTexture = null;
    this.vao = null;
    this.pixels = null;
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
      desynchronized: true,
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
      size: gl.getUniformLocation(this.program, 'uSize'),
      time: gl.getUniformLocation(this.program, 'uTime'),
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

  setSimulationSize(size) {
    this.size = size;
    this.pixels = new Uint8ClampedArray(size * size * 4);
  }

  uploadState(stateData, size) {
    if (size !== this.size || !this.pixels) this.setSimulationSize(size);
    const gl = this.gl;
    const pixels = this.pixels;
    const n2 = size * size;
    for (let i = 0; i < n2; i += 1) {
      const p = i * 4;
      pixels[p] = (stateData[i] + 0.5) * 255;
      pixels[p + 1] = (stateData[n2 + i] + 0.5) * 255;
      pixels[p + 2] = (stateData[n2 * 2 + i] + 0.5) * 255;
      pixels[p + 3] = Math.min(1, Math.abs(stateData[n2 * 3 + i]) * 2.0) * 255;
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.skyTexture);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      size,
      size,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      pixels,
    );
  }

  render(timeSeconds) {
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
    gl.uniform1f(this.locations.size, this.size);
    gl.uniform1f(this.locations.time, timeSeconds);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
