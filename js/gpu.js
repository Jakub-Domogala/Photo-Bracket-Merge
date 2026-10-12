// Ta sama obróbka co w pipeline.js / lens.js / previewLens.js, ale na karcie graficznej (WebGPU):
// korekcja obiektywu, auto-ekspozycja i fuzja Mertensa. Dekodowanie, pomiar korekcji z podglądu,
// auto WB i wykończenie zostają na CPU. Gdy WebGPU nie ma albo coś się nie uda, merge.js liczy
// wszystko na CPU (wynik GPU różni się od CPU tylko zaokrągleniami float32).

import { fillScale } from "./lens.js";

const WG2 = 16;  // workgroup 16x16 dla jąder 2D
const WG1 = 256; // workgroup dla jąder 1D
const MAX_X_GROUPS = 1024;

const COMMON = /* wgsl */ `
fn refl101(p: i32, len: i32) -> i32 {
  if (len == 1) { return 0; }
  var q = p;
  loop {
    if (q >= 0 && q < len) { break; }
    if (q < 0) { q = -q; } else { q = 2 * len - q - 2; }
  }
  return q;
}
fn reflect(p: i32, len: i32) -> i32 {
  var q = p;
  loop {
    if (q >= 0 && q < len) { break; }
    if (q < 0) { q = -q - 1; } else { q = 2 * len - q - 1; }
  }
  return q;
}
fn cubicW(t: f32) -> vec4<f32> {
  let A = -0.75;
  let w0 = ((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A;
  let w1 = ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0;
  let u = 1.0 - t;
  let w2 = ((A + 2.0) * u - (A + 3.0)) * u * u + 1.0;
  return vec4<f32>(w0, w1, w2, 1.0 - w0 - w1 - w2);
}
fn toSrgb(v: f32) -> f32 {
  let x = clamp(v, 0.0, 1.0);
  return select(1.055 * pow(x, 1.0 / 2.4) - 0.055, 12.92 * x, x <= 0.0031308);
}
fn lumAt(b: f32, g: f32, r: f32) -> f32 { return 0.0722 * b + 0.7152 * g + 0.2126 * r; }
`;

// próbkowanie jak cv2.remap(INTER_CUBIC, BORDER_REFLECT) z bufora `buf`
const sampler = (buf) => /* wgsl */ `
fn sample_${buf}(mx0: f32, my0: f32, w: i32, h: i32) -> f32 {
  let mx = clamp(mx0, -f32(w), 2.0 * f32(w));
  let my = clamp(my0, -f32(h), 2.0 * f32(h));
  let fx = floor(mx);
  let fy = floor(my);
  let ix = i32(fx);
  let iy = i32(fy);
  let wx = cubicW(mx - fx);
  let wy = cubicW(my - fy);
  let x0 = reflect(ix - 1, w); let x1 = reflect(ix, w); let x2 = reflect(ix + 1, w); let x3 = reflect(ix + 2, w);
  var acc = 0.0;
  for (var m = 0; m < 4; m++) {
    let row = reflect(iy - 1 + m, h) * w;
    acc += wy[m] * (wx.x * ${buf}[row + x0] + wx.y * ${buf}[row + x1] + wx.z * ${buf}[row + x2] + wx.w * ${buf}[row + x3]);
  }
  return acc;
}`;

// piksel (X, Y) wyniku cv2.pyrUp z bufora `buf` (sw x sh), brzegi jak w pyrUp_ OpenCV
const upsampler = (buf) => /* wgsl */ `
fn hrow_${buf}(row: i32, x: i32, px: i32, sw: i32) -> f32 {
  let xl = select(x - 1, select(0, 1, sw > 1), x == 0);
  let xr = min(x + 1, sw - 1);
  if (px == 0) { return ${buf}[row + xl] + 6.0 * ${buf}[row + x] + ${buf}[row + xr]; }
  return 4.0 * (${buf}[row + x] + ${buf}[row + xr]);
}
fn up_${buf}(X: i32, Y: i32, sw: i32, sh: i32) -> f32 {
  let x = X >> 1u;
  let y = Y >> 1u;
  let px = X & 1;
  let b = y * sw;
  let c = (refl101(2 * y + 2, 2 * sh) >> 1u) * sw;
  if ((Y & 1) == 1) { return 4.0 * (hrow_${buf}(b, x, px, sw) + hrow_${buf}(c, x, px, sw)) / 64.0; }
  let a = (refl101(2 * y - 2, 2 * sh) >> 1u) * sw;
  return (hrow_${buf}(a, x, px, sw) + 6.0 * hrow_${buf}(b, x, px, sw) + hrow_${buf}(c, x, px, sw)) / 64.0;
}`;

const ID2 = /* wgsl */ `@compute @workgroup_size(${WG2}, ${WG2})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= i32(p.w) || y >= i32(p.h)) { return; }
  let i = y * i32(p.w) + x;`;
const ID1 = /* wgsl */ `@compute @workgroup_size(${WG1})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let i = (wid.y * nwg.x + wid.x) * ${WG1}u + li;
  if (i >= p.n) { return; }`;

const rw = (n, name) => `@group(0) @binding(${n}) var<storage, read_write> ${name}: array<f32>;`;
const ro = (n, name) => `@group(0) @binding(${n}) var<storage, read> ${name}: array<f32>;`;
const uni = (fields) => `struct P { ${fields.map(([t, k]) => `${k}: ${t},`).join(" ")} }
@group(0) @binding(0) var<uniform> p: P;`;

const INTERP = /* wgsl */ `
fn interp(off: u32, n: u32, r: f32) -> f32 {
  if (r <= 0.0) { return curve[off]; }
  if (r >= 1.0) { return curve[off + n - 1u]; }
  let q = r * f32(n - 1u);
  let k = u32(floor(q));
  return curve[off + k] + (q - f32(k)) * (curve[off + k + 1u] - curve[off + k]);
}`;

// pola uniformów każdego jądra (typ, nazwa) — kolejność = układ w buforze
const KERNELS = {
  sonyVignette: {
    fields: [["u32", "w"], ["u32", "h"], ["u32", "nk"], ["f32", "cx"], ["f32", "cy"], ["f32", "R"]],
    body: (u) => `${u}
${rw(1, "b")} ${rw(2, "g")} ${rw(3, "r")} ${ro(4, "curve")}
${INTERP}
${ID2}
  let dx = f32(x) - p.cx;
  let dy = f32(y) - p.cy;
  let f = interp(0u, p.nk, sqrt(dx * dx + dy * dy) / p.R);
  b[i] = b[i] / f; g[i] = g[i] / f; r[i] = r[i] / f;
}`,
  },
  sonyRemap: {
    fields: [["u32", "w"], ["u32", "h"], ["u32", "nk"], ["f32", "cx"], ["f32", "cy"], ["f32", "R"]],
    body: (u) => `${u}
${ro(1, "sb")} ${ro(2, "sg")} ${ro(3, "sr")} ${rw(4, "db")} ${rw(5, "dg")} ${rw(6, "dr")} ${ro(7, "curve")}
${INTERP} ${sampler("sb")} ${sampler("sg")} ${sampler("sr")}
${ID2}
  let dx = f32(x) - p.cx;
  let dy = f32(y) - p.cy;
  let rad = sqrt(dx * dx + dy * dy) / p.R;
  let w = i32(p.w);
  let h = i32(p.h);
  let kb = interp(0u, p.nk, rad);
  let kg = interp(p.nk, p.nk, rad);
  let kr = interp(2u * p.nk, p.nk, rad);
  db[i] = max(sample_sb(p.cx + dx * kb, p.cy + dy * kb, w, h), 0.0);
  dg[i] = max(sample_sg(p.cx + dx * kg, p.cy + dy * kg, w, h), 0.0);
  dr[i] = max(sample_sr(p.cx + dx * kr, p.cy + dy * kr, w, h), 0.0);
}`,
  },
  previewRemap: {
    fields: [["u32", "w"], ["u32", "h"], ["u32", "sw"], ["u32", "sh"], ["u32", "ns"], ["f32", "cpx"], ["f32", "cpy"],
      ["f32", "R"], ["f32", "scale"], ["f32", "b1"], ["f32", "b2"], ["f32", "s0"], ["f32", "s1"], ["f32", "s2"],
      ["f32", "s3"], ["f32", "s4"], ["f32", "s5"], ["f32", "s6"], ["f32", "s7"]],
    body: (u) => `${u}
${ro(1, "sb")} ${ro(2, "sg")} ${ro(3, "sr")} ${rw(4, "db")} ${rw(5, "dg")} ${rw(6, "dr")}
${sampler("sb")} ${sampler("sg")} ${sampler("sr")}
${ID2}
  let px = p.cpx + (f32(x) - (f32(p.w) - 1.0) / 2.0) / p.scale;
  let py = p.cpy + (f32(y) - (f32(p.h) - 1.0) / 2.0) / p.scale;
  let dx = (px - p.cpx) / p.R;
  let dy = (py - p.cpy) / p.R;
  let r2 = dx * dx + dy * dy;
  var sol = array<f32, 8>(p.s0, p.s1, p.s2, p.s3, p.s4, p.s5, p.s6, p.s7);
  var s = 0.0;
  var pw = 1.0;
  for (var j = 2u; j < p.ns; j++) { s += sol[j] * pw; pw *= r2; }
  let mx = p.s0 + dx * s;
  let my = p.s1 + dy * s;
  let gain = clamp(exp(-(p.b1 * r2 + p.b2 * r2 * r2)), 1.0, ${4.0.toFixed(1)});
  let sw = i32(p.sw);
  let sh = i32(p.sh);
  db[i] = max(sample_sb(mx, my, sw, sh) * gain, 0.0);
  dg[i] = max(sample_sg(mx, my, sw, sh) * gain, 0.0);
  dr[i] = max(sample_sr(mx, my, sw, sh) * gain, 0.0);
}`,
  },
  logSum: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${ro(1, "b")} ${ro(2, "g")} ${ro(3, "r")} ${rw(4, "part")}
var<workgroup> acc: array<f32, ${WG1}>;
@compute @workgroup_size(${WG1})
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let gi = wid.y * nwg.x + wid.x;
  let i = gi * ${WG1}u + li;
  var v = 0.0;
  if (i < p.n) { v = log(max(lumAt(b[i], g[i], r[i]), 1e-4)); }
  acc[li] = v;
  workgroupBarrier();
  for (var s = ${WG1 / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { acc[li] += acc[li + s]; }
    workgroupBarrier();
  }
  if (li == 0u) { part[gi] = acc[0]; }
}`,
  },
  srgb: {
    fields: [["u32", "n"], ["f32", "gain"]],
    body: (u) => `${u}
${rw(1, "b")} ${rw(2, "g")} ${rw(3, "r")}
${ID1}
  b[i] = toSrgb(b[i] * p.gain); g[i] = toSrgb(g[i] * p.gain); r[i] = toSrgb(r[i] * p.gain);
}`,
  },
  weight: {
    fields: [["u32", "w"], ["u32", "h"]],
    body: (u) => `${u}
${ro(1, "b")} ${ro(2, "g")} ${ro(3, "r")} ${rw(4, "W")}
fn grayAt(k: i32) -> f32 { return 0.299 * b[k] + 0.587 * g[k] + 0.114 * r[k]; }
${ID2}
  let w = i32(p.w);
  let h = i32(p.h);
  let lap = grayAt(y * w + refl101(x - 1, w)) + grayAt(y * w + refl101(x + 1, w))
    + grayAt(refl101(y - 1, h) * w + x) + grayAt(refl101(y + 1, h) * w + x) - 4.0 * grayAt(i);
  let c = vec3<f32>(b[i], g[i], r[i]);
  let mean = (c.x + c.y + c.z) / 3.0;
  let d = c - vec3<f32>(mean);
  let e = c - vec3<f32>(0.5);
  W[i] = abs(lap) * sqrt(dot(d, d)) * exp(-dot(e, e) / 0.08) + 1e-12;
}`,
  },
  addTo: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${rw(1, "sum")} ${ro(2, "W")}
${ID1}
  sum[i] = sum[i] + W[i];
}`,
  },
  divide: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${rw(1, "W")} ${ro(2, "sum")}
${ID1}
  W[i] = W[i] / sum[i];
}`,
  },
  pyrDown: {
    fields: [["u32", "w"], ["u32", "h"], ["u32", "sw"], ["u32", "sh"]],
    body: (u) => `${u}
${ro(1, "src")} ${rw(2, "dst")}
${ID2}
  var k = array<f32, 5>(1.0, 4.0, 6.0, 4.0, 1.0);
  let sw = i32(p.sw);
  let sh = i32(p.sh);
  var acc = 0.0;
  for (var j = 0; j < 5; j++) {
    let row = refl101(2 * y + j - 2, sh) * sw;
    var s = 0.0;
    for (var m = 0; m < 5; m++) { s += k[m] * src[row + refl101(2 * x + m - 2, sw)]; }
    acc += k[j] * s;
  }
  dst[i] = acc / 256.0;
}`,
  },
  lapAcc: { // acc += (g - pyrUp(gn)) * wl
    fields: [["u32", "w"], ["u32", "h"], ["u32", "sw"], ["u32", "sh"]],
    body: (u) => `${u}
${rw(1, "acc")} ${ro(2, "gl")} ${ro(3, "gn")} ${ro(4, "wl")}
${upsampler("gn")}
${ID2}
  acc[i] += (gl[i] - up_gn(x, y, i32(p.sw), i32(p.sh))) * wl[i];
}`,
  },
  fmaTop: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${rw(1, "acc")} ${ro(2, "gl")} ${ro(3, "wl")}
${ID1}
  acc[i] += gl[i] * wl[i];
}`,
  },
  addUp: { // d += pyrUp(s)
    fields: [["u32", "w"], ["u32", "h"], ["u32", "sw"], ["u32", "sh"]],
    body: (u) => `${u}
${rw(1, "d")} ${ro(2, "s")}
${upsampler("s")}
${ID2}
  d[i] += up_s(x, y, i32(p.sw), i32(p.sh));
}`,
  },
  logGray: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${ro(1, "b")} ${ro(2, "g")} ${ro(3, "r")} ${rw(4, "out")}
${ID1}
  out[i] = log(max((b[i] + g[i] + r[i]) / 3.0, 1e-6));
}`,
  },
  crop: {
    fields: [["u32", "w"], ["u32", "h"], ["u32", "sw"], ["u32", "x0"], ["u32", "y0"]],
    body: (u) => `${u}
${ro(1, "src")} ${rw(2, "dst")}
${ID2}
  dst[i] = src[(u32(y) + p.y0) * p.sw + u32(x) + p.x0];
}`,
  },
  clip: {
    fields: [["u32", "n"]],
    body: (u) => `${u}
${rw(1, "d")}
${ID1}
  d[i] = clamp(d[i], 0.0, 1.0);
}`,
  },
};

let devicePromise = null;

/** Urządzenie WebGPU albo null (brak WebGPU / adaptera). */
export function getGpu() {
  devicePromise ??= (async () => {
    try {
      const adapter = await globalThis.navigator?.gpu?.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) return null;
      const L = adapter.limits;
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
          maxBufferSize: L.maxBufferSize,
          maxStorageBuffersPerShaderStage: Math.min(L.maxStorageBuffersPerShaderStage, 8),
        },
      });
      device.lost.then(() => { devicePromise = null; });
      return new Gpu(device);
    } catch {
      return null;
    }
  })();
  return devicePromise;
}

class Gpu {
  #pipelines = new Map();
  #free = new Map();
  constructor(device) {
    this.device = device;
    this.encoder = null;
  }

  get maxFloats() { return Math.floor(this.device.limits.maxStorageBufferBindingSize / 4); }

  #pipeline(name) {
    if (!this.#pipelines.has(name)) {
      const k = KERNELS[name];
      const code = COMMON + k.body(uni(k.fields));
      this.#pipelines.set(name, this.device.createComputePipeline({
        layout: "auto",
        compute: { module: this.device.createShaderModule({ code }), entryPoint: "main" },
      }));
    }
    return this.#pipelines.get(name);
  }

  /** Bufor n floatów (z puli; zawartość nieokreślona). */
  buf(n) {
    const list = this.#free.get(n);
    if (list?.length) return list.pop();
    const b = this.device.createBuffer({ size: Math.max(n, 4) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    b.n = n;
    return b;
  }

  release(...bufs) {
    for (const b of bufs) {
      if (!b) continue;
      if (!this.#free.has(b.n)) this.#free.set(b.n, []);
      this.#free.get(b.n).push(b);
    }
  }

  /** Zwalnia całą pulę (pamięć karty). */
  trim() {
    for (const list of this.#free.values()) for (const b of list) b.destroy();
    this.#free.clear();
  }

  upload(arr) {
    // writeBuffer działa od razu w kolejce — najpierw wysyłamy zapisane polecenia, które mogą
    // jeszcze używać bufora wziętego z puli
    this.submit();
    const b = this.buf(arr.length);
    this.device.queue.writeBuffer(b, 0, arr);
    return b;
  }

  zero(b) { this.#enc().clearBuffer(b); }

  #enc() { return (this.encoder ??= this.device.createCommandEncoder()); }

  /** Dodaje wywołanie jądra; dims = [w, h] (2D) albo n (1D). */
  run(name, values, buffers, dims) {
    const k = KERNELS[name];
    const data = new ArrayBuffer(Math.max(16, Math.ceil(k.fields.length / 4) * 16));
    const dv = new DataView(data);
    k.fields.forEach(([t, key], j) => (t === "u32" ? dv.setUint32(j * 4, values[key], true) : dv.setFloat32(j * 4, values[key], true)));
    const ub = this.device.createBuffer({ size: data.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(ub, 0, data);
    const pipeline = this.#pipeline(name);
    const bind = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [ub, ...buffers].map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    const pass = this.#enc().beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bind);
    if (Array.isArray(dims)) {
      pass.dispatchWorkgroups(Math.ceil(dims[0] / WG2), Math.ceil(dims[1] / WG2));
    } else {
      const groups = Math.ceil(dims / WG1);
      pass.dispatchWorkgroups(Math.min(groups, MAX_X_GROUPS), Math.ceil(groups / MAX_X_GROUPS));
    }
    pass.end();
    (this.uniforms ??= []).push(ub);
  }

  submit() {
    if (this.encoder) this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    for (const ub of this.uniforms ?? []) ub.destroy();
    this.uniforms = [];
  }

  /** Kopiuje bufor do tablicy `out` (Float32Array, może być współdzielona). */
  async read(b, out) {
    const staging = this.device.createBuffer({ size: out.length * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    this.#enc().copyBufferToBuffer(b, 0, staging, 0, out.length * 4);
    this.submit();
    await staging.mapAsync(GPUMapMode.READ);
    out.set(new Float32Array(staging.getMappedRange()));
    staging.unmap();
    staging.destroy();
    return out;
  }
}

// --- kroki ---

function sonyLens(gpu, f, profile, w, h) {
  const cx = (w - 1) / 2, cy = (h - 1) / 2, R = Math.hypot(cx, cy);
  const falloff = Float32Array.from(profile.vignetting, (v) => 2 ** (0.5 - 2 ** (v * 2 ** -13 - 1)));
  const fb = gpu.upload(falloff);
  gpu.run("sonyVignette", { w, h, nk: falloff.length, cx, cy, R }, [f.b, f.g, f.r, fb], [w, h]);
  let base = Float32Array.from(profile.distortion, (d) => d * 2 ** -14 + 1);
  const fs = fillScale(base, cx, cy);
  base = base.map((v) => v * fs);
  const [caR, caB] = profile.ca;
  const nk = base.length;
  const curves = new Float32Array(3 * nk);
  curves.set(base.map((v, i) => v * (caB[i] * 2 ** -21 + 1)), 0);
  curves.set(base, nk);
  curves.set(base.map((v, i) => v * (caR[i] * 2 ** -21 + 1)), 2 * nk);
  const cb = gpu.upload(curves);
  const out = { b: gpu.buf(w * h), g: gpu.buf(w * h), r: gpu.buf(w * h) };
  gpu.run("sonyRemap", { w, h, nk, cx, cy, R }, [f.b, f.g, f.r, out.b, out.g, out.r, cb], [w, h]);
  gpu.release(f.b, f.g, f.r, fb, cb); // pula: bufory wrócą do użytku dopiero w kolejnych poleceniach
  return out;
}

function previewLens(gpu, f, geo, sw, sh) {
  const { width: w, height: h, cp, R, scale, sol, vig } = geo;
  if (sol.length > 8) throw new Error("za dużo wyrazów modelu obiektywu dla GPU");
  const s = Object.fromEntries(Array.from({ length: 8 }, (_, j) => [`s${j}`, sol[j] ?? 0]));
  const out = { b: gpu.buf(w * h), g: gpu.buf(w * h), r: gpu.buf(w * h) };
  gpu.run("previewRemap", { w, h, sw, sh, ns: sol.length, cpx: cp[0], cpy: cp[1], R, scale, b1: vig[0], b2: vig[1], ...s },
    [f.b, f.g, f.r, out.b, out.g, out.r], [w, h]);
  gpu.release(f.b, f.g, f.r);
  return out;
}

async function logAverage(gpu, f, n) {
  const groups = Math.ceil(n / WG1);
  const part = gpu.buf(groups);
  gpu.run("logSum", { n }, [f.b, f.g, f.r, part], n);
  const sums = await gpu.read(part, new Float32Array(groups));
  gpu.release(part);
  let s = 0;
  for (const v of sums) s += v;
  return Math.exp(s / n);
}

function pyramid(gpu, base, w, h, maxlevel) {
  const pyr = [{ buf: base, w, h }];
  for (let l = 0; l < maxlevel; l++) {
    const { buf, w: sw, h: sh } = pyr[l];
    const dw = (sw + 1) >> 1, dh = (sh + 1) >> 1;
    const dst = gpu.buf(dw * dh);
    gpu.run("pyrDown", { w: dw, h: dh, sw, sh }, [buf, dst], [dw, dh]);
    pyr.push({ buf: dst, w: dw, h: dh });
  }
  return pyr;
}

/**
 * Korekcja obiektywu + auto-ekspozycja + Mertens na GPU.
 * @param frames  liniowe klatki CPU {b, g, r, width, height} (nie są zmieniane)
 * @param lens    null | { profiles } (Sony) | { geo } (z podglądu)
 * @param align   async (log-szarości klatek, w, h) => plan z alignPlan() (js/align.js)
 * @param alloc   (n) => Float32Array na wynik
 * @returns       { result: { b, g, r } — sRGB 0..1 po fuzji (jak mergeMertens), width, height }
 */
export async function gpuMerge(gpu, frames, lens, key, align, alloc, progress = () => {}) {
  const { device } = gpu;
  device.pushErrorScope("out-of-memory");
  device.pushErrorScope("validation");
  const sw = frames[0].width, sh = frames[0].height;
  let w = sw, h = sh;
  if (lens?.geo) ({ width: w, height: h } = lens.geo);
  if (Math.max(sw * sh, w * h) > gpu.maxFloats) throw new Error("obraz za duży dla GPU");
  let n = w * h;
  try {
    // korekcja obiektywu
    let imgs = [];
    for (const [k, f] of frames.entries()) {
      let g = { b: gpu.upload(f.b), g: gpu.upload(f.g), r: gpu.upload(f.r) };
      if (lens?.profiles) g = sonyLens(gpu, g, lens.profiles[k], w, h);
      else if (lens?.geo) g = previewLens(gpu, g, lens.geo, sw, sh);
      gpu.submit();
      imgs.push(g);
    }

    // wyrównanie przesuniętych klatek: szarości liczy GPU, przesunięcia — CPU
    progress("Wyrównywanie klatek (GPU)");
    const grays = [];
    for (const g of imgs) {
      const lg = gpu.buf(n);
      gpu.run("logGray", { n }, [g.b, g.g, g.r, lg], n);
      grays.push(await gpu.read(lg, alloc(n)));
      gpu.release(lg);
    }
    const plan = await align(grays, w, h);
    if (plan?.shifts) {
      imgs = imgs.map((g, k) => {
        const [sx, sy] = plan.shifts[k];
        const out = {};
        for (const c of ["b", "g", "r"]) {
          out[c] = gpu.buf(plan.w * plan.h);
          gpu.run("crop", { w: plan.w, h: plan.h, sw: w, x0: plan.x0 + sx, y0: plan.y0 + sy }, [g[c], out[c]], [plan.w, plan.h]);
        }
        gpu.release(g.b, g.g, g.r);
        return out;
      });
      ({ w, h } = plan);
      n = w * h;
    }

    // auto-ekspozycja (wspólne wzmocnienie)
    progress("Auto-ekspozycja (GPU)");
    const levels = [];
    for (const g of imgs) levels.push(await logAverage(gpu, g, n));
    levels.sort((a, b) => a - b);
    const gain = Math.min(Math.max(key / levels[Math.floor(levels.length / 2)], 0.25), 32);
    for (const g of imgs) gpu.run("srgb", { n, gain }, [g.b, g.g, g.r], n);

    // wagi Mertensa
    const weights = imgs.map((g) => {
      const W = gpu.buf(n);
      gpu.run("weight", { w, h }, [g.b, g.g, g.r, W], [w, h]);
      return W;
    });
    const sum = gpu.buf(n);
    gpu.zero(sum);
    for (const W of weights) gpu.run("addTo", { n }, [sum, W], n);
    for (const W of weights) gpu.run("divide", { n }, [W, sum], n);
    gpu.submit();
    gpu.release(sum);

    // fuzja piramid
    const maxlevel = Math.floor(Math.log(Math.min(w, h)) / Math.log(2));
    const acc = {};
    const dims = [[w, h]];
    for (let l = 0; l < maxlevel; l++) dims.push([(dims[l][0] + 1) >> 1, (dims[l][1] + 1) >> 1]);
    for (const c of ["b", "g", "r"]) {
      acc[c] = dims.map(([lw, lh]) => {
        const b = gpu.buf(lw * lh);
        gpu.zero(b);
        return { buf: b, w: lw, h: lh };
      });
    }
    for (let k = 0; k < imgs.length; k++) {
      progress(`Łączenie (GPU): klatka ${k + 1}/${imgs.length}`);
      const wp = pyramid(gpu, weights[k], w, h, maxlevel);
      for (const c of ["b", "g", "r"]) {
        const gp = pyramid(gpu, imgs[k][c], w, h, maxlevel);
        for (let l = 0; l < maxlevel; l++) {
          gpu.run("lapAcc", { w: gp[l].w, h: gp[l].h, sw: gp[l + 1].w, sh: gp[l + 1].h },
            [acc[c][l].buf, gp[l].buf, gp[l + 1].buf, wp[l].buf], [gp[l].w, gp[l].h]);
        }
        const top = gp[maxlevel];
        gpu.run("fmaTop", { n: top.w * top.h }, [acc[c][maxlevel].buf, top.buf, wp[maxlevel].buf], top.w * top.h);
        gpu.submit();
        gpu.release(...gp.map((p) => p.buf));
      }
      gpu.release(...wp.map((p) => p.buf));
    }

    // składanie piramidy i odczyt
    const out = {};
    for (const c of ["b", "g", "r"]) {
      const pyr = acc[c];
      for (let l = maxlevel; l > 0; l--) {
        gpu.run("addUp", { w: pyr[l - 1].w, h: pyr[l - 1].h, sw: pyr[l].w, sh: pyr[l].h }, [pyr[l - 1].buf, pyr[l].buf], [pyr[l - 1].w, pyr[l - 1].h]);
      }
      gpu.run("clip", { n }, [pyr[0].buf], n);
      out[c] = await gpu.read(pyr[0].buf, alloc(n));
      gpu.release(...pyr.map((p) => p.buf));
    }
    return { result: out, width: w, height: h };
  } finally {
    gpu.submit();
    const errors = [await device.popErrorScope(), await device.popErrorScope()].filter(Boolean);
    gpu.trim();
    if (errors.length) throw new Error(`WebGPU: ${errors[0].message}`);
  }
}
