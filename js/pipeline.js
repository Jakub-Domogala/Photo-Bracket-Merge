// Łączenie bracketów — port 1:1 pipeline'u z reference/bracket_merge.py.
// Obrazy trzymamy planarnie jako {b, g, r} Float32Array (kolejność BGR jak w OpenCV,
// bo od niej zależą wagi luminancji i szarości w Mertensie).
//
// Ciężkie pętle są jądrami wierszowymi (args, y0, y1) — liczą wiersze [y0, y1), więc wykonawca
// (js/exec.js) może je rozdzielić między workery. Funkcje kroków są asynchroniczne i dostają
// wykonawcę `exec`. Moduł nie zależy od DOM-u: działa w Web Workerze i w Node (testy).

export const KEY = 0.12;          // docelowa jasność (log-average, liniowo) środkowej klatki
export const BLACK_PCT = 0.5;     // percentyl luminancji ustawiany jako czerń
export const S_CURVE = 0.3;       // siła krzywej S (0 = brak)
export const WHITE_PCT = 99.5;    // percentyl luminancji, który po auto WB nie może przekroczyć 1
export const BRIGHT_PCT = 90;     // ten percentyl luminancji ...
export const BRIGHT_TARGET = 0.9; // ... rozciągamy do tej wartości (białe ściany zamiast szarych)
export const BRIGHT_MAX = 1.4;    // maks. wzmocnienie (żeby nie przepalić scen nocnych)
export const KNEE = 0.75;         // powyżej tego światła są miękko kompresowane zamiast obcinane

const clip01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const lumAt = (b, g, r) => 0.0722 * b + 0.7152 * g + 0.2126 * r;

export const toSrgb1 = (x) => {
  x = clip01(x);
  return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
};
const toLinear1 = (x) => {
  x = clip01(x);
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
};

export function lumRows({ b, g, r, y, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) y[i] = lumAt(b[i], g[i], r[i]);
}

// --- percentyle jak np.percentile (interpolacja liniowa), liczone z drobnego histogramu ---

const HIST_BINS = 1 << 20;

export function minMaxRows({ arr, mask, w }, y0, y1) {
  let lo = Infinity, hi = -Infinity, n = 0;
  for (let i = y0 * w; i < y1 * w; i++) {
    if (mask && !mask[i]) continue;
    const v = arr[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    n++;
  }
  return [lo, hi, n];
}

export function histRows({ arr, mask, w, lo, scale }, y0, y1) {
  const hist = new Uint32Array(HIST_BINS);
  for (let i = y0 * w; i < y1 * w; i++) {
    if (mask && !mask[i]) continue;
    hist[Math.floor((arr[i] - lo) * scale)]++;
  }
  return hist;
}

/** Percentyle qs (tablica, 0..100) obrazu arr (h wierszy po w), opcjonalnie tylko z maską. */
export async function percentiles(exec, arr, qs, w, h, mask = null) {
  let lo = Infinity, hi = -Infinity, n = 0;
  for (const [l, u, c] of await exec.rows("minMax", { arr, mask, w }, h, w)) {
    lo = Math.min(lo, l); hi = Math.max(hi, u); n += c;
  }
  if (n === 0) return qs.map(() => NaN);
  if (hi === lo) return qs.map(() => lo);
  const scale = (HIST_BINS - 1) / (hi - lo);
  // jeden histogram na workera (każdy ma 4 MB, więc nie dzielimy drobniej)
  const parts = await exec.rows("hist", { arr, mask, w, lo, scale }, h, w, { chunks: exec.size });
  const hist = parts[0];
  for (let k = 1; k < parts.length; k++) {
    const p = parts[k];
    for (let bin = 0; bin < HIST_BINS; bin++) hist[bin] += p[bin];
  }
  // wartość k-tego elementu posortowanej tablicy (k ułamkowe) przy założeniu
  // równomiernego rozkładu wewnątrz kubełka
  const valueAt = (k) => {
    let acc = 0;
    for (let bin = 0; bin < HIST_BINS; bin++) {
      const c = hist[bin];
      if (acc + c > k) return lo + (bin + (k - acc + 0.5) / c) / scale;
      acc += c;
    }
    return hi;
  };
  return qs.map((q) => {
    const rank = (q / 100) * (n - 1);
    const k0 = Math.floor(rank);
    const v0 = valueAt(k0);
    return k0 + 1 < n ? v0 + (rank - k0) * (valueAt(k0 + 1) - v0) : v0;
  });
}

// --- dekodowanie: LibRaw zwraca dane z krzywą gamma BT.709 (libraw-wasm ignoruje 'gamm'),
// odwracamy dokładnie tę samą krzywą, którą liczy LibRaw (gamma_curve, mode 2) ---

let inverseGamma = null;

function librawGammaCurve(pwr = 0.45, ts = 4.5, imax = 0x10000) {
  const g = [pwr, ts, 0, 0, 0, 0];
  const bnd = [0, 0];
  bnd[g[1] >= 1 ? 1 : 0] = 1;
  if (g[1] && (g[1] - 1) * (g[0] - 1) <= 0) {
    for (let i = 0; i < 48; i++) {
      g[2] = (bnd[0] + bnd[1]) / 2;
      bnd[(Math.pow(g[2] / g[1], -g[0]) - 1) / g[0] - 1 / g[2] > -1 ? 1 : 0] = g[2];
    }
    g[3] = g[2] / g[1];
    g[4] = g[2] * (1 / g[0] - 1);
  }
  const curve = new Uint16Array(0x10000);
  for (let i = 0; i < 0x10000; i++) {
    const r = i / imax;
    curve[i] = r < 1
      ? Math.trunc(0x10000 * (r < g[3] ? r * g[1] : Math.pow(r, g[0]) * (1 + g[4]) - g[4]))
      : 0xffff;
  }
  return curve;
}

function getInverseGamma() {
  if (inverseGamma) return inverseGamma;
  const curve = librawGammaCurve();
  const sum = new Float64Array(0x10000), cnt = new Uint32Array(0x10000);
  for (let i = 0; i < 0x10000; i++) { sum[curve[i]] += i; cnt[curve[i]]++; }
  inverseGamma = new Float32Array(0x10000);
  let last = 0;
  for (let c = 0; c < 0x10000; c++) {
    if (cnt[c]) last = sum[c] / cnt[c];
    inverseGamma[c] = last / 65535;
  }
  return inverseGamma;
}

/** RGB 16-bit (interleaved, gamma LibRaw) -> liniowe planarne BGR 0..1 w podanych tablicach. */
export function linearize(rgb16, out) {
  const inv = getInverseGamma();
  const { b, g, r } = out;
  for (let i = 0, j = 0; i < b.length; i++, j += 3) {
    r[i] = inv[rgb16[j]];
    g[i] = inv[rgb16[j + 1]];
    b[i] = inv[rgb16[j + 2]];
  }
  return out;
}

// --- auto-ekspozycja ---

export function logSumRows({ b, g, r, w }, y0, y1) {
  let s = 0;
  for (let i = y0 * w; i < y1 * w; i++) {
    const y = lumAt(b[i], g[i], r[i]);
    s += Math.log(y < 1e-4 ? 1e-4 : y);
  }
  return s;
}

export function srgbRows({ b, g, r, w, gain }, y0, y1) {
  for (const ch of [b, g, r]) {
    for (let i = y0 * w; i < y1 * w; i++) ch[i] = toSrgb1(ch[i] * gain);
  }
}

/** Jedno wzmocnienie dla wszystkich klatek (zachowuje odstępy EV), liczone z klatki
 * środkowej pod względem jasności. Zamienia obrazy w miejscu na sRGB. */
export async function autoExposure(exec, linear, key = KEY) {
  const levels = await Promise.all(linear.map(async (img) => {
    const parts = await exec.rows("logSum", { b: img.b, g: img.g, r: img.r, w: img.width }, img.height, img.width);
    return Math.exp(parts.reduce((s, v) => s + v, 0) / (img.width * img.height));
  }));
  levels.sort((a, b) => a - b);
  let gain = key / levels[Math.floor(levels.length / 2)];
  gain = Math.min(Math.max(gain, 0.25), 32);
  await Promise.all(linear.map((img) =>
    exec.rows("srgb", { b: img.b, g: img.g, r: img.r, w: img.width, gain }, img.height, img.width)));
  return linear;
}

// --- Mertens: wierna kopia cv::MergeMertens (OpenCV 4.x/5.x) ---

function reflect101(p, len) {
  if (len === 1) return 0;
  while (p < 0 || p >= len) p = p < 0 ? -p : 2 * len - p - 2;
  return p;
}

// pyrDown: poziomo kolumny 2x-2..2x+2 z jądrem [1 4 6 4 1] (wiersze wejścia), potem pionowo (wiersze wyjścia)
export function pyrDownHRows({ src, w, tmp, dw }, y0, y1) {
  for (let y = y0; y < y1; y++) {
    const row = y * w, out = y * dw;
    for (let x = 0; x < dw; x++) {
      const c = 2 * x;
      let s;
      if (c >= 2 && c + 2 < w) {
        s = src[row + c - 2] + 4 * src[row + c - 1] + 6 * src[row + c] + 4 * src[row + c + 1] + src[row + c + 2];
      } else {
        s = src[row + reflect101(c - 2, w)] + 4 * src[row + reflect101(c - 1, w)] + 6 * src[row + reflect101(c, w)]
          + 4 * src[row + reflect101(c + 1, w)] + src[row + reflect101(c + 2, w)];
      }
      tmp[out + x] = s;
    }
  }
}

export function pyrDownVRows({ tmp, dst, dw, h }, y0, y1) {
  for (let y = y0; y < y1; y++) {
    const c = 2 * y;
    const r0 = reflect101(c - 2, h) * dw, r1 = reflect101(c - 1, h) * dw, r2 = reflect101(c, h) * dw,
      r3 = reflect101(c + 1, h) * dw, r4 = reflect101(c + 2, h) * dw;
    const out = y * dw;
    for (let x = 0; x < dw; x++) {
      dst[out + x] = (tmp[r0 + x] + 4 * tmp[r1 + x] + 6 * tmp[r2 + x] + 4 * tmp[r3 + x] + tmp[r4 + x]) / 256;
    }
  }
}

// pyrUp: poziomo do szerokości 2*sw (wiersze źródła), brzegi jak w pyrUp_ OpenCV
export function pyrUpHRows({ src, sw, tmp }, y0, y1) {
  const fw = 2 * sw;
  for (let y = y0; y < y1; y++) {
    const s = y * sw, o = y * fw;
    if (sw === 1) { tmp[o] = tmp[o + 1] = src[s] * 8; continue; }
    tmp[o] = src[s] * 6 + src[s + 1] * 2;
    tmp[o + 1] = (src[s] + src[s + 1]) * 4;
    for (let x = 1; x < sw - 1; x++) {
      tmp[o + 2 * x] = src[s + x - 1] + src[s + x] * 6 + src[s + x + 1];
      tmp[o + 2 * x + 1] = (src[s + x] + src[s + x + 1]) * 4;
    }
    const l = sw - 1;
    tmp[o + 2 * l] = src[s + l - 1] + src[s + l] * 7;
    tmp[o + 2 * l + 1] = src[s + l] * 8;
  }
}

/** Pionowo, wiersze źródła y -> wiersze wyniku 2y i 2y+1, od razu dodane do d ze znakiem sign
 * (d -= pyrUp(...) dla laplasjanu, d += pyrUp(...) przy składaniu). Wiersz źródłowy dla sy
 * to reflect101(2*sy, 2*sh)/2. */
export function pyrUpVRows({ tmp, sw, sh, d, dw, dh, sign }, y0, y1) {
  const fw = 2 * sw;
  const rowOf = (sy) => (reflect101(2 * sy, 2 * sh) >> 1) * fw;
  const f32 = Math.fround;
  for (let y = y0; y < y1; y++) {
    const a = rowOf(y - 1), b = rowOf(y), c = rowOf(y + 1);
    const d0 = 2 * y * dw, d1 = (2 * y + 1) * dw;
    const odd = 2 * y + 1 < dh;
    for (let x = 0; x < dw; x++) {
      if (odd) d[d1 + x] += sign * f32((tmp[b + x] + tmp[c + x]) * 4 / 64);
      d[d0 + x] += sign * f32((tmp[a + x] + tmp[b + x] * 6 + tmp[c + x]) / 64);
    }
  }
}

async function pyrDown(exec, src, w, h) {
  const dw = (w + 1) >> 1, dh = (h + 1) >> 1;
  const tmp = exec.f32(dw * h), dst = exec.f32(dw * dh);
  await exec.rows("pyrDownH", { src, w, tmp, dw }, h, dw);
  await exec.rows("pyrDownV", { tmp, dst, dw, h }, dh, dw);
  exec.release(tmp);
  return { data: dst, w: dw, h: dh };
}

/** d (dw x dh) += sign * pyrUp(src (sw x sh)). */
async function addPyrUp(exec, src, sw, sh, d, dw, dh, sign) {
  const tmp = exec.f32(2 * sw * sh);
  await exec.rows("pyrUpH", { src, sw, tmp }, sh, 2 * sw);
  await exec.rows("pyrUpV", { tmp, sw, sh, d, dw, dh, sign }, sh, 2 * dw);
  exec.release(tmp);
}

async function buildPyramid(exec, data, w, h, maxlevel) {
  const pyr = [{ data, w, h }];
  for (let l = 0; l < maxlevel; l++) {
    const p = pyr[l];
    pyr.push(await pyrDown(exec, p.data, p.w, p.h));
  }
  return pyr;
}

export function grayRows({ b, g, r, gray, w }, y0, y1) {
  // cvtColor(RGB2GRAY) na danych BGR: kanał 0 (u nas B) dostaje wagę "R"
  for (let i = y0 * w; i < y1 * w; i++) gray[i] = 0.299 * b[i] + 0.587 * g[i] + 0.114 * r[i];
}

export function weightRows({ b, g, r, gray, W, w, h, wc, ws, we }, y0, y1) {
  for (let y = y0; y < y1; y++) {
    const up = reflect101(y - 1, h) * w, dn = reflect101(y + 1, h) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const xl = x > 0 ? x - 1 : reflect101(x - 1, w), xr = x < w - 1 ? x + 1 : reflect101(x + 1, w);
      const lap = gray[row + xl] + gray[row + xr] + gray[up + x] + gray[dn + x] - 4 * gray[i];
      const contrast = Math.abs(lap);
      const mean = (b[i] + g[i] + r[i]) / 3;
      const sat = Math.sqrt((b[i] - mean) ** 2 + (g[i] - mean) ** 2 + (r[i] - mean) ** 2);
      // iloczyn trzech exp(-(c-0.5)²/0.08) jako jedno exp sumy
      const wexp = Math.exp(-((b[i] - 0.5) ** 2 + (g[i] - 0.5) ** 2 + (r[i] - 0.5) ** 2) / 0.08);
      const cw = wc === 1 ? contrast : Math.pow(contrast, wc);
      const sw = ws === 1 ? sat : Math.pow(sat, ws);
      const ew = we === 1 ? wexp : Math.pow(wexp, we);
      W[i] = cw * sw * ew + 1e-12;
    }
  }
}

export function normalizeRows({ Ws, w }, y0, y1) {
  const f32 = Math.fround;
  for (let i = y0 * w; i < y1 * w; i++) {
    let s = 0;
    for (const W of Ws) s = f32(s + W[i]);
    for (const W of Ws) W[i] /= s;
  }
}

export function fmaRows({ acc, d, wd, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) acc[i] += d[i] * wd[i];
}

export function clipRows({ d, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) d[i] = clip01(d[i]);
}

/** Exposure fusion. `images` (sRGB 0..1) są zwalniane w trakcie, żeby oszczędzać pamięć. */
export async function mergeMertens(exec, images, w, h, { contrast = 1, saturation = 1, exposure = 1 } = {}, progress = () => {}) {
  const n = w * h;
  const gray = exec.f32(n);
  const weights = [];
  for (const { b, g, r } of images) {
    const W = exec.f32(n);
    await exec.rows("gray", { b, g, r, gray, w }, h, w);
    await exec.rows("weight", { b, g, r, gray, W, w, h, wc: contrast, ws: saturation, we: exposure }, h, w);
    weights.push(W);
  }
  exec.release(gray);
  await exec.rows("normalize", { Ws: weights, w }, h, w);

  const maxlevel = Math.floor(Math.log(Math.min(w, h)) / Math.log(2));
  const res = { b: null, g: null, r: null };

  for (let k = 0; k < images.length; k++) {
    progress(`Łączenie: klatka ${k + 1}/${images.length}`);
    const wp = await buildPyramid(exec, weights[k], w, h, maxlevel);
    weights[k] = null;
    for (const c of ["b", "g", "r"]) {
      const ip = await buildPyramid(exec, images[k][c], w, h, maxlevel);
      images[k][c] = null;
      for (let l = 0; l < maxlevel; l++) {
        await addPyrUp(exec, ip[l + 1].data, ip[l + 1].w, ip[l + 1].h, ip[l].data, ip[l].w, ip[l].h, -1);
      }
      if (!res[c]) res[c] = ip.map((p) => ({ data: exec.f32(p.data.length).fill(0), w: p.w, h: p.h }));
      for (let l = 0; l <= maxlevel; l++) {
        await exec.rows("fma", { acc: res[c][l].data, d: ip[l].data, wd: wp[l].data, w: ip[l].w }, ip[l].h, ip[l].w);
      }
      exec.release(...ip.map((p) => p.data));
    }
    exec.release(...wp.map((p) => p.data));
    images[k] = null;
  }

  const out = {};
  for (const c of ["b", "g", "r"]) {
    const pyr = res[c];
    for (let l = maxlevel; l > 0; l--) {
      await addPyrUp(exec, pyr[l].data, pyr[l].w, pyr[l].h, pyr[l - 1].data, pyr[l - 1].w, pyr[l - 1].h, 1);
    }
    await exec.rows("clip", { d: pyr[0].data, w }, h, w);
    out[c] = pyr[0].data;
    exec.release(...pyr.slice(1).map((p) => p.data));
  }
  return out;
}

// --- auto balans bieli (opcjonalny) ---

export function awbPrepRows({ src, lin, y, hi, sat, w }, y0, y1) {
  for (const c of ["b", "g", "r"]) {
    const s = src[c], l = lin[c];
    for (let i = y0 * w; i < y1 * w; i++) l[i] = toLinear1(s[i]);
  }
  const { b, g, r } = lin;
  for (let i = y0 * w; i < y1 * w; i++) {
    y[i] = lumAt(b[i], g[i], r[i]);
    const mx = Math.max(b[i], g[i], r[i]), mn = Math.min(b[i], g[i], r[i]);
    hi[i] = mx;
    sat[i] = (mx - mn) / (mx + 1e-6);
  }
}

export function awbMaskRows({ y, hi, mask, p80, p99, w }, y0, y1) {
  let count = 0;
  for (let i = y0 * w; i < y1 * w; i++) {
    mask[i] = y[i] > p80 && y[i] < p99 && hi[i] < 0.97 ? 1 : 0;
    count += mask[i];
  }
  return count;
}

export function awbAvgRows({ lin, mask, sat, satMed, w }, y0, y1) {
  const s = [0, 0, 0, 0];
  for (let i = y0 * w; i < y1 * w; i++) {
    if (mask[i] && sat[i] < satMed) { s[0] += lin.b[i]; s[1] += lin.g[i]; s[2] += lin.r[i]; s[3]++; }
  }
  return s;
}

export function awbScaleRows({ lin, k, y, w }, y0, y1) {
  for (const c of ["b", "g", "r"]) for (let i = y0 * w; i < y1 * w; i++) lin[c][i] *= k[c];
  lumRows({ ...lin, y, w }, y0, y1);
}

export function awbOutRows({ lin, out, norm, w }, y0, y1) {
  for (const c of ["b", "g", "r"]) for (let i = y0 * w; i < y1 * w; i++) out[c][i] = toSrgb1(lin[c][i] / norm);
}

/** Zwraca obraz sRGB po korekcji (zapisuje w tablicach `srgb`). */
export async function autoWhiteBalance(exec, srgb, w, h) {
  const n = w * h;
  const lin = { b: exec.f32(n), g: exec.f32(n), r: exec.f32(n) };
  const y = exec.f32(n), hi = exec.f32(n), sat = exec.f32(n);
  await exec.rows("awbPrep", { src: srgb, lin, y, hi, sat, w }, h, w);
  const [p80, p99] = await percentiles(exec, y, [80, 99], w, h);
  const mask = exec.u8(n);
  const counts = await exec.rows("awbMask", { y, hi, mask, p80, p99, w }, h, w);
  if (counts.reduce((s, v) => s + v, 0) < 1000) {
    exec.release(lin.b, lin.g, lin.r, y, hi, sat, mask);
    return srgb;
  }
  const [satMed] = await percentiles(exec, sat, [50], w, h, mask);
  const avg = [0, 0, 0, 0];
  for (const s of await exec.rows("awbAvg", { lin, mask, sat, satMed, w }, h, w)) for (let c = 0; c < 4; c++) avg[c] += s[c];
  for (let c = 0; c < 3; c++) avg[c] /= avg[3];
  const mean = (avg[0] + avg[1] + avg[2]) / 3;
  const k = { b: mean / avg[0], g: mean / avg[1], r: mean / avg[2] };
  await exec.rows("awbScale", { lin, k, y, w }, h, w);
  const norm = Math.max(1, (await percentiles(exec, y, [WHITE_PCT], w, h))[0]);
  await exec.rows("awbOut", { lin, out: srgb, norm, w }, h, w);
  exec.release(lin.b, lin.g, lin.r, y, hi, sat, mask);
  return srgb;
}

// --- wykończenie: rozjaśnienie bieli, punkt czerni, krzywa S — na luminancji ---

const shoulder = (x, knee = KNEE) => (x <= knee ? x : knee + (1 - knee) * (1 - Math.exp(-(x - knee) / (1 - knee))));

export function shoulderRows({ y, t, k, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) t[i] = shoulder(y[i] * k);
}

export function finishRows({ b, g, r, y, t, lo, rgba, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) {
    let v = clip01((t[i] - lo) / (1 - lo));
    v = (1 - S_CURVE) * v + S_CURVE * (v * v * (3 - 2 * v));
    const ratio = v / Math.max(y[i], 1e-4);
    let ob = b[i] * ratio, og = g[i] * ratio, or = r[i] * ratio;
    // kanały powyżej 1: zamiast obcinać (zmiana odcienia), mieszamy w stronę bieli o tej samej jasności
    const peak = Math.max(ob, og, or);
    const wgt = clip01((peak - 1) / Math.max(peak - v, 1e-4));
    ob = ob * (1 - wgt) + v * wgt;
    og = og * (1 - wgt) + v * wgt;
    or = or * (1 - wgt) + v * wgt;
    const j = i * 4;
    rgba[j] = Math.round(clip01(or) * 255);
    rgba[j + 1] = Math.round(clip01(og) * 255);
    rgba[j + 2] = Math.round(clip01(ob) * 255);
    rgba[j + 3] = 255;
  }
}

/** Zwraca RGBA 8-bit (gotowe do canvas / JPEG, zwykły bufor). */
export async function finish(exec, srgb, w, h) {
  const { b, g, r } = srgb;
  const n = w * h;
  const y = exec.f32(n);
  await exec.rows("lum", { b, g, r, y, w }, h, w);
  const hi = Math.max((await percentiles(exec, y, [BRIGHT_PCT], w, h))[0], 1e-3);
  const k = Math.min(Math.max(BRIGHT_TARGET / hi, 1), BRIGHT_MAX);
  const t = exec.f32(n);
  await exec.rows("shoulder", { y, t, k, w }, h, w);
  const [lo] = await percentiles(exec, t, [BLACK_PCT], w, h);
  const rgba = exec.u8(n * 4);
  await exec.rows("finish", { b, g, r, y, t, lo, rgba, w }, h, w);
  exec.release(y, t, b, g, r);
  if (rgba.buffer instanceof ArrayBuffer) return new Uint8ClampedArray(rgba.buffer);
  const copy = new Uint8ClampedArray(rgba); // ImageData nie przyjmuje pamięci współdzielonej
  exec.release(rgba);
  return copy;
}
