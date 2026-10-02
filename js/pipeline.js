// Łączenie bracketów — port 1:1 pipeline'u z reference/bracket_merge.py.
// Obrazy trzymamy planarnie jako {b, g, r} Float32Array (kolejność BGR jak w OpenCV,
// bo od niej zależą wagi luminancji i szarości w Mertensie).
//
// Moduł nie zależy od DOM-u: działa w Web Workerze i w Node (testy porównawcze).

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

function lum({ b, g, r }) {
  const y = new Float32Array(b.length);
  for (let i = 0; i < y.length; i++) y[i] = lumAt(b[i], g[i], r[i]);
  return y;
}

const toSrgb1 = (x) => {
  x = clip01(x);
  return x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
};
const toLinear1 = (x) => {
  x = clip01(x);
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
};

// --- percentyle jak np.percentile (interpolacja liniowa), liczone z drobnego histogramu ---

const HIST_BINS = 1 << 20;
let histBuf = null;

export function percentile(arr, q, mask = null) {
  let lo = Infinity, hi = -Infinity, n = 0;
  for (let i = 0; i < arr.length; i++) {
    if (mask && !mask[i]) continue;
    const v = arr[i];
    if (v < lo) lo = v;
    if (v > hi) hi = v;
    n++;
  }
  if (n === 0) return NaN;
  if (hi === lo) return lo;
  histBuf ??= new Uint32Array(HIST_BINS);
  histBuf.fill(0);
  const scale = (HIST_BINS - 1) / (hi - lo);
  for (let i = 0; i < arr.length; i++) {
    if (mask && !mask[i]) continue;
    histBuf[Math.floor((arr[i] - lo) * scale)]++;
  }
  // wartość k-tego elementu posortowanej tablicy (k ułamkowe) przy założeniu
  // równomiernego rozkładu wewnątrz kubełka
  const valueAt = (k) => {
    let acc = 0;
    for (let bin = 0; bin < HIST_BINS; bin++) {
      const c = histBuf[bin];
      if (acc + c > k) return lo + (bin + (k - acc + 0.5) / c) / scale;
      acc += c;
    }
    return hi;
  };
  const rank = (q / 100) * (n - 1);
  const k0 = Math.floor(rank);
  const v0 = valueAt(k0);
  return k0 + 1 < n ? v0 + (rank - k0) * (valueAt(k0 + 1) - v0) : v0;
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

/** RGB 16-bit (interleaved, gamma LibRaw) -> liniowe planarne BGR 0..1. */
export function linearize(rgb16, width, height) {
  const inv = getInverseGamma();
  const n = width * height;
  const b = new Float32Array(n), g = new Float32Array(n), r = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 3) {
    r[i] = inv[rgb16[j]];
    g[i] = inv[rgb16[j + 1]];
    b[i] = inv[rgb16[j + 2]];
  }
  return { b, g, r, width, height };
}

// --- auto-ekspozycja ---

function logAverage(img) {
  let s = 0;
  const { b, g, r } = img;
  for (let i = 0; i < b.length; i++) {
    const y = lumAt(b[i], g[i], r[i]);
    s += Math.log(y < 1e-4 ? 1e-4 : y);
  }
  return Math.exp(s / b.length);
}

/** Jedno wzmocnienie dla wszystkich klatek (zachowuje odstępy EV), liczone z klatki
 * środkowej pod względem jasności. Zamienia obrazy w miejscu na sRGB. */
export function autoExposure(linear, key = KEY) {
  const levels = linear.map(logAverage).sort((a, b) => a - b);
  let gain = key / levels[Math.floor(levels.length / 2)];
  gain = Math.min(Math.max(gain, 0.25), 32);
  for (const img of linear) {
    for (const ch of [img.b, img.g, img.r]) {
      for (let i = 0; i < ch.length; i++) ch[i] = toSrgb1(ch[i] * gain);
    }
  }
  return linear;
}

// --- Mertens: wierna kopia cv::MergeMertens (OpenCV 4.x/5.x) ---

function reflect101(p, len) {
  if (len === 1) return 0;
  while (p < 0 || p >= len) p = p < 0 ? -p : 2 * len - p - 2;
  return p;
}

function pyrDown(src, w, h) {
  const dw = (w + 1) >> 1, dh = (h + 1) >> 1;
  const tmp = new Float32Array(dw * h);
  // poziomo: kolumny 2x-2..2x+2, kernel [1 4 6 4 1]
  for (let y = 0; y < h; y++) {
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
  const dst = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const c = 2 * y;
    const r0 = reflect101(c - 2, h) * dw, r1 = reflect101(c - 1, h) * dw, r2 = reflect101(c, h) * dw,
      r3 = reflect101(c + 1, h) * dw, r4 = reflect101(c + 2, h) * dw;
    const out = y * dw;
    for (let x = 0; x < dw; x++) {
      dst[out + x] = (tmp[r0 + x] + 4 * tmp[r1 + x] + 6 * tmp[r2 + x] + 4 * tmp[r3 + x] + tmp[r4 + x]) / 256;
    }
  }
  return { data: dst, w: dw, h: dh };
}

function pyrUp(src, sw, sh, dw, dh) {
  // poziomo do szerokości 2*sw (potem przycinamy do dw), brzegi jak w pyrUp_ OpenCV
  const fw = 2 * sw;
  const tmp = new Float32Array(fw * sh);
  for (let y = 0; y < sh; y++) {
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
  // pionowo: wiersz źródłowy dla sy to reflect101(2*sy, 2*sh)/2
  const rowOf = (sy) => (reflect101(2 * sy, 2 * sh) >> 1) * fw;
  const dst = new Float32Array(dw * dh);
  for (let y = 0; y < sh; y++) {
    const a = rowOf(y - 1), b = rowOf(y), c = rowOf(y + 1);
    const d0 = 2 * y * dw, y1 = Math.min(2 * y + 1, dh - 1), d1 = y1 * dw;
    for (let x = 0; x < dw; x++) {
      const t1 = (tmp[b + x] + tmp[c + x]) * 4 / 64;
      const t0 = (tmp[a + x] + tmp[b + x] * 6 + tmp[c + x]) / 64;
      dst[d1 + x] = t1;
      dst[d0 + x] = t0;
    }
  }
  return dst;
}

function buildPyramid(data, w, h, maxlevel) {
  const pyr = [{ data, w, h }];
  for (let l = 0; l < maxlevel; l++) {
    const p = pyr[l];
    pyr.push(pyrDown(p.data, p.w, p.h));
  }
  return pyr;
}

function mertensWeight(img, w, h, wc, ws, we) {
  const { b, g, r } = img;
  const n = w * h;
  // cvtColor(RGB2GRAY) na danych BGR: kanał 0 (u nas B) dostaje wagę "R"
  const gray = new Float32Array(n);
  for (let i = 0; i < n; i++) gray[i] = 0.299 * b[i] + 0.587 * g[i] + 0.114 * r[i];
  const W = new Float32Array(n);
  for (let y = 0; y < h; y++) {
    const up = reflect101(y - 1, h) * w, dn = reflect101(y + 1, h) * w, row = y * w;
    for (let x = 0; x < w; x++) {
      const i = row + x;
      const xl = x > 0 ? x - 1 : reflect101(x - 1, w), xr = x < w - 1 ? x + 1 : reflect101(x + 1, w);
      const lap = gray[row + xl] + gray[row + xr] + gray[up + x] + gray[dn + x] - 4 * gray[i];
      const contrast = Math.abs(lap);
      const mean = (b[i] + g[i] + r[i]) / 3;
      const sat = Math.sqrt((b[i] - mean) ** 2 + (g[i] - mean) ** 2 + (r[i] - mean) ** 2);
      const wexp = Math.exp(-((b[i] - 0.5) ** 2) / 0.08) * Math.exp(-((g[i] - 0.5) ** 2) / 0.08)
        * Math.exp(-((r[i] - 0.5) ** 2) / 0.08);
      const cw = wc === 1 ? contrast : Math.pow(contrast, wc);
      const sw = ws === 1 ? sat : Math.pow(sat, ws);
      const ew = we === 1 ? wexp : Math.pow(wexp, we);
      W[i] = cw * sw * ew + 1e-12;
    }
  }
  return W;
}

/** Exposure fusion. `images` (sRGB 0..1) są zwalniane w trakcie, żeby oszczędzać pamięć. */
export function mergeMertens(images, w, h, { contrast = 1, saturation = 1, exposure = 1 } = {}, progress = () => {}) {
  const n = w * h;
  const weights = images.map((img) => mertensWeight(img, w, h, contrast, saturation, exposure));
  const sum = new Float32Array(n);
  for (const W of weights) for (let i = 0; i < n; i++) sum[i] += W[i];
  for (const W of weights) for (let i = 0; i < n; i++) W[i] /= sum[i];

  const maxlevel = Math.floor(Math.log(Math.min(w, h)) / Math.log(2));
  const res = { b: null, g: null, r: null };

  for (let k = 0; k < images.length; k++) {
    progress(`Łączenie: klatka ${k + 1}/${images.length}`);
    const wp = buildPyramid(weights[k], w, h, maxlevel);
    weights[k] = null;
    for (const c of ["b", "g", "r"]) {
      const ip = buildPyramid(images[k][c], w, h, maxlevel);
      images[k][c] = null;
      for (let l = 0; l < maxlevel; l++) {
        const up = pyrUp(ip[l + 1].data, ip[l + 1].w, ip[l + 1].h, ip[l].w, ip[l].h);
        const d = ip[l].data;
        for (let i = 0; i < d.length; i++) d[i] -= up[i];
      }
      if (!res[c]) res[c] = ip.map((p) => ({ data: new Float32Array(p.data.length), w: p.w, h: p.h }));
      for (let l = 0; l <= maxlevel; l++) {
        const d = ip[l].data, wd = wp[l].data, acc = res[c][l].data;
        for (let i = 0; i < d.length; i++) acc[i] += d[i] * wd[i];
      }
    }
    images[k] = null;
  }

  const out = {};
  for (const c of ["b", "g", "r"]) {
    const pyr = res[c];
    for (let l = maxlevel; l > 0; l--) {
      const up = pyrUp(pyr[l].data, pyr[l].w, pyr[l].h, pyr[l - 1].w, pyr[l - 1].h);
      const d = pyr[l - 1].data;
      for (let i = 0; i < d.length; i++) d[i] += up[i];
    }
    const d = pyr[0].data;
    for (let i = 0; i < d.length; i++) d[i] = clip01(d[i]);
    out[c] = d;
  }
  return out;
}

// --- auto balans bieli (opcjonalny) ---

export function autoWhiteBalance(srgb) {
  const n = srgb.b.length;
  const lin = { b: new Float32Array(n), g: new Float32Array(n), r: new Float32Array(n) };
  for (const c of ["b", "g", "r"]) for (let i = 0; i < n; i++) lin[c][i] = toLinear1(srgb[c][i]);
  const y = lum(lin);
  const hi = new Float32Array(n), sat = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const mx = Math.max(lin.b[i], lin.g[i], lin.r[i]), mn = Math.min(lin.b[i], lin.g[i], lin.r[i]);
    hi[i] = mx;
    sat[i] = (mx - mn) / (mx + 1e-6);
  }
  const p80 = percentile(y, 80), p99 = percentile(y, 99);
  const mask = new Uint8Array(n);
  let count = 0;
  for (let i = 0; i < n; i++) {
    if (y[i] > p80 && y[i] < p99 && hi[i] < 0.97) { mask[i] = 1; count++; }
  }
  if (count < 1000) return srgb;
  const satMed = percentile(sat, 50, mask);
  const avg = [0, 0, 0];
  let m = 0;
  for (let i = 0; i < n; i++) {
    if (mask[i] && sat[i] < satMed) { avg[0] += lin.b[i]; avg[1] += lin.g[i]; avg[2] += lin.r[i]; m++; }
  }
  for (let c = 0; c < 3; c++) avg[c] /= m;
  const mean = (avg[0] + avg[1] + avg[2]) / 3;
  const k = { b: mean / avg[0], g: mean / avg[1], r: mean / avg[2] };
  for (const c of ["b", "g", "r"]) for (let i = 0; i < n; i++) lin[c][i] *= k[c];
  const norm = Math.max(1, percentile(lum(lin), WHITE_PCT));
  const out = {};
  for (const c of ["b", "g", "r"]) {
    out[c] = new Float32Array(n);
    for (let i = 0; i < n; i++) out[c][i] = toSrgb1(lin[c][i] / norm);
  }
  return out;
}

// --- wykończenie: rozjaśnienie bieli, punkt czerni, krzywa S — na luminancji ---

const shoulder = (x, knee = KNEE) => (x <= knee ? x : knee + (1 - knee) * (1 - Math.exp(-(x - knee) / (1 - knee))));

/** Zwraca RGBA 8-bit (gotowe do canvas / JPEG). */
export function finish(srgb, width, height) {
  const { b, g, r } = srgb;
  const n = width * height;
  const y = lum(srgb);
  const hi = Math.max(percentile(y, BRIGHT_PCT), 1e-3);
  const k = Math.min(Math.max(BRIGHT_TARGET / hi, 1), BRIGHT_MAX);
  const t = new Float32Array(n);
  for (let i = 0; i < n; i++) t[i] = shoulder(y[i] * k);
  const lo = percentile(t, BLACK_PCT);
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0; i < n; i++) {
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
  return rgba;
}
