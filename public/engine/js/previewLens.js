// Korekcja obiektywu zmierzona z podglądu JPEG zapisanego w pliku RAW — dla aparatów bez
// profilu Sony. Port fit_preview_correction / apply_preview_correction z reference/bracket_merge.py.
//
// Aparaty (zwłaszcza kompakty, np. Canon G7 X) prostują dystorsję i winietę tylko w swoich
// JPEG-ach; RAW zostaje surowy, a producent nie zapisuje w nim współczynników. Pełnowymiarowy
// podgląd w pliku RAW to właśnie ten poprawiony JPEG, więc mierzymy z niego, gdzie w surowym
// obrazie leży każdy punkt kadru aparatu, i odtwarzamy to samo przekształcenie.
// Model: raw = c + (d / R) * (q0 + q1 r² + q2 r⁴ ...), gdzie d = punkt podglądu - środek
// podglądu, R = pół przekątnej podglądu, r = |d| / R.

import { percentiles, minMaxRows, histRows } from "./pipeline.js";
import { sampleCubic } from "./lens.js";

const PREVIEW_MIN_SCALE = 0.75;   // podgląd musi mieć co najmniej tyle szerokości RAW-a
const MATCH_LEVELS = [            // [zmniejszenie, pół-bok łaty, zasięg szukania, wyrazy modelu, min. NCC]
  [16, 8, 28, 2, 0.8],
  [4, 12, 6, 3, 0.85],
  [2, 12, 3, 3, 0.85],
];
const MIN_TEXTURE = 0.015;        // min. odchylenie std łaty podglądu (płaskie ściany nic nie mówią)
const MIN_MATCHES = 50;
const MAX_FIT_ERROR = 1.5;        // maks. mediana błędu dopasowania (piksele pełnej rozdzielczości)
const VIG_FACTOR = 8;             // winieta: porównanie na obrazach zmniejszonych tyle razy
const VIG_RINGS = 10, VIG_LEVELS = 16;
const VIG_MIN_V = 0.08, VIG_MAX_V = 0.92;  // zakres jasności podglądu (bez czerni i przepaleń)
const VIG_MAX_GAIN = 4.0;

const f32 = Math.fround;
// percentyl w bieżącym wątku (rawGray i tak działa już w workerze dekodującym)
const ROWS = { minMax: minMaxRows, hist: histRows };
const inline = { size: 1, rows: async (name, args, h) => [ROWS[name](args, 0, h)] };

// --- przygotowanie obrazów ---

/** Podgląd RGBA (uint8) -> szarość (b+g+r)/3/255, obrócona jak wynik LibRaw. */
export function previewGray({ data, width, height }, flip, rawWidth, rawHeight, alloc = (n) => new Float32Array(n)) {
  let g = alloc(width * height);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = f32(f32((data[j + 2] + data[j + 1] + data[j]) / 3) / 255);
  let w = width, h = height;
  // kody LibRaw: 3 = 180°, 5 = 90° w lewo, 6 = 90° w prawo. Przeglądarka mogła już obrócić
  // podgląd według EXIF, więc 5/6 stosujemy tylko, gdy orientacja nie zgadza się z RAW-em.
  const turned = (w > h) !== (rawWidth > rawHeight);
  if (flip === 3) {
    g.reverse();
  } else if ((flip === 5 || flip === 6) && turned) {
    const out = alloc(g.length);
    for (let i = 0; i < w; i++) {
      for (let j = 0; j < h; j++) {
        // np.rot90(k=1): out[i, j] = in[j, W-1-i];  np.rot90(k=-1): out[i, j] = in[H-1-j, i]
        out[i * h + j] = flip === 5 ? g[j * w + (w - 1 - i)] : g[(h - 1 - j) * w + i];
      }
    }
    g = out;
    [w, h] = [h, w];
  }
  return { gray: g, width: w, height: h };
}

/** Liniowa szarość (b+g+r)/3 jak img.mean(-1) w numpy (float32). */
function linearGray({ b, g, r }, alloc) {
  const out = alloc(b.length);
  for (let i = 0; i < out.length; i++) out[i] = f32(f32(f32(b[i] + g[i]) + r[i]) / 3);
  return out;
}

async function toSrgbGray(lin, width, height, alloc) {
  const scale = Math.max((await percentiles(inline, lin, [99], width, height))[0], 1e-6);
  const out = alloc(lin.length);
  for (let i = 0; i < out.length; i++) {
    let x = lin[i] / scale;
    x = x < 0 ? 0 : x > 1 ? 1 : x;
    out[i] = x <= 0.0031308 ? 12.92 * x : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
  }
  return out;
}

function boxDown(img, w, h, f) {
  const dw = Math.floor(w / f), dh = Math.floor(h / f);
  const out = new Float32Array(dw * dh);
  const rowSum = new Float64Array(dw);
  for (let y = 0; y < dh; y++) {
    rowSum.fill(0);
    for (let k = 0; k < f; k++) {
      const row = (y * f + k) * w;
      for (let x = 0; x < dw; x++) {
        let s = 0;
        for (let m = 0, p = row + x * f; m < f; m++, p++) s += img[p];
        rowSum[x] += s;
      }
    }
    for (let x = 0; x < dw; x++) out[y * dw + x] = rowSum[x] / (f * f);
  }
  return { data: out, w: dw, h: dh };
}

// --- model ---

function lensMap(sol, x, y, cp, R, out) {
  const dx = (x - cp[0]) / R, dy = (y - cp[1]) / R;
  const r2 = dx * dx + dy * dy;
  let s = 0, p = 1;
  for (let j = 2; j < sol.length; j++) { s += sol[j] * p; p *= r2; }
  out[0] = sol[0] + dx * s;
  out[1] = sol[1] + dy * s;
}

function median(values) {
  const a = Float64Array.from(values).sort();
  const n = a.length, m = n >> 1;
  return n % 2 ? a[m] : (a[m - 1] + a[m]) / 2;
}

/** Rozwiązuje M x = v (eliminacja Gaussa z wyborem elementu głównego). */
function solve(M, v) {
  const n = v.length;
  const a = M.map((row, i) => [...row, v[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
    [a[c], a[p]] = [a[p], a[c]];
    for (let r = c + 1; r < n; r++) {
      const k = a[r][c] / a[c][c];
      for (let j = c; j <= n; j++) a[r][j] -= k * a[c][j];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = a[r][n];
    for (let j = r + 1; j < n; j++) s -= a[r][j] * x[j];
    x[r] = s / a[r][r];
  }
  return x;
}

/** Ważone najmniejsze kwadraty przez równania normalne: wiersze rows[k] z wartościami b[k]. */
function leastSquares(rows, b, weights = null) {
  const n = rows[0].length;
  const M = Array.from({ length: n }, () => new Array(n).fill(0));
  const v = new Array(n).fill(0);
  rows.forEach((row, k) => {
    const wk = weights ? weights[k] : 1;
    for (let i = 0; i < n; i++) {
      v[i] += row[i] * wk * b[k];
      for (let j = 0; j < n; j++) M[i][j] += row[i] * wk * row[j];
    }
  });
  return solve(M, v);
}

const parabolaPeak = (a, b, c) => {
  const d = a - 2 * b + c;
  return d >= 0 ? 0 : 0.5 * (a - c) / d;
};

/** NCC jak cv2.matchTemplate(TM_CCOEFF_NORMED) łaty (2P+1)² w oknie (2P+2S+1)². */
function matchPatch(pd, px0, py0, rd, wx0, wy0, P, S, scores) {
  const side = 2 * P + 1, n = side * side, span = 2 * S + 1;
  let tMean = 0;
  for (let y = 0; y < side; y++) for (let x = 0; x < side; x++) tMean += pd.data[(py0 + y) * pd.w + px0 + x];
  tMean /= n;
  const t = new Float64Array(n);
  let tNorm = 0;
  for (let y = 0, k = 0; y < side; y++) {
    for (let x = 0; x < side; x++, k++) {
      t[k] = pd.data[(py0 + y) * pd.w + px0 + x] - tMean;
      tNorm += t[k] * t[k];
    }
  }
  let best = -Infinity, bx = 0, by = 0;
  for (let v = 0; v < span; v++) {
    for (let u = 0; u < span; u++) {
      let cross = 0, sw = 0, sw2 = 0;
      for (let y = 0, k = 0; y < side; y++) {
        const row = (wy0 + v + y) * rd.w + wx0 + u;
        for (let x = 0; x < side; x++, k++) {
          const val = rd.data[row + x];
          cross += t[k] * val;
          sw += val;
          sw2 += val * val;
        }
      }
      const varW = sw2 - (sw * sw) / n;
      const score = varW > 1e-12 && tNorm > 0 ? cross / Math.sqrt(tNorm * varW) : 0;
      scores[v * span + u] = score;
      if (score > best) { best = score; bx = u; by = v; }
    }
  }
  return { best, bx, by };
}

/** Dopasowania łat podglądu do RAW-a jednej klatki (jądro, klatki liczą się równolegle). */
export function matchPair({ pair: { raw, prev }, sol, cp, R, f, P, S, minNcc }) {
  const o = (f - 1) / 2, span = 2 * S + 1;
  const scores = new Float64Array(span * span);
  const pos = [0, 0];
  const pts = [];
  const rd = boxDown(raw.gray, raw.width, raw.height, f);
  const pd = boxDown(prev.gray, prev.width, prev.height, f);
  for (let y = P; y < pd.h - P; y += 2 * P) {
    for (let x = P; x < pd.w - P; x += 2 * P) {
      // odchylenie std łaty (jak patch.std() w numpy)
      let m = 0, m2 = 0;
      const side = 2 * P + 1;
      for (let yy = y - P; yy <= y + P; yy++) {
        for (let xx = x - P; xx <= x + P; xx++) { const val = pd.data[yy * pd.w + xx]; m += val; m2 += val * val; }
      }
      m /= side * side;
      if (Math.sqrt(Math.max(m2 / (side * side) - m * m, 0)) < MIN_TEXTURE) continue;

      const px = x * f + o, py = y * f + o;
      lensMap(sol, px, py, cp, R, pos);
      const ix = Math.floor((pos[0] - o) / f + 0.5), iy = Math.floor((pos[1] - o) / f + 0.5);
      if (ix - P - S < 0 || iy - P - S < 0 || ix + P + S >= rd.w || iy + P + S >= rd.h) continue;
      const { best, bx, by } = matchPatch(pd, x - P, y - P, rd, ix - P - S, iy - P - S, P, S, scores);
      if (best < minNcc || bx === 0 || bx === 2 * S || by === 0 || by === 2 * S) continue;
      const at = (u, v) => scores[v * span + u];
      const sx = parabolaPeak(at(bx - 1, by), at(bx, by), at(bx + 1, by));
      const sy = parabolaPeak(at(bx, by - 1), at(bx, by), at(bx, by + 1));
      pts.push([px, py, (ix - S + bx + sx) * f + o, (iy - S + by + sy) * f + o]);
    }
  }
  return pts;
}

function fitLensModel(pts, cp, R, terms, tol) {
  const rows = [], b = [];
  for (const [px, py, rx, ry] of pts) {
    const dx = (px - cp[0]) / R, dy = (py - cp[1]) / R, r2 = dx * dx + dy * dy;
    const rowX = [1, 0], rowY = [0, 1];
    for (let j = 0, p = 1; j < terms; j++, p *= r2) { rowX.push(dx * p); rowY.push(dy * p); }
    rows.push(rowX, rowY);
    b.push(rx, ry);
  }
  let keep = pts.map(() => true);
  let sol = null, err = null;
  for (let it = 0; it < 6; it++) {
    const kept = keep.reduce((n, k) => n + k, 0);
    if (kept < 12) return null;
    const kr = [], kb = [];
    keep.forEach((k, i) => { if (k) { kr.push(rows[2 * i], rows[2 * i + 1]); kb.push(b[2 * i], b[2 * i + 1]); } });
    sol = leastSquares(kr, kb);
    err = pts.map((_, i) => {
      const ex = rows[2 * i].reduce((s, a, j) => s + a * sol[j], 0) - b[2 * i];
      const ey = rows[2 * i + 1].reduce((s, a, j) => s + a * sol[j], 0) - b[2 * i + 1];
      return Math.hypot(ex, ey);
    });
    const limit = Math.max(3 * median(err.filter((_, i) => keep[i])), tol);
    keep = err.map((e) => e < limit);
  }
  return { sol, error: median(err.filter((_, i) => keep[i])) };
}

// --- winieta ---

function bilinear(img, x, y) {
  const { data, w, h } = img;
  x = Math.min(Math.max(x, 0), w - 1);
  y = Math.min(Math.max(y, 0), h - 1);
  const x0 = Math.min(Math.floor(x), w - 2), y0 = Math.min(Math.floor(y), h - 2);
  const fx = x - x0, fy = y - y0, i = y0 * w + x0;
  return (data[i] * (1 - fx) + data[i + 1] * fx) * (1 - fy) + (data[i + w] * (1 - fx) + data[i + w + 1] * fx) * fy;
}

/** Spadek światła na brzegach, który aparat skorygował w podglądzie. Dla pikseli o tej samej
 * jasności w podglądzie porównujemy RAW w kolejnych pierścieniach ze środkiem kadru — krzywa
 * tonalna aparatu się wtedy skraca. fitVignetting zwraca [b1, b2]: log(spadek) = b1 r² + b2 r⁴. */
export function vignetteSamples({ pair: { lin, prev }, sol, cp, R }) {
  const f = VIG_FACTOR, o = (f - 1) / 2;
  const samples = []; // [pierścień, średni promień, log(spadek), waga]
  const pos = [0, 0];
  const pd = boxDown(prev.gray, prev.width, prev.height, f);
  const rd = boxDown(lin, prev.rawWidth, prev.rawHeight, f);
  const n = pd.w * pd.h;
  const g = new Float64Array(n), r = new Float64Array(n), ring = new Int32Array(n), level = new Int32Array(n);
  const valid = new Uint8Array(n);
  for (let y = 0, i = 0; y < pd.h; y++) {
    for (let x = 0; x < pd.w; x++, i++) {
      const px = x * f + o, py = y * f + o;
      lensMap(sol, px, py, cp, R, pos);
      g[i] = bilinear(rd, (pos[0] - o) / f, (pos[1] - o) / f);
      r[i] = Math.hypot(px - cp[0], py - cp[1]) / R;
      ring[i] = Math.min(Math.trunc(r[i] * VIG_RINGS), VIG_RINGS - 1);
      const v = pd.data[i];
      valid[i] = v >= VIG_MIN_V && v < VIG_MAX_V && g[i] > 0;
      level[i] = Math.trunc((v - VIG_MIN_V) / (VIG_MAX_V - VIG_MIN_V) * VIG_LEVELS);
    }
  }
  for (let lv = 0; lv < VIG_LEVELS; lv++) {
    const cells = Array.from({ length: VIG_RINGS }, () => ({ g: [], r: [] }));
    for (let i = 0; i < n; i++) {
      if (valid[i] && level[i] === lv) { cells[ring[i]].g.push(g[i]); cells[ring[i]].r.push(r[i]); }
    }
    const center = [...cells[0].g, ...cells[1].g, ...cells[2].g];
    if (center.length < 50) continue;
    const ref = median(center);
    for (let rb = 3; rb < VIG_RINGS; rb++) {
      const c = cells[rb];
      if (c.g.length >= 30) {
        samples.push([rb, c.r.reduce((s, v) => s + v, 0) / c.r.length, Math.log(median(c.g) / ref), c.g.length]);
      }
    }
  }
  return samples;
}

function fitVignetting(samples) {
  if (new Set(samples.filter((s) => s[0] >= 5).map((s) => s[0])).size < 3) return [0, 0];
  return leastSquares(samples.map((s) => [s[1] ** 2, s[1] ** 4]), samples.map((s) => s[2]), samples.map((s) => s[3]));
}

// --- całość ---

/** Szarość RAW-a do porównań z podglądem: liniowa i w sRGB (jądro, liczone przy dekodowaniu). */
export async function rawGray(img, alloc = (n) => new Float32Array(n)) {
  const lin = linearGray(img, alloc);
  return { lin, srgb: await toSrgbGray(lin, img.width, img.height, alloc) };
}

/**
 * Mierzy korekcję aparatu z podglądów całego bracketu.
 * @param frames   liniowe klatki {b, g, r, width, height, gray: wynik rawGray()}
 * @param previews wyniki previewGray() albo null (brak podglądu)
 * @returns opis korekcji albo null
 */
export async function fitPreviewCorrection(exec, frames, previews, progress = () => {}) {
  const wr = frames[0].width, hr = frames[0].height;
  if (previews.some((p) => !p) || previews.some((p) => p.width !== previews[0].width || p.height !== previews[0].height)) return null;
  const wp = previews[0].width, hp = previews[0].height;
  if (wp < PREVIEW_MIN_SCALE * wr || hp < PREVIEW_MIN_SCALE * hr) return null;
  const cp = [(wp - 1) / 2, (hp - 1) / 2];
  const R = Math.hypot(cp[0], cp[1]);

  const pairs = frames.map((img, k) => ({
    lin: img.gray.lin,
    raw: { gray: img.gray.srgb, width: wr, height: hr },
    prev: { ...previews[k], rawWidth: wr, rawHeight: hr },
  }));

  let sol = [(wr - 1) / 2, (hr - 1) / 2, (wr / wp) * R];
  let error = null;
  for (const [k, [f, P, S, terms, minNcc]] of MATCH_LEVELS.entries()) {
    progress(`Pomiar korekcji obiektywu z podglądu (${k + 1}/${MATCH_LEVELS.length})`);
    const pts = (await exec.each("matchPair", pairs.map((pair) => ({ pair, sol, cp, R, f, P, S, minNcc })))).flat();
    if (pts.length < MIN_MATCHES) return null;
    const fit = fitLensModel(pts, cp, R, terms, 0.25 * f);
    if (!fit) return null;
    ({ sol, error } = fit);
  }
  if (error > MAX_FIT_ERROR) return null;
  const vig = fitVignetting((await exec.each("vignetteSamples", pairs.map((pair) => ({ pair, sol, cp, R })))).flat());
  let scale = sol[2] / R; // pikseli RAW na piksel podglądu w środku kadru
  if (Math.abs(scale - 1) < 0.03) scale = 1;
  return { sol, cp, R, scale, vig, width: Math.round(wp * scale), height: Math.round(hp * scale), error };
}

/** Mapa przekształcenia (pozycje w RAW-ie) i wzmocnienie winiety dla wierszy wyniku. */
export function previewMapRows({ mapX, mapY, gain, w, h, cp, R, scale, sol, vig: [b1, b2] }, y0, y1) {
  const pos = [0, 0];
  for (let y = y0; y < y1; y++) {
    const py = cp[1] + (y - (h - 1) / 2) / scale;
    for (let x = 0, i = y * w; x < w; x++, i++) {
      const px = cp[0] + (x - (w - 1) / 2) / scale;
      lensMap(sol, px, py, cp, R, pos);
      mapX[i] = pos[0];
      mapY[i] = pos[1];
      const r2 = ((px - cp[0]) ** 2 + (py - cp[1]) ** 2) / (R * R);
      gain[i] = Math.min(Math.max(Math.exp(-(b1 * r2 + b2 * r2 * r2)), 1), VIG_MAX_GAIN);
    }
  }
}

export function previewRemapRows({ src, dst, sw, sh, w, mapX, mapY, gain }, y0, y1) {
  for (const c of ["b", "g", "r"]) {
    const s = src[c], d = dst[c];
    for (let i = y0 * w; i < y1 * w; i++) {
      const v = f32(f32(sampleCubic(s, sw, sh, mapX[i], mapY[i])) * gain[i]);
      d[i] = v > 0 ? v : 0;
    }
  }
}

/** Mapa korekcji — wspólna dla wszystkich klatek bracketu. */
export async function previewCorrectionMap(exec, geo) {
  const { width: w, height: h } = geo;
  const n = w * h;
  const map = { mapX: exec.f32(n), mapY: exec.f32(n), gain: exec.f32(n), w };
  await exec.rows("previewMap", { ...map, h, cp: geo.cp, R: geo.R, scale: geo.scale, sol: geo.sol, vig: geo.vig }, h, w);
  return map;
}

/** Przekształca liniowy obraz do kadru aparatu i wyrównuje winietę. Zwraca nowy obraz. */
export async function applyPreviewCorrection(exec, img, geo, map) {
  const { width: w, height: h } = geo;
  const n = w * h;
  const dst = { b: exec.f32(n), g: exec.f32(n), r: exec.f32(n) };
  await exec.rows("previewRemap", { src: { b: img.b, g: img.g, r: img.r }, dst, sw: img.width, sh: img.height, ...map }, h, w);
  return { ...dst, width: w, height: h };
}
