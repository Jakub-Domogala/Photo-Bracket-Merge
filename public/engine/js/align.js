// Wyrównanie klatek przesuniętych lub przekrzywionych między zdjęciami — port align_frames z
// reference/bracket_merge.py. Najpierw przesunięcie o całe piksele: klatki porównujemy mapami
// progu medianowego (Ward 2003), które nie zależą od ekspozycji. Potem poprawka afiniczna
// (obrót, pochylenie, skala, ułamki piksela) metodą Gaussa-Newtona na gradiencie log-jasności.
// Wynik przycinamy do wspólnej części.

import { percentiles } from "./pipeline.js";
import { sampleCubic } from "./lens.js";

export const MAX_CROP = 0.03; // przycięcie kadru najwyżej o tyle (z każdej osi), inaczej bez wyrównania
const ALIGN_TOL = 0.05;       // piksele tak bliskie medianie (log jasności) nie liczą się przy porównaniu

const f32 = Math.fround;
const MIN_GRAY = f32(1e-6);

export function logGrayRows({ b, g, r, out, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) {
    const m = f32(f32(f32(b[i] + g[i]) + r[i]) / 3);
    out[i] = Math.log(m > MIN_GRAY ? m : MIN_GRAY);
  }
}

export function down2Rows({ src, w, dst, dw }, y0, y1) {
  for (let y = y0; y < y1; y++) {
    const a = 2 * y * w, c = a + w;
    for (let x = 0; x < dw; x++) {
      const k = 2 * x;
      dst[y * dw + x] = f32(f32(f32(src[a + k] + src[c + k]) + src[a + k + 1]) + src[c + k + 1]) / 4;
    }
  }
}

/** 2 = jaśniej niż mediana, 1 = ciemniej, 0 = za blisko mediany. */
export function codeRows({ src, med, code, w }, y0, y1) {
  for (let i = y0 * w; i < y1 * w; i++) {
    const d = src[i] - med;
    code[i] = Math.abs(d) <= ALIGN_TOL ? 0 : d > 0 ? 2 : 1;
  }
}

/** Liczba pikseli (w wierszach a), gdzie a[y, x] i b[y + sy, x + sx] leżą po różnych stronach mediany. */
export function shiftErrorRows({ a, b, w, h, sx, sy }, y0, y1) {
  let n = 0;
  const ya = Math.max(y0, -sy), yb = Math.min(y1, h - sy);
  const xa = Math.max(0, -sx), xb = Math.min(w, w - sx);
  for (let y = ya; y < yb; y++) {
    const ra = y * w, rb = (y + sy) * w + sx;
    for (let x = xa; x < xb; x++) {
      const p = a[ra + x], q = b[rb + x];
      if (p && q && p !== q) n++;
    }
  }
  return n;
}

export function cropRows({ src, dst, w, dw, x0, y0 }, ya, yb) {
  for (let y = ya; y < yb; y++) {
    const s = (y + y0) * w + x0;
    dst.set(src.subarray(s, s + dw), y * dw);
  }
}

// --- poprawka afiniczna (te same stałe i kolejność działań co w reference/bracket_merge.py) ---

const AFFINE_WORK_W = 2000;       // szarości zmniejszamy tak długo, aż szerokość <= tyle (potem 3 poziomy piramidy)
const AFFINE_LEVELS = 3;
const AFFINE_ITERS = [20, 12, 8]; // iteracje od najmniejszego poziomu
const AFFINE_LO = Math.log(0.004); // log-jasność poniżej / powyżej: piksel prawie czarny / przepalony
const AFFINE_HI = Math.log(0.9);
const AFFINE_MAX_ROT = 3.0;       // stopnie
const AFFINE_MAX_SCALE = 0.05;    // |skala - 1|
const AFFINE_MAX_MOVE = 0.06;     // poprawka nie może przesunąć rogu o więcej niż tyle dłuższego boku
const AFFINE_MIN_CC = 0.5;        // słabsze dopasowanie traktujemy jako nieudane (zostaje samo przesunięcie)
const AFFINE_SKIP_PX = 0.5;       // poprawka mniejsza niż tyle pikseli (pełna rozdzielczość) nie jest warta resamplingu
const AFFINE_MAX_STEP = 0.02;     // największa zmiana parametru w jednej iteracji
export const AFFINE_MARGIN = 2;   // klatki warpowane tracą na brzegu tyle pikseli (odbicie brzegu psuje obraz)

function smooth121(a, w, h) {
  const t = new Float64Array(w * h), out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) t[row + x] = (a[row + Math.max(x - 1, 0)] + 2 * a[row + x] + a[row + Math.min(x + 1, w - 1)]) / 4;
  }
  for (let y = 0; y < h; y++) {
    const up = Math.max(y - 1, 0) * w, mid = y * w, dn = Math.min(y + 1, h - 1) * w;
    for (let x = 0; x < w; x++) out[mid + x] = (t[up + x] + 2 * t[mid + x] + t[dn + x]) / 4;
  }
  return out;
}

function centralGradient(a, w, h) {
  const gx = new Float64Array(w * h), gy = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x > 0 && x < w - 1) gx[i] = (a[i + 1] - a[i - 1]) / 2;
      if (y > 0 && y < h - 1) gy[i] = (a[i + w] - a[i - w]) / 2;
    }
  }
  return [gx, gy];
}

/** Piksele o log-jasności w zakresie roboczym, z marginesem 3 px wokół reszty. */
function validMask(g, w, h) {
  const ok = new Uint8Array(w * h);
  for (let i = 0; i < ok.length; i++) ok[i] = g[i] > AFFINE_LO && g[i] < AFFINE_HI ? 1 : 0;
  const hs = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let d = -3; d <= 3; d++) { const xx = x + d; if (xx >= 0 && xx < w) n += ok[y * w + xx]; }
      hs[y * w + x] = n;
    }
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let n = 0;
      for (let d = -3; d <= 3; d++) { const yy = y + d; if (yy >= 0 && yy < h) n += hs[yy * w + x]; }
      out[y * w + x] = n === 49 ? 1 : 0;
    }
  }
  return out;
}

function down2Gray(src, w, h) {
  const dw = w >> 1, dh = h >> 1, dst = new Float32Array(dw * dh);
  down2Rows({ src, w, dst, dw }, 0, dh);
  return dst;
}

/** Piramida cech (gradient log-jasności) i masek jednej klatki; gray = log-szarość pełnej rozdzielczości. */
function affineLevels(gray, w, h) {
  let g = gray, gw = w, gh = h, lam = 1;
  while (gw > AFFINE_WORK_W) { g = down2Gray(g, gw, gh); gw >>= 1; gh >>= 1; lam *= 2; }
  const levels = [];
  for (let l = 0; l < AFFINE_LEVELS; l++) {
    const g64 = Float64Array.from(g);
    const [gx, gy] = centralGradient(smooth121(g64, gw, gh), gw, gh);
    const mag = new Float64Array(gw * gh);
    for (let i = 0; i < mag.length; i++) mag[i] = Math.sqrt(gx[i] * gx[i] + gy[i] * gy[i]);
    const f = smooth121(mag, gw, gh);
    const [fx, fy] = centralGradient(f, gw, gh);
    levels.push({ f, gx: fx, gy: fy, m: validMask(g64, gw, gh), lam, w: gw, h: gh });
    if (l < AFFINE_LEVELS - 1) { g = down2Gray(g, gw, gh); gw >>= 1; gh >>= 1; lam *= 2; }
  }
  return levels;
}

// bufory robocze affineTerms (ten sam rozmiar wraca w każdej iteracji)
let scratchBuf = null;
function scratch(cap) {
  if (scratchBuf?.cap !== cap) {
    const f = () => new Float64Array(cap);
    scratchBuf = { cap, av: f(), bv: f(), gxv: f(), gyv: f(), xcv: f(), ycv: f() };
  }
  return scratchBuf;
}

/**
 * Składniki jednej iteracji na jednym poziomie. Model (współrzędne ciągłe, środek piksela w +0.5):
 * u = M (x - C) + C + s t, M = I + [[p0, p1], [p2, p3]], t = (p4, p5), C = środek obrazu,
 * s = połowa dłuższego boku — parametry są bezwymiarowe, więc ten sam wektor p opisuje wszystkie poziomy.
 * @returns { cc, n, H (6x6), g (6) } albo null
 */
function affineTerms(A, B, p, fullW, fullH) {
  const { lam, f: fa, m: ma, w, h } = A;
  const cx = fullW / 2 / lam, cy = fullH / 2 / lam, s = Math.max(fullW, fullH) / 2 / lam;
  const { av, bv, gxv, gyv, xcv, ycv } = scratch(w * h);
  let n = 0;
  for (let y = 0; y < h; y++) {
    const yc = y + 0.5 - cy;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!ma[i]) continue;
      const xc = x + 0.5 - cx;
      const ux = (1 + p[0]) * xc + p[1] * yc + cx + s * p[4] - 0.5;
      const uy = p[2] * xc + (1 + p[3]) * yc + cy + s * p[5] - 0.5;
      if (!(ux >= 1 && ux <= w - 3 && uy >= 1 && uy <= h - 3)) continue;
      if (!B.m[Math.floor(uy + 0.5) * w + Math.floor(ux + 0.5)]) continue;
      const x0 = Math.floor(ux), y0 = Math.floor(uy), fx = ux - x0, fy = uy - y0, j = y0 * w + x0;
      const bil = (arr) => (arr[j] * (1 - fx) + arr[j + 1] * fx) * (1 - fy) + (arr[j + w] * (1 - fx) + arr[j + w + 1] * fx) * fy;
      av[n] = fa[i]; bv[n] = bil(B.f); gxv[n] = bil(B.gx); gyv[n] = bil(B.gy); xcv[n] = xc; ycv[n] = yc;
      n++;
    }
  }
  if (n < 500) return null;
  let ma_ = 0, mb_ = 0;
  for (let i = 0; i < n; i++) { ma_ += av[i]; mb_ += bv[i]; }
  ma_ /= n; mb_ /= n;
  let va = 0, vb = 0;
  for (let i = 0; i < n; i++) { va += (av[i] - ma_) ** 2; vb += (bv[i] - mb_) ** 2; }
  const sa = Math.sqrt(va / n), sb = Math.sqrt(vb / n);
  if (sa < 1e-9 || sb < 1e-9) return null;
  const H = Array.from({ length: 6 }, () => new Float64Array(6)), g = new Float64Array(6), J = new Float64Array(6);
  let cc = 0;
  for (let i = 0; i < n; i++) {
    const a = (av[i] - ma_) / sa, b = (bv[i] - mb_) / sb, gx = gxv[i] / sb, gy = gyv[i] / sb, xc = xcv[i], yc = ycv[i];
    cc += a * b;
    J[0] = gx * xc; J[1] = gx * yc; J[2] = gy * xc; J[3] = gy * yc; J[4] = gx * s; J[5] = gy * s;
    const r = a - b;
    for (let u = 0; u < 6; u++) {
      g[u] += J[u] * r;
      for (let v = u; v < 6; v++) H[u][v] += J[u] * J[v];
    }
  }
  for (let u = 0; u < 6; u++) for (let v = 0; v < u; v++) H[u][v] = H[v][u];
  return { cc: cc / n, n, H, g };
}

/** Eliminacja Gaussa z częściowym wyborem elementu głównego (jak solve_gauss w referencji). */
function solveGauss(H, g) {
  const n = g.length;
  const M = H.map((row, i) => { const r = Float64Array.from([...row, g[i]]); r[i] += row[i] * 1e-9; return r; });
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-300) return new Float64Array(n);
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let j = 0; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  const x = new Float64Array(n);
  for (let r = n - 1; r >= 0; r--) {
    let acc = M[r][n];
    for (let j = r + 1; j < n; j++) acc -= M[r][j] * x[j];
    x[r] = acc / M[r][r];
  }
  return x;
}

/**
 * Dopasowuje klatkę do odniesienia, zaczynając od przesunięcia o całe piksele.
 * @returns { A: [a, b, c, d, e, f] | null, status } — A mapuje współrzędne indeksów pikseli odniesienia
 *   na współrzędne klatki; status: "ok" | "skip" (poprawka za mała) | "weak" (dopasowanie niepewne) |
 *   "implausible" (obrót / skala / przesunięcie poza zakresem)
 */
export function fitAffine(refLv, lv, shift, w, h) {
  const sFull = Math.max(w, h) / 2;
  const p0 = [0, 0, 0, 0, shift[0] / sFull, shift[1] / sFull];
  const p = p0.slice();
  for (let l = AFFINE_LEVELS - 1; l >= 0; l--) {
    for (let it = 0; it < AFFINE_ITERS[AFFINE_LEVELS - 1 - l]; it++) {
      const t = affineTerms(refLv[l], lv[l], p, w, h);
      if (!t) return { A: null, status: "weak" };
      const step = solveGauss(t.H, t.g).map((v) => Math.min(Math.max(v, -AFFINE_MAX_STEP), AFFINE_MAX_STEP));
      let big = 0;
      for (let k = 0; k < 6; k++) { p[k] += step[k]; big = Math.max(big, Math.abs(step[k])); }
      if (big * Math.max(w, h) / 2 / refLv[l].lam < 0.005) break;
    }
  }
  const t0 = affineTerms(refLv[0], lv[0], p0, w, h), t1 = affineTerms(refLv[0], lv[0], p, w, h);
  if (!t0 || !t1 || t1.cc < AFFINE_MIN_CC || t1.cc < t0.cc) return { A: null, status: "weak" };
  const m00 = 1 + p[0], m01 = p[1], m10 = p[2], m11 = 1 + p[3];
  const Tx = p[4] * sFull, Ty = p[5] * sFull, Cx = w / 2, Cy = h / 2;
  const A = [m00, m01, Cx + Tx - 0.5 - (m00 * (Cx - 0.5) + m01 * (Cy - 0.5)),
    m10, m11, Cy + Ty - 0.5 - (m10 * (Cx - 0.5) + m11 * (Cy - 0.5))];
  let moved = 0;
  for (const [x, y] of [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]]) {
    moved = Math.max(moved, Math.abs(A[0] * x + A[1] * y + A[2] - (x + shift[0])), Math.abs(A[3] * x + A[4] * y + A[5] - (y + shift[1])));
  }
  const angle = Math.atan2(m10 - m01, m00 + m11) * 180 / Math.PI;
  const scale = Math.sqrt(Math.abs(m00 * m11 - m01 * m10));
  if (Math.abs(angle) > AFFINE_MAX_ROT || Math.abs(scale - 1) > AFFINE_MAX_SCALE || moved > AFFINE_MAX_MOVE * Math.max(w, h)) {
    return { A: null, status: "implausible" };
  }
  if (moved < AFFINE_SKIP_PX) return { A: null, status: "skip" };
  return { A, status: "ok" };
}

/** Jądro dla exec.each: dopasowanie jednej klatki (cechy odniesienia liczone w każdym wywołaniu). */
export function alignAffine({ ref, gray, shift, w, h }) {
  return fitAffine(affineLevels(ref, w, h), affineLevels(gray, w, h), shift, w, h);
}

/** Odwrotność przekształcenia [a, b, c, d, e, f] (jak cv2.invertAffineTransform). */
function invertAffine([a, b, c, d, e, f]) {
  const det = a * e - b * d, ia = e / det, ib = -b / det, id = -d / det, ie = a / det;
  return [ia, ib, -(ia * c + ib * f), id, ie, -(id * c + ie * f)];
}

/**
 * Największy prostokąt we współrzędnych odniesienia, który po przekształceniu każdej klatki leży
 * w jej kadrze (rogi liczone konserwatywnie). Dla samych przesunięć o całe piksele daje dokładnie
 * ten sam wycinek co przesunięcia same.
 * @param transforms [{ A: [a, b, c, d, e, f], m: margines }]
 * @returns [x0, y0, w, h]
 */
export function commonCrop(transforms, w, h) {
  let x0 = -Infinity, x1 = Infinity, y0 = -Infinity, y1 = Infinity;
  for (const { A, m } of transforms) {
    const inv = invertAffine(A);
    const [tl, tr, bl, br] = [[m, m], [w - 1 - m, m], [m, h - 1 - m], [w - 1 - m, h - 1 - m]]
      .map(([x, y]) => [inv[0] * x + inv[1] * y + inv[2], inv[3] * x + inv[4] * y + inv[5]]);
    x0 = Math.max(x0, tl[0], bl[0]); x1 = Math.min(x1, tr[0], br[0]);
    y0 = Math.max(y0, tl[1], tr[1]); y1 = Math.min(y1, bl[1], br[1]);
  }
  const r3 = (v) => Math.round(v * 1000) / 1000;
  const cx0 = Math.ceil(r3(x0)) + 0, cy0 = Math.ceil(r3(y0)) + 0;
  return [cx0, cy0, Math.floor(r3(x1)) - cx0 + 1, Math.floor(r3(y1)) - cy0 + 1];
}

/** Klatka {b, g, r} po przekształceniu A (jak warp_frame w referencji): wiersze [ya, yb) wyniku cw x ch. */
export function affineWarpRows({ src, dst, w, h, cw, x0, y0, A }, ya, yb) {
  const chans = ["b", "g", "r"].map((c) => [src[c], dst[c]]);
  for (let y = ya; y < yb; y++) {
    const Y = f32(y0 + y);
    for (let x = 0; x < cw; x++) {
      const X = f32(x0 + x);
      const mx = f32(A[0] * X + A[1] * Y + A[2]), my = f32(A[3] * X + A[4] * Y + A[5]);
      for (const [s, d] of chans) {
        const v = sampleCubic(s, w, h, mx, my);
        d[y * cw + x] = v > 0 ? v : 0;
      }
    }
  }
}

async function codePyramid(exec, gray, w, h, levels) {
  const codes = [];
  const [med] = await percentiles(exec, gray, [50], w, h); // mediana pełnej rozdzielczości dla wszystkich poziomów
  let level = { data: gray, w, h };
  for (let l = 0; l <= levels; l++) {
    const code = exec.u8(level.w * level.h);
    await exec.rows("code", { src: level.data, med, code, w: level.w }, level.h, level.w);
    codes.push({ data: code, w: level.w, h: level.h });
    if (l === levels) break;
    const dw = level.w >> 1, dh = level.h >> 1;
    const dst = exec.f32(dw * dh);
    await exec.rows("down2", { src: level.data, w: level.w, dst, dw }, dh, dw);
    if (level.data !== gray) exec.release(level.data);
    level = { data: dst, w: dw, h: dh };
  }
  if (level.data !== gray) exec.release(level.data);
  return codes;
}

async function findShift(exec, refPyr, pyr) {
  let sx = 0, sy = 0;
  for (let l = refPyr.length - 1; l >= 0; l--) {
    const { data: a, w, h } = refPyr[l], b = pyr[l].data;
    const cand = [];
    for (const dy of [-1, 0, 1]) for (const dx of [-1, 0, 1]) cand.push([2 * sx + dx, 2 * sy + dy]);
    const errors = await Promise.all(cand.map(async ([cx, cy]) =>
      (await exec.rows("shiftError", { a, b, w, h, sx: cx, sy: cy }, h, w)).reduce((s, v) => s + v, 0)));
    let best = 0;
    for (let k = 1; k < cand.length; k++) if (errors[k] < errors[best]) best = k;
    [sx, sy] = cand[best];
  }
  return [sx, sy];
}

/**
 * Przekształcenia klatek względem klatki o środkowej jasności.
 * @param grays  log-szarości klatek (wyniki logGrayRows, w x h)
 * @returns null (nic do zrobienia), { implausible } (nic do zrobienia, ale część klatek odrzucona jako
 *          zbyt obrócona / przeskalowana), { tooBig: true, implausible }
 *          albo { shifts: [[sx, sy]], affines: [A | null], x0, y0, w, h, implausible } — wycinek we
 *          współrzędnych klatki odniesienia; klatka k bez affines[k] zaczyna się w (x0 + sx, y0 + sy),
 *          z affines[k] = [a, b, c, d, e, f] jest przekształcana (współrzędne odniesienia -> klatki)
 */
export async function alignPlan(exec, grays, w, h) {
  let levels = Math.ceil(Math.log2(MAX_CROP * Math.max(w, h) + 1)) - 1;
  while (levels > 0 && Math.min(w, h) >> levels < 32) levels--;
  const means = grays.map((g) => { let s = 0; for (let i = 0; i < g.length; i++) s += g[i]; return s / g.length; });
  const ref = means.map((m, k) => [m, k]).sort((p, q) => p[0] - q[0])[grays.length >> 1][1];
  const pyrs = [];
  for (const g of grays) pyrs.push(await codePyramid(exec, g, w, h, levels));
  const shifts = [];
  for (let k = 0; k < grays.length; k++) shifts.push(k === ref ? [0, 0] : await findShift(exec, pyrs[ref], pyrs[k]));
  for (const pyr of pyrs) exec.release(...pyr.map((p) => p.data));

  const others = grays.map((_, k) => k).filter((k) => k !== ref);
  const fits = await exec.each("alignAffine", others.map((k) => ({ ref: grays[ref], gray: grays[k], shift: shifts[k], w, h })));
  const affines = grays.map(() => null), implausible = [];
  fits.forEach((fit, j) => {
    if (fit.status === "ok") affines[others[j]] = fit.A;
    else if (fit.status === "implausible") implausible.push(others[j]);
  });
  if (shifts.every(([sx, sy]) => !sx && !sy) && affines.every((a) => !a)) return implausible.length ? { implausible } : null;
  const [x0, y0, cw, ch] = commonCrop(shifts.map(([sx, sy], k) =>
    affines[k] ? { A: affines[k], m: AFFINE_MARGIN } : { A: [1, 0, sx, 0, 1, sy], m: 0 }), w, h);
  if (cw < (1 - MAX_CROP) * w || ch < (1 - MAX_CROP) * h) return { tooBig: true, implausible };
  return { shifts, affines, x0, y0, w: cw, h: ch, implausible };
}

/** Wyrównuje i przycina liniowe klatki {b, g, r} (w x h) na CPU. Zwraca { frames, plan }. */
export async function alignFrames(exec, frames, w, h) {
  const grays = [];
  for (const { b, g, r } of frames) {
    const out = exec.f32(w * h);
    await exec.rows("logGray", { b, g, r, out, w }, h, w);
    grays.push(out);
  }
  const plan = await alignPlan(exec, grays, w, h);
  exec.release(...grays);
  if (!plan?.shifts) return { frames, plan };
  const out = [];
  for (const [k, img] of frames.entries()) {
    const [sx, sy] = plan.shifts[k];
    const next = { width: plan.w, height: plan.h };
    for (const c of ["b", "g", "r"]) next[c] = exec.f32(plan.w * plan.h);
    if (plan.affines[k]) {
      await exec.rows("affineWarp", { src: { b: img.b, g: img.g, r: img.r }, dst: { b: next.b, g: next.g, r: next.r },
        w, h, cw: plan.w, x0: plan.x0, y0: plan.y0, A: plan.affines[k] }, plan.h, plan.w);
    } else {
      for (const c of ["b", "g", "r"]) {
        await exec.rows("crop", { src: img[c], dst: next[c], w, dw: plan.w, x0: plan.x0 + sx, y0: plan.y0 + sy }, plan.h, plan.w);
      }
    }
    exec.release(img.b, img.g, img.r);
    out.push(next);
  }
  return { frames: out, plan };
}
