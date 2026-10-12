// Wyrównanie klatek przesuniętych między zdjęciami (tylko przesunięcie, bez obrotu) — port
// align_frames z reference/bracket_merge.py. Klatki porównujemy mapami progu medianowego
// (Ward 2003), które nie zależą od ekspozycji, i przycinamy do wspólnej części.

import { percentiles } from "./pipeline.js";

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
 * Przesunięcia klatek względem klatki o środkowej jasności.
 * @param grays  log-szarości klatek (wyniki logGrayRows, w x h)
 * @returns null (brak przesunięcia), { tooBig: true } albo { shifts: [[sx, sy]], x0, y0, w, h }
 *          — wycinek we współrzędnych klatki odniesienia; klatka k zaczyna się w (x0 + sx, y0 + sy)
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
  if (shifts.every(([sx, sy]) => !sx && !sy)) return null;
  const x0 = Math.max(0, ...shifts.map(([sx]) => -sx)), x1 = Math.min(w, ...shifts.map(([sx]) => w - sx));
  const y0 = Math.max(0, ...shifts.map(([, sy]) => -sy)), y1 = Math.min(h, ...shifts.map(([, sy]) => h - sy));
  if (x1 - x0 < (1 - MAX_CROP) * w || y1 - y0 < (1 - MAX_CROP) * h) return { tooBig: true };
  return { shifts, x0, y0, w: x1 - x0, h: y1 - y0 };
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
    for (const c of ["b", "g", "r"]) {
      next[c] = exec.f32(plan.w * plan.h);
      await exec.rows("crop", { src: img[c], dst: next[c], w, dw: plan.w, x0: plan.x0 + sx, y0: plan.y0 + sy }, plan.h, plan.w);
    }
    exec.release(img.b, img.g, img.r);
    out.push(next);
  }
  return { frames: out, plan };
}
