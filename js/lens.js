// Korekcja obiektywu z profilu, który Sony zapisuje w każdym ARW (ten sam, którego aparat
// używa do JPEG-ów). Port read_lens_profile / fill_scale / correct_lens z bracket_merge.py.

const TAG_MAKE = 0x010f, TAG_SUBIFDS = 0x014a, TAG_EXIF = 0x8769;
const TAG_VIGNETTING = 0x7032, TAG_CA = 0x7035, TAG_DISTORTION = 0x7037;
const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8 };

/** Czyta tagi 0x7032/0x7035/0x7037 z IFD-ów pliku ARW (TIFF). Zwraca null, jeśli ich nie ma
 * albo plik nie jest RAW-em Sony (inne formaty, np. CR3/RAF, nie są TIFF-ami, a w TIFF-ach
 * innych producentów te numery tagów mogą znaczyć coś innego). */
export function readLensProfile(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  if (dv.byteLength < 8) return null;
  const order = dv.getUint16(0);
  if (order !== 0x4949 && order !== 0x4d4d) return null;
  const le = order === 0x4949;
  const u16 = (o) => dv.getUint16(o, le), u32 = (o) => dv.getUint32(o, le);
  if (u16(2) !== 42) return null;
  let make = "";
  const found = {};
  const seen = new Set();

  const readValues = (type, count, off) => {
    const out = [];
    for (let i = 0; i < count; i++) {
      const p = off + i * TYPE_SIZE[type];
      out.push(type === 3 ? u16(p) : type === 8 ? dv.getInt16(p, le) : type === 4 ? u32(p) : dv.getInt32(p, le));
    }
    return out;
  };

  const walk = (off, depth) => {
    if (!off || off + 2 > dv.byteLength || seen.has(off) || depth > 4) return;
    seen.add(off);
    const count = u16(off);
    for (let k = 0; k < count; k++) {
      const e = off + 2 + k * 12;
      if (e + 12 > dv.byteLength) return;
      const tag = u16(e), type = u16(e + 2), n = u32(e + 4);
      const size = (TYPE_SIZE[type] || 1) * n;
      const valOff = size <= 4 ? e + 8 : u32(e + 8);
      if (![2, 3, 4, 8, 9].includes(type) || valOff + size > dv.byteLength) continue;
      if (tag === TAG_MAKE && depth === 0 && type === 2) {
        make = new TextDecoder().decode(new Uint8Array(arrayBuffer, valOff, n)).replace(/\0.*$/, "");
      } else if (tag === TAG_VIGNETTING || tag === TAG_CA || tag === TAG_DISTORTION) {
        found[tag] ??= readValues(type, n, valOff);
      } else if (tag === TAG_SUBIFDS || tag === TAG_EXIF) {
        for (const sub of readValues(type, n, valOff)) walk(sub, depth + 1);
      }
    }
    const next = off + 2 + count * 12;
    if (next + 4 <= dv.byteLength) walk(u32(next), depth);
  };
  walk(u32(4), 0);
  if (!make.toUpperCase().startsWith("SONY")) return null;

  const curve = (tag, parts = 1) => {
    const v = found[tag];
    if (!v) return null;
    const n = Math.floor(v[0] / parts);
    return Array.from({ length: parts }, (_, i) => Float32Array.from(v.slice(1 + i * n, 1 + (i + 1) * n)));
  };
  const vig = curve(TAG_VIGNETTING), dist = curve(TAG_DISTORTION), ca = curve(TAG_CA, 2);
  if (!vig || !dist) return null;
  return { vignetting: vig[0], distortion: dist[0], ca: ca ?? [new Float32Array(dist[0].length), new Float32Array(dist[0].length)] };
}

// np.interp(r, linspace(0, 1, n), values)
function interpKnots(values, r) {
  const n = values.length;
  if (r <= 0) return values[0];
  if (r >= 1) return values[n - 1];
  const p = r * (n - 1), i = Math.floor(p);
  return values[i] + (p - i) * (values[i + 1] - values[i]);
}

/** Największe powiększenie, przy którym brzeg wyniku nadal mieści się w surowym kadrze. */
function fillScale(curve, cx, cy) {
  const R = Math.hypot(cx, cy);
  let best = Infinity;
  for (let k = 0; k <= 2000; k++) {
    const t = -1 + k / 1000;
    for (const [ex, ey] of [[t * cx, cy], [cx, t * cy]]) {
      const s = interpKnots(curve, Math.hypot(ex, ey) / R);
      best = Math.min(best, cx / Math.max(Math.abs(ex * s), 1e-6), cy / Math.max(Math.abs(ey * s), 1e-6));
    }
  }
  return best;
}

// współczynniki INTER_CUBIC OpenCV (A = -0.75); OpenCV 5 próbkuje w dokładnych pozycjach
// (bez kwantyzacji do 1/32 piksela jak w 4.x)
const A = -0.75;
function cubicWeights(t, out) {
  out[0] = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A;
  out[1] = ((A + 2) * t - (A + 3)) * t * t + 1;
  out[2] = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1;
  out[3] = 1 - out[0] - out[1] - out[2];
}

const reflect = (p, len) => {
  while (p < 0 || p >= len) p = p < 0 ? -p - 1 : 2 * len - p - 1;
  return p;
};

/** Stosuje profil do liniowego obrazu {b, g, r} (w miejscu zwraca nowe kanały). */
export function correctLens(img, profile, width, height) {
  const w = width, h = height, n = w * h;
  const cx = (w - 1) / 2, cy = (h - 1) / 2, R = Math.hypot(cx, cy);

  // winieta: współczynnik spadku światła -> dzielimy, rogi się rozjaśniają
  // (Python liczy współrzędne w float32 — Math.fround odtwarza to dla zgodności z OpenCV)
  const f32 = Math.fround;
  const radius = (x, y) => {
    const dx = f32(x - cx), dy = f32(y - cy);
    return f32(f32(Math.sqrt(f32(f32(dx * dx) + f32(dy * dy)))) / R);
  };
  const falloff = Float32Array.from(profile.vignetting, (v) => 2 ** (0.5 - 2 ** (v * 2 ** -13 - 1)));
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const f = f32(interpKnots(falloff, radius(x, y)));
      img.b[i] /= f; img.g[i] /= f; img.r[i] /= f;
    }
  }

  // dystorsja (+ aberracja chromatyczna dla R i B)
  let base = Float32Array.from(profile.distortion, (d) => d * 2 ** -14 + 1);
  const fs = fillScale(base, cx, cy);
  base = base.map((v) => v * fs);
  const [caR, caB] = profile.ca;
  const scales = {
    r: base.map((v, i) => v * (caR[i] * 2 ** -21 + 1)),
    g: base,
    b: base.map((v, i) => v * (caB[i] * 2 ** -21 + 1)),
  };

  const out = {};
  const wx = new Float64Array(4), wy = new Float64Array(4);
  for (const c of ["b", "g", "r"]) {
    const src = img[c], dst = new Float32Array(n), sc = scales[c];
    for (let y = 0; y < h; y++) {
      const dy = f32(y - cy);
      for (let x = 0; x < w; x++) {
        const dx = f32(x - cx);
        const s = f32(interpKnots(sc, radius(x, y)));
        const mx = f32(cx + f32(dx * s)), my = f32(cy + f32(dy * s));
        const ix = Math.floor(mx), iy = Math.floor(my);
        cubicWeights(mx - ix, wx);
        cubicWeights(my - iy, wy);
        const x0 = reflect(ix - 1, w), x1 = reflect(ix, w), x2 = reflect(ix + 1, w), x3 = reflect(ix + 2, w);
        let acc = 0;
        for (let m = 0; m < 4; m++) {
          const row = reflect(iy - 1 + m, h) * w;
          acc += wy[m] * (wx[0] * src[row + x0] + wx[1] * src[row + x1] + wx[2] * src[row + x2] + wx[3] * src[row + x3]);
        }
        dst[y * w + x] = acc > 0 ? acc : 0;
      }
    }
    out[c] = dst;
    img[c] = null;
  }
  return { ...img, ...out };
}
