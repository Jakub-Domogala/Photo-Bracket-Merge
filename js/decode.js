// Dekodowanie RAW (LibRaw WASM) — jądro uruchamiane osobno dla każdej klatki, więc klatki
// bracketu dekodują się równolegle w osobnych workerach.

import { linearize } from "./pipeline.js";
import { readLensProfile } from "./lens.js";
import { previewGray, rawGray } from "./previewLens.js";

// Ustawienia jak rawpy.postprocess w wersji Python (gamma i tak jest odwracana w linearize)
const DECODE_SETTINGS = { useCameraWb: true, noAutoBright: true, outputBps: 16, outputColor: 1 };
const DEFAULT_JPEG_DECODER = new URL("./jpeg.js", import.meta.url).href;

let librawPromise = null;

function loadLibRaw() {
  librawPromise ??= (async () => {
    const { default: factory } = await import("../vendor/libraw-wasm/libraw.js");
    const wasmUrl = new URL("../vendor/libraw-wasm/libraw.wasm", import.meta.url);
    if (globalThis.process?.versions?.node) { // Node (testy): fetch nie czyta file://
      const { readFileSync } = await import("node:fs");
      return factory({ wasmBinary: readFileSync(wasmUrl) });
    }
    return factory();
  })();
  return librawPromise;
}

/**
 * Dekoduje plik RAW do liniowego, planarnego BGR.
 * @param args { buffer: ArrayBuffer, name, lensCorrection, jpegDecoder?: URL modułu z decodeJpeg() }
 * @param alloc (n) => Float32Array — tablice wyniku (współdzielone w puli workerów)
 * @returns { b, g, r, width, height, profile, preview, gray }
 *   profile — profil obiektywu Sony albo null; przy lensCorrection bez profilu: preview
 *   (wynik previewGray albo null) i gray (wynik rawGray) do pomiaru korekcji z podglądu
 */
export async function decodeRaw({ buffer, name, lensCorrection, jpegDecoder = DEFAULT_JPEG_DECODER }, alloc) {
  const profile = lensCorrection ? readLensProfile(buffer) : null;
  const LibRaw = await loadLibRaw();
  const raw = new LibRaw.LibRaw();
  let out, thumb = null, flip = 0;
  try {
    raw.open(new Uint8Array(buffer), DECODE_SETTINGS);
    const img = raw.imageData();
    if (img.bits !== 16 || img.colors !== 3) throw new Error(`${name}: nieoczekiwany format: ${img.bits} bit, ${img.colors} kanały`);
    const n = img.width * img.height;
    out = linearize(img.data, { b: alloc(n), g: alloc(n), r: alloc(n), width: img.width, height: img.height });
    if (lensCorrection && !profile) {
      try { thumb = raw.thumbnailData(); } catch { /* brak podglądu */ }
      flip = raw.metadata().flip;
    }
  } finally {
    raw.delete?.();
  }
  out.profile = profile;
  out.preview = null;
  if (thumb?.format === "jpeg") {
    const { decodeJpeg } = await import(jpegDecoder);
    out.preview = previewGray(await decodeJpeg(thumb.data), flip, out.width, out.height, alloc);
  }
  if (lensCorrection && !profile) out.gray = await rawGray(out, alloc);
  return out;
}
