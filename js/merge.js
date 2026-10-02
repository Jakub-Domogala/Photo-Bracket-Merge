// Cały bracket: dekodowanie -> (korekcja obiektywu) -> auto-ekspozycja -> Mertens
// -> (auto WB) -> wykończenie. Odpowiednik merge() z reference/bracket_merge.py.

import { linearize, autoExposure, mergeMertens, autoWhiteBalance, finish, KEY } from "./pipeline.js";
import { readLensProfile, correctLens } from "./lens.js";

// Ustawienia jak rawpy.postprocess w wersji Python (gamma i tak jest odwracana w linearize)
const DECODE_SETTINGS = { useCameraWb: true, noAutoBright: true, outputBps: 16, outputColor: 1 };

/** Dekoduje ARW modułem LibRaw (libraw-wasm) do liniowego, planarnego BGR. */
export function decodeArw(LibRawModule, buffer) {
  const raw = new LibRawModule.LibRaw();
  try {
    raw.open(new Uint8Array(buffer), DECODE_SETTINGS);
    const img = raw.imageData();
    if (img.bits !== 16 || img.colors !== 3) throw new Error(`Nieoczekiwany format: ${img.bits} bit, ${img.colors} kanały`);
    return linearize(img.data, img.width, img.height);
  } finally {
    raw.delete?.();
  }
}

/**
 * @param files     [{ name, buffer: ArrayBuffer }] — klatki jednego bracketu
 * @param opts      { lensCorrection, autoWb, key }
 * @returns         { rgba: Uint8ClampedArray, width, height, warnings: string[] }
 */
export function mergeBracket(LibRawModule, files, opts = {}, progress = () => {}) {
  const warnings = [];
  const frames = [];
  let width, height;
  for (const [i, f] of files.entries()) {
    progress(`Dekodowanie ${f.name} (${i + 1}/${files.length})`);
    let img = decodeArw(LibRawModule, f.buffer);
    if (width === undefined) ({ width, height } = img);
    else if (img.width !== width || img.height !== height) throw new Error(`${f.name}: inny rozmiar niż reszta bracketu`);
    if (opts.lensCorrection) {
      const profile = readLensProfile(f.buffer);
      if (!profile) warnings.push(`${f.name} nie ma profilu obiektywu, pominięto korekcję.`);
      else {
        progress(`Korekcja obiektywu ${f.name}`);
        img = correctLens(img, profile, width, height);
      }
    }
    frames.push(img);
  }

  progress("Auto-ekspozycja");
  autoExposure(frames, opts.key ?? KEY);
  let result = mergeMertens(frames, width, height, {}, progress);
  if (opts.autoWb) {
    progress("Auto balans bieli");
    result = autoWhiteBalance(result);
  }
  progress("Wykończenie");
  return { rgba: finish(result, width, height), width, height, warnings };
}
