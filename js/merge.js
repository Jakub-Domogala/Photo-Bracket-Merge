// Cały bracket: dekodowanie -> (korekcja obiektywu) -> auto-ekspozycja -> Mertens
// -> (auto WB) -> wykończenie. Odpowiednik merge() z reference/bracket_merge.py.

import { linearize, autoExposure, mergeMertens, autoWhiteBalance, finish, KEY } from "./pipeline.js";
import { readLensProfile, correctLens } from "./lens.js";
import { previewGray, fitPreviewCorrection, applyPreviewCorrection } from "./previewLens.js";

// Ustawienia jak rawpy.postprocess w wersji Python (gamma i tak jest odwracana w linearize)
const DECODE_SETTINGS = { useCameraWb: true, noAutoBright: true, outputBps: 16, outputColor: 1 };

/** Dekoduje plik RAW modułem LibRaw (libraw-wasm) do liniowego, planarnego BGR.
 * Z `withPreview` dołącza też podgląd JPEG z pliku ({ jpeg: Uint8Array, flip } albo null). */
export function decodeRaw(LibRawModule, buffer, withPreview = false) {
  const raw = new LibRawModule.LibRaw();
  try {
    raw.open(new Uint8Array(buffer), DECODE_SETTINGS);
    const img = raw.imageData();
    if (img.bits !== 16 || img.colors !== 3) throw new Error(`Nieoczekiwany format: ${img.bits} bit, ${img.colors} kanały`);
    const out = linearize(img.data, img.width, img.height);
    if (withPreview) {
      let thumb = null;
      try { thumb = raw.thumbnailData(); } catch { /* brak podglądu */ }
      out.preview = thumb?.format === "jpeg" ? { jpeg: thumb.data, flip: raw.metadata().flip } : null;
    }
    return out;
  } finally {
    raw.delete?.();
  }
}

/**
 * @param files     [{ name, buffer: ArrayBuffer }] — klatki jednego bracketu
 * @param opts      { lensCorrection, autoWb, key, decodeJpeg }
 *                  decodeJpeg(Uint8Array) -> Promise<{ data: RGBA, width, height }> — potrzebny
 *                  do korekcji obiektywu aparatów bez profilu Sony
 * @returns         { rgba: Uint8ClampedArray, width, height, warnings: string[] }
 */
export async function mergeBracket(LibRawModule, files, opts = {}, progress = () => {}) {
  const warnings = [];
  let frames = [];
  let width, height;
  for (const [i, f] of files.entries()) {
    progress(`Dekodowanie ${f.name} (${i + 1}/${files.length})`);
    const img = decodeRaw(LibRawModule, f.buffer, opts.lensCorrection);
    if (width === undefined) ({ width, height } = img);
    else if (img.width !== width || img.height !== height) throw new Error(`${f.name}: inny rozmiar niż reszta bracketu`);
    frames.push(img);
  }

  if (opts.lensCorrection) {
    // profil Sony z ARW, a gdy go nie ma — korekcja zmierzona z podglądu JPEG w pliku RAW
    const profiles = files.map((f) => readLensProfile(f.buffer));
    if (profiles.every(Boolean)) {
      frames = frames.map((img, i) => {
        progress(`Korekcja obiektywu ${files[i].name}`);
        return correctLens(img, profiles[i], width, height);
      });
    } else {
      progress("Wczytywanie podglądów JPEG");
      const previews = [];
      for (const img of frames) {
        previews.push(img.preview && opts.decodeJpeg
          ? previewGray(await opts.decodeJpeg(img.preview.jpeg), img.preview.flip, width, height)
          : null);
        img.preview = null;
      }
      const geo = fitPreviewCorrection(frames, previews, progress);
      if (!geo) {
        warnings.push(`${files[0].name}…: brak profilu obiektywu i nie udało się zmierzyć korekcji z podglądu, pominięto korekcję.`);
      } else {
        frames = frames.map((img, i) => {
          progress(`Korekcja obiektywu ${files[i].name}`);
          return applyPreviewCorrection(img, geo);
        });
        ({ width, height } = geo);
      }
    }
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
