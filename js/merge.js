// Cały bracket: dekodowanie -> (korekcja obiektywu) -> auto-ekspozycja -> Mertens
// -> (auto WB) -> wykończenie. Odpowiednik merge() z reference/bracket_merge.py.

import { autoExposure, mergeMertens, autoWhiteBalance, finish, KEY } from "./pipeline.js";
import { correctLens } from "./lens.js";
import { fitPreviewCorrection, previewCorrectionMap, applyPreviewCorrection, rawGray } from "./previewLens.js";
import { InlineExec } from "./exec.js";

/**
 * @param files     [{ name, buffer: ArrayBuffer }] — klatki jednego bracketu (bufory są przekazywane
 *                  workerom dekodującym i potem nieużywalne)
 * @param opts      { lensCorrection, autoWb, key,
 *                    exec      — wykonawca obliczeń (js/exec.js), domyślnie w bieżącym wątku,
 *                    decoders  — (n) => wykonawca do dekodowania n klatek naraz (zamykany po dekodowaniu),
 *                    jpegDecoder — URL modułu z decodeJpeg() dla podglądów (domyślnie js/jpeg.js) }
 * @returns         { rgba: Uint8ClampedArray, width, height, warnings: string[] }
 */
export async function mergeBracket(files, opts = {}, progress = () => {}) {
  const exec = opts.exec ?? new InlineExec();
  const warnings = [];

  progress(files.length > 1 ? `Dekodowanie ${files.length} plików` : `Dekodowanie ${files[0].name}`);
  const decoder = opts.decoders ? opts.decoders(files.length) : exec;
  let frames;
  try {
    frames = await decoder.each("decode", files.map((f) => ({
      buffer: f.buffer, name: f.name, lensCorrection: !!opts.lensCorrection,
      jpegDecoder: opts.jpegDecoder, shared: decoder !== exec || !(exec instanceof InlineExec),
    })), files.map((f) => [f.buffer]));
  } finally {
    if (decoder !== exec) decoder.close();
  }
  let { width, height } = frames[0];
  for (const [i, img] of frames.entries()) {
    if (img.width !== width || img.height !== height) throw new Error(`${files[i].name}: inny rozmiar niż reszta bracketu`);
  }

  if (opts.lensCorrection) {
    // profil Sony z ARW, a gdy go nie ma — korekcja zmierzona z podglądu JPEG w pliku RAW
    if (frames.every((img) => img.profile)) {
      progress("Korekcja obiektywu");
      for (let i = 0; i < frames.length; i++) {
        const img = frames[i];
        frames[i] = await correctLens(exec, img, img.profile);
        exec.release(img.b, img.g, img.r);
      }
    } else {
      for (const img of frames) img.gray ??= await rawGray(img, (n) => exec.f32(n));
      const geo = await fitPreviewCorrection(exec, frames, frames.map((img) => img.preview), progress);
      for (const img of frames) {
        exec.release(img.gray.lin, img.gray.srgb, img.preview?.gray);
        img.gray = img.preview = null;
      }
      if (!geo) {
        warnings.push(`${files[0].name}…: brak profilu obiektywu i nie udało się zmierzyć korekcji z podglądu, pominięto korekcję.`);
      } else {
        progress("Korekcja obiektywu");
        const map = await previewCorrectionMap(exec, geo);
        for (let i = 0; i < frames.length; i++) {
          const img = frames[i];
          frames[i] = await applyPreviewCorrection(exec, img, geo, map);
          exec.release(img.b, img.g, img.r);
        }
        exec.release(map.mapX, map.mapY, map.gain);
        ({ width, height } = geo);
      }
    }
  }

  progress("Auto-ekspozycja");
  await autoExposure(exec, frames, opts.key ?? KEY);
  let result = await mergeMertens(exec, frames, width, height, {}, progress);
  if (opts.autoWb) {
    progress("Auto balans bieli");
    result = await autoWhiteBalance(exec, result, width, height);
  }
  progress("Wykończenie");
  return { rgba: await finish(exec, result, width, height), width, height, warnings };
}
