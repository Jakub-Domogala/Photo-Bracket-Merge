// Cały bracket: dekodowanie -> (korekcja obiektywu) -> auto-ekspozycja -> Mertens
// -> (auto WB) -> wykończenie. Odpowiednik merge() z reference/bracket_merge.py.

import { autoExposure, mergeMertens, autoWhiteBalance, finish, KEY } from "./pipeline.js";
import { correctLens } from "./lens.js";
import { fitPreviewCorrection, previewCorrectionMap, applyPreviewCorrection, rawGray } from "./previewLens.js";
import { InlineExec } from "./exec.js";
import { getGpu, gpuMerge } from "./gpu.js";

/**
 * @param files     [{ name, buffer: ArrayBuffer }] — klatki jednego bracketu (bufory są przekazywane
 *                  workerom dekodującym i potem nieużywalne)
 * @param opts      { lensCorrection, autoWb, key,
 *                    exec      — wykonawca obliczeń (js/exec.js), domyślnie w bieżącym wątku,
 *                    decoders  — (n) => wykonawca do dekodowania n klatek naraz (zamykany po dekodowaniu),
 *                    jpegDecoder — URL modułu z decodeJpeg() dla podglądów (domyślnie js/jpeg.js),
 *                    gpu       — true: korekcja, auto-ekspozycja i fuzja na WebGPU, jeśli jest }
 * @returns         { rgba: Uint8ClampedArray, width, height, warnings: string[], gpu: czy użyto GPU }
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

  // plan korekcji obiektywu: profil Sony z ARW, a gdy go nie ma — korekcja zmierzona
  // z podglądu JPEG w pliku RAW
  let lens = null;
  if (opts.lensCorrection) {
    if (frames.every((img) => img.profile)) {
      lens = { profiles: frames.map((img) => img.profile) };
    } else {
      for (const img of frames) img.gray ??= await rawGray(img, (n) => exec.f32(n));
      const geo = await fitPreviewCorrection(exec, frames, frames.map((img) => img.preview), progress);
      for (const img of frames) {
        exec.release(img.gray.lin, img.gray.srgb, img.preview?.gray);
        img.gray = img.preview = null;
      }
      if (geo) lens = { geo };
      else warnings.push(`${files[0].name}…: brak profilu obiektywu i nie udało się zmierzyć korekcji z podglądu, pominięto korekcję.`);
    }
  }
  if (lens?.geo) ({ width, height } = lens.geo);
  const key = opts.key ?? KEY;

  // korekcja + auto-ekspozycja + Mertens: na GPU, jeśli jest, inaczej (albo po błędzie) na CPU
  let result = null, gpuUsed = false;
  const gpu = opts.gpu ? await getGpu() : null;
  if (gpu) {
    progress("Korekcja obiektywu (GPU)");
    try {
      result = await gpuMerge(gpu, frames, lens, key, (n) => exec.f32(n), progress);
      gpuUsed = true;
      for (const img of frames) exec.release(img.b, img.g, img.r);
    } catch (err) {
      console.warn("GPU niedostępne, liczę na CPU:", err);
    }
  }
  if (!result) {
    if (lens) progress("Korekcja obiektywu");
    const map = lens?.geo ? await previewCorrectionMap(exec, lens.geo) : null;
    for (let i = 0; i < frames.length; i++) {
      const img = frames[i];
      if (lens?.profiles) frames[i] = await correctLens(exec, img, lens.profiles[i]);
      else if (lens?.geo) frames[i] = await applyPreviewCorrection(exec, img, lens.geo, map);
      if (lens) exec.release(img.b, img.g, img.r);
    }
    if (map) exec.release(map.mapX, map.mapY, map.gain);
    progress("Auto-ekspozycja");
    await autoExposure(exec, frames, key);
    result = await mergeMertens(exec, frames, width, height, {}, progress);
  }
  if (opts.autoWb) {
    progress("Auto balans bieli");
    result = await autoWhiteBalance(exec, result, width, height);
  }
  progress("Wykończenie");
  return { rgba: await finish(exec, result, width, height), width, height, warnings, gpu: gpuUsed };
}
