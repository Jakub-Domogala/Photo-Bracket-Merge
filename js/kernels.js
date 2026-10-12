// Rejestr jąder dla wykonawców z js/exec.js. Jądra wierszowe mają postać (args, y0, y1);
// pozostałe (args) => wynik albo Promise. Argumenty i wyniki muszą dać się sklonować
// (postMessage), a tablice, do których jądra piszą, muszą być współdzielone (exec.f32).

import * as P from "./pipeline.js";
import { sonyVignetteRows, sonyRemapRows } from "./lens.js";
import { matchPair, vignetteSamples, previewMapRows, previewRemapRows } from "./previewLens.js";
import { decodeRaw } from "./decode.js";

const shared = (n) => new Float32Array(new SharedArrayBuffer(n * 4));

export const kernels = {
  lum: P.lumRows,
  minMax: P.minMaxRows,
  hist: P.histRows,
  logSum: P.logSumRows,
  srgb: P.srgbRows,
  pyrDownH: P.pyrDownHRows,
  pyrDownV: P.pyrDownVRows,
  pyrUpH: P.pyrUpHRows,
  pyrUpV: P.pyrUpVRows,
  gray: P.grayRows,
  weight: P.weightRows,
  normalize: P.normalizeRows,
  fma: P.fmaRows,
  clip: P.clipRows,
  awbPrep: P.awbPrepRows,
  awbMask: P.awbMaskRows,
  awbAvg: P.awbAvgRows,
  awbScale: P.awbScaleRows,
  awbOut: P.awbOutRows,
  shoulder: P.shoulderRows,
  finish: P.finishRows,
  sonyVignette: sonyVignetteRows,
  sonyRemap: sonyRemapRows,
  previewMap: previewMapRows,
  previewRemap: previewRemapRows,
  matchPair,
  vignetteSamples,
  // w workerze wynik trafia do pamięci współdzielonej, w bieżącym wątku — do zwykłej
  decode: (args) => decodeRaw(args, args.shared ? shared : (n) => new Float32Array(n)),
};
