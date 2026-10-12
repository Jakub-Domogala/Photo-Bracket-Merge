// Web Worker: dekodowanie i łączenie bracketów poza wątkiem UI. Ciężkie kroki rozdziela
// między pulę workerów obliczeń (js/compute-worker.js), a gdy nie ma SharedArrayBuffer —
// liczy wszystko tutaj.
// Wejście:  { groups: [[{ name, buffer }]], opts: { lensCorrection, autoWb } }
// Wyjście:  { type: "progress", text } | { type: "result", name, blob, warnings } | { type: "done" } | { type: "error", message }

import { mergeBracket } from "./merge.js";
import { PoolExec, InlineExec, browserWorker, canShare, COMPUTE_WORKER_URL } from "./exec.js";

const JPEG_QUALITY = 0.95;
const MAX_WORKERS = 16;

async function encodeJpeg(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
}

self.onmessage = async ({ data: { groups, opts } }) => {
  const threads = Math.min(navigator.hardwareConcurrency || 4, MAX_WORKERS);
  const spawn = () => browserWorker(COMPUTE_WORKER_URL);
  const exec = canShare ? new PoolExec(spawn, threads) : new InlineExec();
  // każdy dekoder to osobna instancja LibRaw (kilkaset MB), więc przy małej pamięci — po jednym
  const parallelDecodes = (navigator.deviceMemory ?? 8) <= 4 ? 1 : threads;
  const decoders = canShare ? (n) => new PoolExec(spawn, Math.min(n, parallelDecodes)) : null;
  try {
    for (const [k, group] of groups.entries()) {
      const prefix = groups.length > 1 ? `[${k + 1}/${groups.length}] ` : "";
      const progress = (text) => self.postMessage({ type: "progress", text: prefix + text });
      // setTimeout pozwala wiadomościom o postępie wyjść przed kolejnym długim krokiem
      await new Promise((r) => setTimeout(r));
      const { rgba, width, height, warnings } = await mergeBracket(group, { ...opts, exec, decoders }, progress);
      progress("Zapisywanie JPEG");
      const blob = await encodeJpeg(rgba, width, height);
      const name = `${group[0].name.replace(/\.[^.]+$/, "")}_${group.at(-1).name.replace(/\.[^.]+$/, "")}.jpg`;
      self.postMessage({ type: "result", name, blob, warnings });
      groups[k] = null;
    }
    self.postMessage({ type: "done" });
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message || String(err) });
  } finally {
    exec.close();
  }
};
