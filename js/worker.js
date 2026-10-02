// Web Worker: dekodowanie i łączenie bracketów poza wątkiem UI.
// Wejście:  { groups: [[{ name, buffer }]], opts: { lensCorrection, autoWb } }
// Wyjście:  { type: "progress", text } | { type: "result", name, blob, warnings } | { type: "done" } | { type: "error", message }

import factory from "../vendor/libraw-wasm/libraw.js";
import { mergeBracket } from "./merge.js";

const JPEG_QUALITY = 0.95;
const modulePromise = factory();

async function encodeJpeg(rgba, width, height) {
  const canvas = new OffscreenCanvas(width, height);
  canvas.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return canvas.convertToBlob({ type: "image/jpeg", quality: JPEG_QUALITY });
}

self.onmessage = async ({ data: { groups, opts } }) => {
  try {
    const LibRaw = await modulePromise;
    for (const [k, group] of groups.entries()) {
      const prefix = groups.length > 1 ? `[${k + 1}/${groups.length}] ` : "";
      const progress = (text) => self.postMessage({ type: "progress", text: prefix + text });
      // setTimeout pozwala wiadomościom o postępie wyjść przed kolejnym długim krokiem
      await new Promise((r) => setTimeout(r));
      const { rgba, width, height, warnings } = mergeBracket(LibRaw, group, opts, progress);
      progress("Zapisywanie JPEG");
      const blob = await encodeJpeg(rgba, width, height);
      const name = `${group[0].name.replace(/\.[^.]+$/, "")}_${group.at(-1).name.replace(/\.[^.]+$/, "")}.jpg`;
      self.postMessage({ type: "result", name, blob, warnings });
      groups[k] = null;
    }
    self.postMessage({ type: "done" });
  } catch (err) {
    self.postMessage({ type: "error", message: err?.message || String(err) });
  }
};
