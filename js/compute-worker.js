// Worker obliczeń puli PoolExec (js/exec.js): wykonuje jądra z js/kernels.js.
// Wejście: { name, args, rows: [y0, y1] | null }  Wyjście: { result } | { error }
// Działa jako Web Worker i jako worker_threads w Node (testy).

import { kernels } from "./kernels.js";

const port = globalThis.WorkerGlobalScope ? self : (await import("node:worker_threads")).parentPort;

// tablice zwykłych (niewspółdzielonych) buforów w wyniku przekazujemy bez kopiowania
function transferables(value, out = new Set()) {
  if (ArrayBuffer.isView(value)) {
    if (value.buffer instanceof ArrayBuffer) out.add(value.buffer);
  } else if (value && typeof value === "object") {
    for (const v of Object.values(value)) transferables(v, out);
  }
  return out;
}

async function handle({ name, args, rows }) {
  try {
    const result = rows ? kernels[name](args, rows[0], rows[1]) : await kernels[name](args);
    port.postMessage({ result }, [...transferables(result)]);
  } catch (err) {
    port.postMessage({ error: err?.stack || err?.message || String(err) });
  }
}

if (port.on) port.on("message", handle);
else port.onmessage = (e) => handle(e.data);
