// Uruchamia pipeline JS w Node na folderze z bracketem i zapisuje wynik jako surowe RGBA (+ .json z rozmiarem).
// Użycie: node tests/run_js.mjs <folder_z_raw> <wyjście_bez_rozszerzenia> [--lens-correction] [--auto-wb]
//         [--threads N]  (0 = wszystko w jednym wątku; domyślnie liczba rdzeni, maks. 16)
//         [--timing]     (czas każdego kroku)
import fs from "fs";
import path from "path";
import os from "os";
import { Worker } from "worker_threads";
import { mergeBracket } from "../js/merge.js";
import { PoolExec, InlineExec, COMPUTE_WORKER_URL } from "../js/exec.js";
import { isRaw } from "../js/formats.js";

const [dir, out, ...flags] = process.argv.slice(2);
const flagValue = (name, def) => (flags.includes(name) ? Number(flags[flags.indexOf(name) + 1]) : def);
const threads = flagValue("--threads", Math.min(os.availableParallelism(), 16));

function nodeWorker(url) {
  const w = new Worker(url);
  return {
    post: (msg, transfer) => w.postMessage(msg, transfer),
    onMessage: (fn) => w.on("message", fn),
    onError: (fn) => w.on("error", fn),
    terminate: () => w.terminate(),
  };
}
const spawn = () => nodeWorker(COMPUTE_WORKER_URL);
const exec = threads ? new PoolExec(spawn, threads) : new InlineExec();
const decoders = threads ? (n) => new PoolExec(spawn, Math.min(n, threads)) : null;

const files = fs.readdirSync(dir).filter(isRaw).sort()
  .map((name) => { const b = fs.readFileSync(path.join(dir, name)); return { name, buffer: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) }; });
const t0 = performance.now();
let stage = null, tStage = t0;
const progress = (text) => {
  const now = performance.now();
  if (flags.includes("--timing") && stage) console.log(`  ${stage}: ${((now - tStage) / 1000).toFixed(2)} s, rss ${(process.memoryUsage().rss / 1e9).toFixed(2)} GB`);
  stage = text; tStage = now;
};
const { rgba, width, height, warnings } = await mergeBracket(files, {
  exec, decoders, jpegDecoder: new URL("./jpeg_node.mjs", import.meta.url).href,
  lensCorrection: flags.includes("--lens-correction"), autoWb: flags.includes("--auto-wb"),
}, progress);
progress(null);
exec.close();
warnings.forEach((w) => console.warn(w));
fs.writeFileSync(out + ".rgba", rgba);
fs.writeFileSync(out + ".json", JSON.stringify({ width, height }));
console.log(`${path.basename(dir)}: ${width}x${height} in ${((performance.now() - t0) / 1000).toFixed(1)} s (${threads || 1} threads)`);
