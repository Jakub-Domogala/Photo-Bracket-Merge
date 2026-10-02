// Uruchamia pipeline JS w Node na folderze z bracketem i zapisuje wynik jako surowe RGBA (+ .json z rozmiarem).
// Użycie: node tests/run_js.mjs <folder_z_raw> <wyjście_bez_rozszerzenia> [--lens-correction] [--auto-wb]
import fs from "fs";
import path from "path";
import factory from "../vendor/libraw-wasm/libraw.js";
import { mergeBracket } from "../js/merge.js";
import { isRaw } from "../js/formats.js";

const [dir, out, ...flags] = process.argv.slice(2);
const M = await factory({ wasmBinary: fs.readFileSync(new URL("../vendor/libraw-wasm/libraw.wasm", import.meta.url)) });
const files = fs.readdirSync(dir).filter(isRaw).sort()
  .map((name) => { const b = fs.readFileSync(path.join(dir, name)); return { name, buffer: b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) }; });
const t0 = Date.now();
const { rgba, width, height, warnings } = mergeBracket(M, files, {
  lensCorrection: flags.includes("--lens-correction"), autoWb: flags.includes("--auto-wb"),
});
warnings.forEach((w) => console.warn(w));
fs.writeFileSync(out + ".rgba", rgba);
fs.writeFileSync(out + ".json", JSON.stringify({ width, height }));
console.log(`${path.basename(dir)}: ${width}x${height} in ${((Date.now() - t0) / 1000).toFixed(1)} s`);
