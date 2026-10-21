// Porównuje wyrównanie JS z referencją Python na danych z tests/dump_align.py:
// plan (macierze afiniczne, wycinek) z alignPlan i piksele z affineWarpRows.
// Użycie: node tests/check_align.mjs <prefiks_wyjścia_dump_align>
import fs from "fs";
import { InlineExec } from "../js/exec.js";
import { alignPlan, affineWarpRows } from "../js/align.js";

const base = process.argv[2];
const meta = JSON.parse(fs.readFileSync(base + ".json"));
const read = (f) => { const b = fs.readFileSync(f); return new Float32Array(b.buffer, b.byteOffset, b.length / 4); };

const grays = [0, 1, 2].map((i) => read(`${base}_${i}.f32`));
const plan = await alignPlan(new InlineExec(), grays, meta.w, meta.h);
if (!plan?.affines) { console.log('NIEZGODNE: brak planu z poprawką afiniczną', JSON.stringify(plan)); process.exit(1); }
const jsA = plan.affines.filter(Boolean), pyA = meta.fits.filter(([A]) => A).map(([A]) => A);
let planDiff = 0;
jsA.forEach((A, i) => A.forEach((v, j) => { planDiff = Math.max(planDiff, Math.abs(v - pyA[i][j])); }));
const sameCrop = [plan.x0, plan.y0, plan.w, plan.h].join() === meta.crop.join();
console.log(`plan: ${jsA.length}/${pyA.length} klatek z poprawką, max różnica macierzy ${planDiff.toExponential(1)}, wycinek ${sameCrop ? "taki sam" : "RÓŻNY"}`);

const [x0, y0, cw, ch] = meta.crop;
const src = { b: read(`${base}_src0.f32`), g: read(`${base}_src1.f32`), r: read(`${base}_src2.f32`) };
const dst = { b: new Float32Array(cw * ch), g: new Float32Array(cw * ch), r: new Float32Array(cw * ch) };
affineWarpRows({ src, dst, w: meta.w, h: meta.h, cw, x0, y0, A: meta.fits[0][0] }, 0, ch);
let worst = 0;
["b", "g", "r"].forEach((c, i) => {
  const ref = read(`${base}_dst${i}.f32`);
  for (let k = 0; k < ref.length; k++) worst = Math.max(worst, Math.abs(ref[k] - dst[c][k]));
});
console.log(`warp: max różnica pikseli względem cv2.remap ${worst.toExponential(1)} (skala 0-1)`);
if (planDiff > 1e-6 || !sameCrop || worst > 1e-4 || jsA.length !== pyA.length) { console.log("NIEZGODNE"); process.exit(1); }
console.log("zgodne");
