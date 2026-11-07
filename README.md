# Photo-Bracket-Merge

Merges bracketed RAW exposures (e.g. −2 / 0 / +2 EV, tripod) into one natural-looking image using Mertens exposure fusion. Built for interior / real-estate photos; no per-photo tuning.

**Everything runs in the browser** — photos are never uploaded. Drop the RAW files, click **Połącz**, then **Pobierz**. One bracket gives a JPEG; several give a ZIP. Tick **TIFF 16-bit** to also get an uncompressed 16-bit sRGB TIFF (~150 MB per 24 MP photo) for further editing: a second button **Pobierz TIFF 16-bit** appears next to the JPEG one (a separate ZIP when there are several brackets). Both files come from the same finished floating-point image, so the TIFF is the JPEG without the 8-bit rounding.

Pipeline: RAW → linear RGB (camera WB) → optional lens correction (Sony: profile embedded in the ARW; other cameras: measured from the camera's JPEG preview inside the RAW) → alignment of frames moved between shots (whole-pixel shift from median threshold bitmaps, then an affine refinement for tilt, rotation, scale and sub-pixel shifts; crops at most 3 %, otherwise skipped) → common auto-exposure for the bracket → Mertens fusion → optional auto WB → tone finish (white stretch with soft highlight shoulder, black point, gentle S-curve, all on luminance so hues are preserved).

## Supported formats

Anything LibRaw decodes: Sony `.ARW/.SRF/.SR2`, Canon `.CR2/.CR3/.CRW`, Nikon `.NEF/.NRW`, Fujifilm `.RAF`, `.DNG`, Olympus/OM `.ORF`, Panasonic `.RW2`, Pentax `.PEF`, Samsung `.SRW`, Sigma `.X3F`, Hasselblad `.3FR/.FFF`, Phase One `.IIQ` and more (see `js/formats.js`). Tested against the Python reference with sample files from [raw.pixls.us](https://raw.pixls.us): ARW, CR2, CR3, NEF, RAF, DNG and RW2 match (≤ 0.002/255 mean difference). Olympus ORF differs slightly (≈ 2/255 mean, ~1 % colour) because the WASM LibRaw build decodes it a bit differently from rawpy's LibRaw 0.22.1 — the difference is already in the decoded RAW, not in the pipeline.

- Lens correction: Sony ARW files carry the camera's correction profile, which is applied directly. Other brands don't store coefficients, but most RAW files embed a full-size JPEG preview the camera already corrected (distortion, vignetting, crop). The app matches that preview against the RAW (patch matching at 1/16, 1/4 and 1/2 resolution, pooled over the bracket), fits a radial model and applies it, so the result has the camera JPEG's framing. Needs a preview at least 75 % of the RAW size; otherwise the page says the correction was skipped. This matters most for compacts like the Canon G7 X, whose RAWs are strongly distorted with dark corners.
- Fujifilm X-Trans `.RAF` files are much slower to decode (~45 s per frame); 50 MP files take about a minute per bracket.
- Very large files (e.g. 80 MP high-resolution-mode shots) need more memory than a browser tab usually gets.

## Layout

- `index.html`, `src/` — the UI: Vite + React + Tailwind 4 + shadcn/ui (base-luma), with a Buy-me-a-coffee footer. `pnpm dev` / `pnpm build` (deployed to GitHub Pages by `.github/workflows/deploy.yml`).
- `public/engine/` — the processing engine, served as-is (not bundled: LibRaw-Wasm spawns its own workers) and loaded by `src/lib/engine.ts`. Plain ES modules; the paths below are relative to it.
  - `js/pipeline.js` — the processing steps, a 1:1 port of the Python reference (Mertens mirrors OpenCV's implementation, including pyramid border handling).
  - `js/exec.js`, `js/kernels.js`, `js/compute-worker.js` — parallel execution. Heavy loops are written as row kernels `(args, y0, y1)`; `PoolExec` splits the rows across a pool of workers (one per CPU core, images in `SharedArrayBuffer`), `InlineExec` runs the same kernels on one thread when shared memory isn't available. Both give the same output.
  - `js/gpu.js` — WebGPU version of lens correction, auto-exposure and Mertens fusion (used when available; on any GPU error the CPU path runs instead). Output differs from the CPU path only by float32 rounding (≤ 0.002/255 mean, ~0.0002 % of pixels off by more than 1 level). Add `?cpu` to the page URL to force the CPU path.
  - `js/align.js` — alignment of frames moved between shots (no UI option, always on). Step 1 finds a whole-pixel shift (max 3 % crop). Step 2 refines it with an affine fit (Gauss-Newton on the gradient of log luminance, 3-level pyramid at ≤ 2000 px, clipped and near-black pixels masked, so frames of different exposure compare directly). The fit is only applied if it moves a corner by ≥ 0.5 px, so tripod brackets keep the exact pixel-shift crop; implausible fits (> 3° rotation, > 5 % scale, correlation below the unrefined one) are ignored, with a warning when the rotation/scale is out of range. Warped frames are resampled with bicubic interpolation (WebGPU kernel `affine` / CPU `affineWarp`).
  - `js/decode.js` — RAW decoding (each frame of a bracket in its own worker); `js/jpeg.js` — preview JPEG decoding.
  - `js/lens.js` — reads the Sony lens profile (ARW tags `0x7032/0x7035/0x7037`) and applies it.
  - `js/previewLens.js` — lens correction measured from the embedded JPEG preview (non-Sony cameras).
  - `js/worker.js` — runs everything off the UI thread and owns the worker pool; `js/zip.js` — ZIP for several brackets; `js/tiff.js` — minimal 16-bit RGB TIFF writer (little-endian, uncompressed, one strip).
- `js/formats.js` — accepted RAW extensions.
- `vendor/libraw-wasm/` — [LibRaw-Wasm](https://github.com/ybouane/LibRaw-Wasm) 1.6.0 (LibRaw compiled to WebAssembly) for RAW decoding.
- `public/coi-serviceworker.js` — [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) 0.1.7. LibRaw-Wasm needs `SharedArrayBuffer`, which requires COOP/COEP headers that GitHub Pages can't set; this service worker adds them.
- `reference/bracket_merge.py` — the original Python implementation (also a CLI), kept as the source of truth for tests.
- `tests/` — JS-vs-Python comparison (`tests/vendor/` holds the jpeg-js decoder Node needs for previews).

## Run locally

```sh
pnpm install && pnpm dev    # http://localhost:5173
```

## Verify the JS port against Python

```sh
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
node --max-old-space-size=8192 tests/run_js.mjs <folder_with_raw> /tmp/out [--lens-correction] [--auto-wb] [--threads N] [--timing] [--tiff]
.venv/bin/python tests/compare.py <folder_with_raw> /tmp/out [--lens-correction] [--auto-wb]
```

Alignment check against the reference on a bracket whose frames are distorted by a known transform (rotation up to 0.8°, shear, scale, 20 px shift; recovered to ≤ 0.6 px at 6000 px width):

```sh
.venv/bin/python tests/dump_align.py <folder_with_raw> /tmp/align [seed]
node tests/check_align.mjs /tmp/align      # plan and warped pixels must match cv2 (≤ 1e-6 / 1e-4)
```

With `--tiff` the 16-bit TIFF is written too and `compare.py` checks it against the Python 16-bit result (mean difference ≈ 0.2/65535 on photo7).

On 14 test brackets the outputs match: mean difference ≤ 0.001/255 (0.03/255 with auto WB), ≤ 0.003 % of pixels off by more than 2 levels.

## Notes

- libraw-wasm ignores the `gamm` setting and always outputs LibRaw's BT.709 curve; `js/pipeline.js` rebuilds LibRaw's exact `gamma_curve` and inverts it to get linear data.
- The preview-based fit depends on the JPEG decoder: in Node (jpeg-js) results drift slightly from Python (≈1/255 mean, sub-pixel geometry); in Chrome (libjpeg-turbo, like OpenCV) they match to ≈0.4/255. With the same fitted parameters JS and Python agree to 0.001/255.
- Full-resolution processing (24 MP × 3) needs ~3 GB RAM in the tab. Speed for one 3-frame bracket (Chrome, 16-core Apple Silicon Mac, median of 3 runs):

  | Bracket | before (1 thread) | CPU worker pool | + WebGPU |
  |---|---|---|---|
  | Sony 24 MP + lens profile | 23.3 s | 3.1 s | 1.8 s |
  | Canon G7 X 20 MP + preview lens correction | 18.1 s | 3.7 s | 3.0 s |
  | Sony 24 MP + auto WB | 14.2 s | 2.6 s | 2.3 s |

  What's left is mostly RAW decoding (LibRaw, ~1.3–1.8 s, frames in parallel) and, for non-Sony cameras, the preview fit (~0.6 s); the GPU part takes ~0.25 s. With `navigator.deviceMemory` ≤ 4 GB frames are decoded one at a time to limit memory.
