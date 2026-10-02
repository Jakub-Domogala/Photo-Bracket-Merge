# Photo-Bracket-Merge

Merges bracketed RAW exposures (e.g. −2 / 0 / +2 EV, tripod) into one natural-looking image using Mertens exposure fusion. Built for interior / real-estate photos; no per-photo tuning.

**Everything runs in the browser** — photos are never uploaded. Drop the RAW files, click **Połącz**, then **Pobierz**. One bracket gives a JPEG; several give a ZIP.

Pipeline: RAW → linear RGB (camera WB) → optional lens correction from the Sony profile embedded in the ARW (Sony only) → common auto-exposure for the bracket → Mertens fusion → optional auto WB → tone finish (white stretch with soft highlight shoulder, black point, gentle S-curve, all on luminance so hues are preserved).

## Supported formats

Anything LibRaw decodes: Sony `.ARW/.SRF/.SR2`, Canon `.CR2/.CR3/.CRW`, Nikon `.NEF/.NRW`, Fujifilm `.RAF`, `.DNG`, Olympus/OM `.ORF`, Panasonic `.RW2`, Pentax `.PEF`, Samsung `.SRW`, Sigma `.X3F`, Hasselblad `.3FR/.FFF`, Phase One `.IIQ` and more (see `js/formats.js`). Tested against the Python reference with sample files from [raw.pixls.us](https://raw.pixls.us): ARW, CR2, CR3, NEF, RAF, DNG and RW2 match (≤ 0.002/255 mean difference). Olympus ORF differs slightly (≈ 2/255 mean, ~1 % colour) because the WASM LibRaw build decodes it a bit differently from rawpy's LibRaw 0.22.1 — the difference is already in the decoded RAW, not in the pipeline.

- Lens correction only works for Sony (it uses the profile stored in the ARW); other files are merged without it and the page says so.
- Fujifilm X-Trans `.RAF` files are much slower to decode (~45 s per frame); 50 MP files take about a minute per bracket.
- Very large files (e.g. 80 MP high-resolution-mode shots) need more memory than a browser tab usually gets.

## Layout

- `index.html`, `js/` — the web app (no build step, plain ES modules).
  - `js/pipeline.js` — the processing steps, a 1:1 port of the Python reference (Mertens mirrors OpenCV's implementation, including pyramid border handling).
  - `js/lens.js` — reads the Sony lens profile (ARW tags `0x7032/0x7035/0x7037`) and applies it.
  - `js/worker.js` — runs everything in a Web Worker; `js/app.js` — UI; `js/zip.js` — ZIP for several brackets.
- `js/formats.js` — accepted RAW extensions.
- `vendor/libraw-wasm/` — [LibRaw-Wasm](https://github.com/ybouane/LibRaw-Wasm) 1.6.0 (LibRaw compiled to WebAssembly) for RAW decoding.
- `coi-serviceworker.js` — [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) 0.1.7. LibRaw-Wasm needs `SharedArrayBuffer`, which requires COOP/COEP headers that GitHub Pages can't set; this service worker adds them.
- `reference/bracket_merge.py` — the original Python implementation (also a CLI), kept as the source of truth for tests.
- `tests/` — JS-vs-Python comparison.

## Run locally

Any static server works (the service worker handles isolation on `localhost` too):

```sh
python3 -m http.server 8000    # http://127.0.0.1:8000
```

## Verify the JS port against Python

```sh
python3.10 -m venv .venv && .venv/bin/pip install -r requirements.txt
node --max-old-space-size=8192 tests/run_js.mjs <folder_with_raw> /tmp/out [--lens-correction] [--auto-wb]
.venv/bin/python tests/compare.py <folder_with_raw> /tmp/out [--lens-correction] [--auto-wb]
```

On 14 test brackets the outputs match: mean difference ≤ 0.001/255 (0.03/255 with auto WB), ≤ 0.003 % of pixels off by more than 2 levels.

## Notes

- libraw-wasm ignores the `gamm` setting and always outputs LibRaw's BT.709 curve; `js/pipeline.js` rebuilds LibRaw's exact `gamma_curve` and inverts it to get linear data.
- Full-resolution processing (24 MP × 3) needs ~2.5 GB RAM in the tab and ~10–15 s per bracket.
