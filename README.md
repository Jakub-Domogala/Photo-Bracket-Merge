# Photo-Bracket-Merge

Merges bracketed Sony `.ARW` exposures (e.g. −2 / 0 / +2 EV, tripod) into one natural-looking image using Mertens exposure fusion. Built for interior / real-estate photos; no per-photo tuning.

Pipeline: RAW → linear RGB (camera WB) → optional lens correction from the Sony profile embedded in the ARW → common auto-exposure for the bracket → Mertens fusion → optional auto WB → tone finish (white stretch with soft highlight shoulder, black point, gentle S-curve, all on luminance so hues are preserved).

## Setup

```sh
python3.10 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Web app

```sh
.venv/bin/python app.py    # http://127.0.0.1:8000
```

Drop the `.ARW` files, click **Połącz**, then **Pobierz**. One bracket gives a JPEG; several give a ZIP.

## CLI

```sh
.venv/bin/python bracket_merge.py folder_with_arw -o results [--lens-correction] [--auto-wb] [-n 3]
```

Files are sorted by name and grouped every N (default 3); subfolders are processed too. Outputs a 16-bit TIFF (`--no-tiff` to skip) and a JPEG.
