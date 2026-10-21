"""Porównuje wynik pipeline'u JS (tests/run_js.mjs) z referencją Python (reference/bracket_merge.py).

Użycie: python tests/compare.py <folder_z_raw> <wynik_js_bez_rozszerzenia> [--lens-correction] [--auto-wb]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
from bracket_merge import RAW_EXTENSIONS, merge  # noqa: E402

folder, js_out, *flags = sys.argv[1:]
paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in RAW_EXTENSIONS)
ref = merge(paths, 1.0, 1.0, 1.0, auto_wb="--auto-wb" in flags, lens_correction="--lens-correction" in flags)
ref16 = (ref * 65535).round().astype(np.uint16)
ref = (ref * 255).round().astype(np.uint8)[..., ::-1]  # BGR -> RGB

meta = json.loads(Path(js_out + ".json").read_text())
js = np.fromfile(js_out + ".rgba", np.uint8).reshape(meta["height"], meta["width"], 4)[..., :3]
assert js.shape == ref.shape, (js.shape, ref.shape)

d = np.abs(js.astype(np.int16) - ref.astype(np.int16))
print(f"{Path(folder).name} {' '.join(flags)}: mean |diff| {d.mean():.3f}/255, "
      f"99.9th pct {np.percentile(d, 99.9):.0f}, max {d.max()}, "
      f"pixels off by >2: {(d.max(-1) > 2).mean() * 100:.3f}%")

if Path(js_out + ".tif").exists():  # 16-bit TIFF z JS (--tiff) kontra referencja
    tif = cv2.imread(js_out + ".tif", cv2.IMREAD_UNCHANGED)  # BGR, uint16
    assert tif is not None and tif.dtype == np.uint16 and tif.shape == ref16.shape, (None if tif is None else (tif.dtype, tif.shape))
    d16 = np.abs(tif.astype(np.int32) - ref16.astype(np.int32))
    print(f"  TIFF 16-bit: mean |diff| {d16.mean():.2f}/65535, 99.9th pct {np.percentile(d16, 99.9):.0f}, max {d16.max()}")
