"""Porównuje wynik pipeline'u JS (tests/run_js.mjs) z referencją Python (reference/bracket_merge.py).

Użycie: python tests/compare.py <folder_z_arw> <wynik_js_bez_rozszerzenia> [--lens-correction] [--auto-wb]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
from bracket_merge import merge  # noqa: E402

folder, js_out, *flags = sys.argv[1:]
paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() == ".arw")
ref = merge(paths, 1.0, 1.0, 1.0, auto_wb="--auto-wb" in flags, lens_correction="--lens-correction" in flags)
ref = (ref * 255).round().astype(np.uint8)[..., ::-1]  # BGR -> RGB

meta = json.loads(Path(js_out + ".json").read_text())
js = np.fromfile(js_out + ".rgba", np.uint8).reshape(meta["height"], meta["width"], 4)[..., :3]
assert js.shape == ref.shape, (js.shape, ref.shape)

d = np.abs(js.astype(np.int16) - ref.astype(np.int16))
print(f"{Path(folder).name} {' '.join(flags)}: mean |diff| {d.mean():.3f}/255, "
      f"99.9th pct {np.percentile(d, 99.9):.0f}, max {d.max()}, "
      f"pixels off by >2: {(d.max(-1) > 2).mean() * 100:.3f}%")
