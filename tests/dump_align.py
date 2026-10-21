"""Przygotowuje dane do tests/check_align.mjs: bierze bracket z folderu, zniekształca klatki 0 i 2
znanym przekształceniem afinicznym (obrót do ±0.8°, skala ±1 %, pochylenie, przesunięcie do 20 px),
liczy plan wyrównania referencją Python i zapisuje log-szarości klatek, plan oraz wynik
warp_frame dla pierwszej klatki.

Użycie: python tests/dump_align.py <folder_z_raw> <prefiks_wyjścia> [ziarno]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "reference"))
import bracket_merge as bm  # noqa: E402

folder, out = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 1)
paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in bm.RAW_EXTENSIONS)[:3]
frames = [bm.load_raw(p) for p in paths]
h, w = frames[0].shape[:2]


def random_affine():
    ang = np.deg2rad(rng.uniform(-.8, .8))
    sc, shear, shift = 1 + rng.uniform(-.01, .01), rng.uniform(-.004, .004), rng.uniform(-20, 20, 2)
    c = np.array([w / 2 - .5, h / 2 - .5])
    L = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) * sc @ np.array([[1, shear], [0, 1]])
    return np.hstack([L, (c - L @ c + shift)[:, None]])


truth = {}
for k in (0, 2):
    truth[k] = random_affine()
    frames[k] = cv2.warpAffine(frames[k], truth[k], (w, h), flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_REFLECT)

rec = {"fits": []}
fit_affine, common_crop = bm.fit_affine, bm.common_crop


def spy_fit(*a, **kw):
    A, status = fit_affine(*a, **kw)
    rec["fits"].append([None if A is None else A.ravel().tolist(), status])
    return A, status


def spy_crop(*a, **kw):
    rec["crop"] = list(common_crop(*a, **kw))
    return rec["crop"]


bm.fit_affine, bm.common_crop = spy_fit, spy_crop
aligned = bm.align_frames(frames, "dump_align")
rec["out_shape"] = list(aligned[0].shape)

for i, img in enumerate(frames):
    np.log(np.maximum(img.mean(-1), np.float32(1e-6))).astype(np.float32).tofile(f"{out}_{i}.f32")
A = np.array(rec["fits"][0][0]).reshape(2, 3)
x0, y0, cw, ch = rec["crop"]
warped = bm.warp_frame(frames[0], A, x0, y0, cw, ch)
for c in range(3):
    np.ascontiguousarray(frames[0][..., c]).tofile(f"{out}_src{c}.f32")
    np.ascontiguousarray(warped[..., c]).tofile(f"{out}_dst{c}.f32")
json.dump({"w": w, "h": h, **rec}, open(out + ".json", "w"))

# jak dobrze referencja odzyskała prawdziwe przekształcenie (rogi, piksele pełnej rozdzielczości)
pts = np.array([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1], [w / 2, h / 2]])
for (k, M), (fit, status) in zip(truth.items(), rec["fits"]):
    inv = cv2.invertAffineTransform(M)
    est = np.array(fit).reshape(2, 3) if fit else None
    err = np.abs(pts @ est[:, :2].T + est[:, 2] - (pts @ inv[:, :2].T + inv[:, 2])).max() if est is not None else float("nan")
    print(f"klatka {k}: {status}, błąd względem prawdy {err:.2f} px (szerokość {w} px)")
