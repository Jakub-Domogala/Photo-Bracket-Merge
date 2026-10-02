#!/usr/bin/env python3
"""
Łączy brackety ARW (Sony) w jedno zdjęcie metodą exposure fusion (Mertens).
Zakłada statyw i brak ruchu w kadrze, więc nie wykonuje wyrównywania.

Pipeline (bez ręcznego strojenia per zdjęcie):
  1. RAW -> liniowe RGB (16 bit, balans bieli z aparatu); opcjonalnie (--lens-correction)
     korekcja obiektywu z profilu Sony zapisanego w ARW (dystorsja, winieta, aberracja)
  2. auto-ekspozycja: wspólne wzmocnienie dla całego bracketu tak, żeby
     środkowa klatka miała średnią jasność (log-average) = KEY
  3. exposure fusion (Mertens) na klatkach w sRGB
  4. opcjonalnie (--auto-wb) auto balans bieli (jasne, mało nasycone piksele -> neutralna szarość);
     domyślnie zostaje balans bieli z aparatu
  5. wykończenie: rozjaśnienie bieli (ściany mają być białe, okna miękko
     kompresowane), punkt czerni + delikatna krzywa S

Pliki w folderze są sortowane po nazwie i grupowane po N (domyślnie 3),
np. DSC00001-3 -> wynik 1, DSC00004-6 -> wynik 2 itd. Podfoldery są
przetwarzane tak samo (np. data/input/photo1, data/input/photo2).

Instalacja:  pip install -r requirements.txt
Użycie:      python bracket_merge.py folder_z_arw -o wyniki
"""

import argparse
from pathlib import Path

import cv2
import exifread
import numpy as np
import rawpy

KEY = 0.12            # docelowa jasność (log-average, liniowo) środkowej klatki
BLACK_PCT = 0.5       # percentyl luminancji ustawiany jako czerń
S_CURVE = 0.3         # siła krzywej S (0 = brak)
WHITE_PCT = 99.5      # percentyl luminancji, który po auto WB nie może przekroczyć 1
BRIGHT_PCT = 90       # ten percentyl luminancji ...
BRIGHT_TARGET = 0.9   # ... rozciągamy do tej wartości (białe ściany zamiast szarych)
BRIGHT_MAX = 1.4      # maks. wzmocnienie (żeby nie przepalić scen nocnych)
KNEE = 0.75           # powyżej tego światła są miękko kompresowane zamiast obcinane


def lum(bgr: np.ndarray) -> np.ndarray:
    return 0.0722 * bgr[..., 0] + 0.7152 * bgr[..., 1] + 0.2126 * bgr[..., 2]


def to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4)).astype(np.float32)


def log_average(y: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.clip(y, 1e-4, None)))))


def load_raw(path: Path) -> np.ndarray:
    """Wywołuje ARW do liniowego 16-bit sRGB i zwraca float32 BGR w zakresie 0-1."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=True,     # jasność ustawiamy sami, wspólnie dla całego bracketu
            output_bps=16,
            gamma=(1, 1),            # liniowo — gamma dopiero po auto-ekspozycji
            output_color=rawpy.ColorSpace.sRGB,
        )
    rgb = rgb.astype(np.float32) / 65535.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def read_lens_profile(path: Path):
    """Profil korekcji obiektywu, który Sony zapisuje w każdym ARW (ten sam, którego aparat
    używa do JPEG-ów). Zwraca None, jeśli plik go nie ma (np. obiektyw manualny)."""
    with open(path, "rb") as f:
        tags = exifread.process_file(f, details=True)

    def curve(tag, parts=1):
        t = tags.get(f"EXIF SubIFD0 Tag {tag}")
        if t is None:
            return None
        n = t.values[0] // parts
        return [np.array(t.values[1 + i * n:1 + (i + 1) * n], np.float32) for i in range(parts)]

    vig, dist, ca = curve("0x7032"), curve("0x7037"), curve("0x7035", parts=2)
    if vig is None or dist is None:
        return None
    return {"vignetting": vig[0], "distortion": dist[0],
            "ca": ca or [np.zeros_like(dist[0])] * 2}


def fill_scale(curve: np.ndarray, cx: float, cy: float) -> float:
    """Największe powiększenie, przy którym brzeg wyniku nadal mieści się w surowym kadrze
    (bez pustych rogów) — tak samo kadruje aparat (zmierzone na JPEG-ach z a6400)."""
    t = np.linspace(-1, 1, 2001)
    edge = np.concatenate([np.stack([t * cx, np.full_like(t, cy)], 1),
                           np.stack([np.full_like(t, cx), t * cy], 1)])
    r = np.hypot(edge[:, 0], edge[:, 1]) / np.hypot(cx, cy)
    s = np.interp(r, np.linspace(0, 1, len(curve)), curve)
    sx = np.maximum(np.abs(edge[:, 0] * s), 1e-6)
    sy = np.maximum(np.abs(edge[:, 1] * s), 1e-6)
    return float(min((cx / sx).min(), (cy / sy).min()))


def correct_lens(img: np.ndarray, profile: dict) -> np.ndarray:
    """Stosuje profil Sony (wzory jak w darktable): krzywe zdefiniowane w 'nc' punktach
    na promieniu 0..1 (od środka do rogu). Obraz musi być liniowy."""
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2, (h - 1) / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx * dx + dy * dy) / np.hypot(cx, cy)

    def spline(values):
        return np.interp(r, np.linspace(0, 1, len(values)), values).astype(np.float32)

    # winieta: współczynnik spadku światła -> dzielimy, rogi się rozjaśniają
    falloff = spline(2 ** (0.5 - 2 ** (profile["vignetting"] * 2 ** -13 - 1)))
    img = img / falloff[..., None]

    # dystorsja (+ aberracja chromatyczna dla R i B): skąd w surowym obrazie wziąć piksel
    base = profile["distortion"] * 2 ** -14 + 1
    base = base * fill_scale(base, cx, cy)
    ca_r, ca_b = profile["ca"]
    scales = {2: base * (ca_r * 2 ** -21 + 1), 1: base, 0: base * (ca_b * 2 ** -21 + 1)}  # BGR
    out = np.empty_like(img)
    for c, sc in scales.items():
        s = spline(sc)
        out[..., c] = cv2.remap(img[..., c], cx + dx * s, cy + dy * s,
                                cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return np.clip(out, 0, None)


def auto_exposure(linear: list, key: float) -> list:
    """Jedno wzmocnienie dla wszystkich klatek (zachowuje odstępy EV), liczone ze środkowej
    pod względem jasności (a nie kolejności plików — Sony domyślnie robi 0, -, +)."""
    levels = sorted(log_average(lum(img)) for img in linear)
    gain = np.clip(key / levels[len(levels) // 2], 0.25, 32.0)
    return [to_srgb(img * gain) for img in linear]


def auto_white_balance(srgb: np.ndarray) -> np.ndarray:
    """Szacuje kolor oświetlenia z jasnych, mało nasyconych pikseli (białe ściany, szafki)."""
    lin = to_linear(srgb)
    y = lum(lin)
    hi = lin.max(-1)
    sat = (hi - lin.min(-1)) / (hi + 1e-6)
    mask = (y > np.percentile(y, 80)) & (y < np.percentile(y, 99)) & (hi < 0.97)
    if mask.sum() < 1000:
        return srgb
    mask &= sat < np.percentile(sat[mask], 50)
    avg = lin[mask].mean(axis=0)
    lin = lin * (avg.mean() / avg)
    lin /= max(1.0, float(np.percentile(lum(lin), WHITE_PCT)))
    return to_srgb(lin)


def shoulder(x: np.ndarray, knee: float = KNEE) -> np.ndarray:
    """Liniowo do 'knee', powyżej miękko dochodzi do 1 (okna nie wypalają się na płasko)."""
    soft = knee + (1 - knee) * (1 - np.exp(-(x - knee) / (1 - knee)))
    return np.where(x <= knee, x, soft).astype(np.float32)


def finish(srgb: np.ndarray) -> np.ndarray:
    """Mertens daje szare biele i 'mgiełkę': rozciągamy jasne partie, ustawiamy
    punkt czerni i dodajemy delikatną krzywą S.

    Krzywa działa na luminancji, a R/G/B są skalowane razem — osobno na kanałach
    ciepłe światło przesuwało się z pomarańczu w żółć (R dochodził do 1 pierwszy)."""
    y = lum(srgb)
    hi = max(float(np.percentile(y, BRIGHT_PCT)), 1e-3)
    t = shoulder(y * np.clip(BRIGHT_TARGET / hi, 1.0, BRIGHT_MAX))
    lo = np.percentile(t, BLACK_PCT)
    t = np.clip((t - lo) / (1 - lo), 0, 1)
    t = (1 - S_CURVE) * t + S_CURVE * (t * t * (3 - 2 * t))

    out = srgb * (t / np.maximum(y, 1e-4))[..., None]
    # kanały powyżej 1: zamiast obcinać (zmiana odcienia), mieszamy w stronę bieli
    # o tej samej jasności
    peak = out.max(-1, keepdims=True)
    w = np.clip((peak - 1) / np.maximum(peak - t[..., None], 1e-4), 0, 1)
    return out * (1 - w) + t[..., None] * w


def load_frame(path: Path, lens_correction: bool) -> np.ndarray:
    img = load_raw(path)
    if lens_correction:
        profile = read_lens_profile(path)
        if profile is None:
            print(f"  Uwaga: {path.name} nie ma profilu obiektywu, pomijam korekcję.")
        else:
            img = correct_lens(img, profile)
    return img


def merge(paths, contrast, saturation, exposure, key=KEY, auto_wb=False,
          lens_correction=False) -> np.ndarray:
    imgs = auto_exposure([load_frame(p, lens_correction) for p in paths], key)
    fusion = cv2.createMergeMertens(
        contrast_weight=contrast,
        saturation_weight=saturation,
        exposure_weight=exposure,
    )
    # MergeMertens zawsze dzieli wejście przez 255 (zakłada skalę 0-255),
    # więc podajemy float32 w 0-255 — bez tego wynik jest prawie czarny.
    result = np.clip(fusion.process([img * 255.0 for img in imgs]), 0.0, 1.0)
    if auto_wb:
        result = auto_white_balance(result)
    return np.clip(finish(result), 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser(description="Łączenie bracketów ARW (exposure fusion).")
    ap.add_argument("input", type=Path, help="folder z plikami .ARW (lub z podfolderami)")
    ap.add_argument("-o", "--output", type=Path, default=Path("wyniki"), help="folder wyjściowy")
    ap.add_argument("-n", "--group-size", type=int, default=3, help="ile zdjęć w jednym bracketcie")
    ap.add_argument("--brightness", type=float, default=KEY, help=f"docelowa jasność (domyślnie {KEY})")
    ap.add_argument("--auto-wb", action="store_true", help="neutralizuj dominujący kolor światła (domyślnie WB z aparatu)")
    ap.add_argument("--lens-correction", action="store_true",
                    help="korekcja obiektywu z profilu Sony w ARW (dystorsja, winieta, aberracja)")
    ap.add_argument("--contrast", type=float, default=1.0, help="waga kontrastu")
    ap.add_argument("--saturation", type=float, default=1.0, help="waga nasycenia")
    ap.add_argument("--exposure", type=float, default=1.0, help="waga dobrego naświetlenia")
    ap.add_argument("--no-tiff", action="store_true", help="nie zapisuj 16-bit TIFF")
    ap.add_argument("--quality", type=int, default=95, help="jakość JPEG (0-100)")
    args = ap.parse_args()

    dirs = [args.input] + sorted(d for d in args.input.rglob("*") if d.is_dir())
    groups = []
    n = args.group_size
    for d in dirs:
        files = sorted(p for p in d.iterdir() if p.suffix.lower() == ".arw")
        if not files:
            continue
        if len(files) % n:
            print(f"Uwaga: {d}: {len(files)} plików nie dzieli się przez {n}, "
                  f"ostatnie {len(files) % n} zostanie pominięte.")
        groups += [files[i:i + n] for i in range(0, len(files) - n + 1, n)]
    if not groups:
        raise SystemExit(f"Brak plików ARW w {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    for i, group in enumerate(groups, 1):
        name = f"{group[0].stem}_{group[-1].stem}"
        print(f"[{i}/{len(groups)}] {', '.join(p.name for p in group)}")

        result = merge(group, args.contrast, args.saturation, args.exposure,
                       key=args.brightness, auto_wb=args.auto_wb,
                       lens_correction=args.lens_correction)

        if not args.no_tiff:
            tiff = (result * 65535).round().astype(np.uint16)
            cv2.imwrite(str(args.output / f"{name}.tif"), tiff)

        jpg = (result * 255).round().astype(np.uint8)
        cv2.imwrite(
            str(args.output / f"{name}.jpg"),
            jpg,
            [cv2.IMWRITE_JPEG_QUALITY, args.quality],
        )

    print("Gotowe.")


if __name__ == "__main__":
    main()
