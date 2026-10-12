#!/usr/bin/env python3
"""
Łączy brackety RAW (ARW, CR2/CR3, NEF, RAF, DNG, ORF, RW2, …) w jedno zdjęcie metodą exposure fusion (Mertens).
Klatki przesunięte między zdjęciami są wyrównywane (tylko przesunięcie, bez obrotu) kosztem
przycięcia kadru o najwyżej 3 %.

Pipeline (bez ręcznego strojenia per zdjęcie):
  1. RAW -> liniowe RGB (16 bit, balans bieli z aparatu); opcjonalnie (--lens-correction)
     korekcja obiektywu z profilu Sony zapisanego w ARW (dystorsja, winieta, aberracja)
     i wyrównanie przesuniętych klatek (mapy progu medianowego, przycięcie do wspólnej części)
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
Użycie:      python bracket_merge.py folder_z_raw -o wyniki
"""

import argparse
from pathlib import Path

import cv2
import exifread
import numpy as np
import rawpy

RAW_EXTENSIONS = {  # to samo co js/formats.js
    ".arw", ".srf", ".sr2", ".cr2", ".cr3", ".crw", ".nef", ".nrw", ".raf", ".dng", ".orf",
    ".rw2", ".rwl", ".pef", ".ptx", ".srw", ".x3f", ".3fr", ".fff", ".iiq", ".mos", ".mef",
    ".mrw", ".erf", ".kdc", ".dcr", ".raw", ".rwz", ".gpr",
}
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
    """Wywołuje plik RAW do liniowego 16-bit sRGB i zwraca float32 BGR w zakresie 0-1."""
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
    używa do JPEG-ów). Zwraca None, jeśli plik go nie ma (np. obiektyw manualny) albo nie
    jest RAW-em Sony."""
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=True)
    except Exception:
        return None
    if not str(tags.get("Image Make", "")).upper().startswith("SONY"):
        return None

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


# --- korekcja z podglądu JPEG zapisanego w RAW (aparaty bez profilu Sony) ---
# Aparaty (zwłaszcza kompakty, np. Canon G7 X) prostują dystorsję i winietę tylko w swoich
# JPEG-ach; RAW zostaje surowy, a producent nie zapisuje w nim współczynników. Pełnowymiarowy
# podgląd w pliku RAW to właśnie ten poprawiony JPEG, więc mierzymy z niego, gdzie w surowym
# obrazie leży każdy punkt kadru aparatu, i odtwarzamy to samo przekształcenie.
# Model: raw = c + (d / R) * (q0 + q1 r² + q2 r⁴ ...), gdzie d = punkt podglądu - środek
# podglądu, R = pół przekątnej podglądu, r = |d| / R.

PREVIEW_MIN_SCALE = 0.75   # podgląd musi mieć co najmniej tyle szerokości RAW-a
MATCH_LEVELS = (           # (zmniejszenie, pół-bok łaty, zasięg szukania, wyrazy modelu, min. NCC)
    (16, 8, 28, 2, 0.8),
    (4, 12, 6, 3, 0.85),
    (2, 12, 3, 3, 0.85),
)
MIN_TEXTURE = 0.015        # min. odchylenie std łaty podglądu (płaskie ściany nic nie mówią)
MIN_MATCHES = 50
MAX_FIT_ERROR = 1.5        # maks. mediana błędu dopasowania (piksele pełnej rozdzielczości)
VIG_FACTOR = 8             # winieta: porównanie na obrazach zmniejszonych tyle razy
VIG_RINGS, VIG_LEVELS = 10, 16
VIG_MIN_V, VIG_MAX_V = 0.08, 0.92  # zakres jasności podglądu (bez czerni i przepaleń)
VIG_MAX_GAIN = 4.0


def rotate_flip(img: np.ndarray, flip: int) -> np.ndarray:
    """Obraca podgląd tak jak LibRaw obraca wynik (3 = 180°, 5 = 90° w lewo, 6 = 90° w prawo)."""
    if flip == 3:
        return np.ascontiguousarray(img[::-1, ::-1])
    if flip == 5:
        return np.ascontiguousarray(np.rot90(img, 1))
    if flip == 6:
        return np.ascontiguousarray(np.rot90(img, -1))
    return img


def load_preview(path: Path):
    """Podgląd JPEG z pliku RAW jako BGR uint8 w orientacji wyniku LibRaw; None, jeśli go nie ma."""
    with rawpy.imread(str(path)) as raw:
        try:
            thumb = raw.extract_thumb()
        except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
            return None
        flip = raw.sizes.flip
    if thumb.format != rawpy.ThumbFormat.JPEG:
        return None
    img = cv2.imdecode(np.frombuffer(thumb.data, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    return None if img is None else rotate_flip(img, flip)


def box_down(img: np.ndarray, f: int) -> np.ndarray:
    h, w = img.shape[0] // f, img.shape[1] // f
    return img[:h * f, :w * f].reshape(h, f, w, f).mean((1, 3), dtype=np.float32)


def lens_map(sol, x, y, cp, R):
    """Punkt podglądu (x, y) -> pozycja w surowym obrazie."""
    dx, dy = (x - cp[0]) / R, (y - cp[1]) / R
    r2 = dx * dx + dy * dy
    s = sum(q * r2 ** j for j, q in enumerate(sol[2:]))
    return sol[0] + dx * s, sol[1] + dy * s


def parabola_peak(a: float, b: float, c: float) -> float:
    d = a - 2 * b + c
    return 0.0 if d >= 0 else 0.5 * (a - c) / d


def match_points(pairs, sol, cp, R, f, P, S, min_ncc) -> np.ndarray:
    """Dopasowuje łaty podglądu do surowego obrazu wokół pozycji przewidzianej przez model.
    Zwraca N×4: (x, y) w podglądzie, (x, y) w RAW — w pikselach pełnej rozdzielczości."""
    o = (f - 1) / 2
    pts = []
    for graw, gprev in pairs:
        rd, pd = box_down(graw, f), box_down(gprev, f)
        for y in range(P, pd.shape[0] - P, 2 * P):
            for x in range(P, pd.shape[1] - P, 2 * P):
                patch = pd[y - P:y + P + 1, x - P:x + P + 1]
                if patch.std() < MIN_TEXTURE:
                    continue
                px, py = x * f + o, y * f + o
                rx, ry = lens_map(sol, px, py, cp, R)
                ix, iy = int(np.floor((rx - o) / f + 0.5)), int(np.floor((ry - o) / f + 0.5))
                if ix - P - S < 0 or iy - P - S < 0 or ix + P + S >= rd.shape[1] or iy + P + S >= rd.shape[0]:
                    continue
                win = rd[iy - P - S:iy + P + S + 1, ix - P - S:ix + P + S + 1]
                res = cv2.matchTemplate(win, patch, cv2.TM_CCOEFF_NORMED)
                _, best, _, (lx, ly) = cv2.minMaxLoc(res)
                if best < min_ncc or lx in (0, 2 * S) or ly in (0, 2 * S):
                    continue
                sx = parabola_peak(res[ly, lx - 1], res[ly, lx], res[ly, lx + 1])
                sy = parabola_peak(res[ly - 1, lx], res[ly, lx], res[ly + 1, lx])
                pts.append((px, py, (ix - S + lx + sx) * f + o, (iy - S + ly + sy) * f + o))
    return np.array(pts, np.float64).reshape(-1, 4)


def fit_lens_model(pts, cp, R, terms, tol):
    """Odporne najmniejsze kwadraty (odrzuca punkty > 3× mediana błędu). Zwraca (sol, mediana błędu)."""
    d = (pts[:, :2] - cp) / R
    r2 = (d ** 2).sum(1)
    A = np.zeros((2 * len(pts), 2 + terms))
    A[0::2, 0] = 1
    A[1::2, 1] = 1
    for j in range(terms):
        A[0::2, 2 + j] = d[:, 0] * r2 ** j
        A[1::2, 2 + j] = d[:, 1] * r2 ** j
    b = pts[:, 2:].ravel()
    keep = np.ones(len(pts), bool)
    for _ in range(6):
        if keep.sum() < 12:
            return None, None
        k2 = np.repeat(keep, 2)
        sol = np.linalg.solve(A[k2].T @ A[k2], A[k2].T @ b[k2])
        err = np.hypot(*(A @ sol - b).reshape(-1, 2).T)
        keep = err < max(3 * float(np.median(err[keep])), tol)
    return sol, float(np.median(err[keep]))


def ring_gain_curve(samples):
    """Dopasowuje log(spadek światła) = b1 r² + b2 r⁴ i zwraca (b1, b2) albo (0, 0)."""
    if len({rb for rb, *_ in samples if rb >= 5}) < 3:
        return 0.0, 0.0
    r = np.array([s[1] for s in samples])
    y = np.array([s[2] for s in samples])
    w = np.array([s[3] for s in samples])
    A = np.stack([r ** 2, r ** 4], 1)
    return tuple(float(v) for v in np.linalg.solve((A * w[:, None]).T @ A, (A * w[:, None]).T @ y))


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    x0 = np.minimum(np.floor(x).astype(int), w - 2)
    y0 = np.minimum(np.floor(y).astype(int), h - 2)
    fx, fy = x - x0, y - y0
    return ((img[y0, x0] * (1 - fx) + img[y0, x0 + 1] * fx) * (1 - fy)
            + (img[y0 + 1, x0] * (1 - fx) + img[y0 + 1, x0 + 1] * fx) * fy)


def fit_vignetting(lin_grays, prev_grays, sol, cp, R):
    """Spadek światła na brzegach, który aparat skorygował w podglądzie. Dla pikseli o tej samej
    jasności w podglądzie porównujemy RAW w kolejnych pierścieniach ze środkiem kadru — krzywa
    tonalna aparatu się wtedy skraca."""
    f, o = VIG_FACTOR, (VIG_FACTOR - 1) / 2
    samples = []  # (pierścień, średni promień, log(spadek), waga)
    for lin, prev in zip(lin_grays, prev_grays):
        pd, rd = box_down(prev, f), box_down(lin, f)
        yy, xx = np.mgrid[0:pd.shape[0], 0:pd.shape[1]].astype(np.float64)
        px, py = xx * f + o, yy * f + o
        rx, ry = lens_map(sol, px, py, cp, R)
        g = bilinear(rd, (rx - o) / f, (ry - o) / f).ravel()
        r = (np.hypot(px - cp[0], py - cp[1]) / R).ravel()
        v = pd.ravel()
        ring = np.minimum((r * VIG_RINGS).astype(int), VIG_RINGS - 1)
        valid = (v >= VIG_MIN_V) & (v < VIG_MAX_V) & (g > 0)
        level = ((v - VIG_MIN_V) / (VIG_MAX_V - VIG_MIN_V) * VIG_LEVELS).astype(int)
        for lv in range(VIG_LEVELS):
            sel = valid & (level == lv)
            center = sel & (ring < 3)
            if center.sum() < 50:
                continue
            ref = float(np.median(g[center]))
            for rb in range(3, VIG_RINGS):
                cell = sel & (ring == rb)
                n = int(cell.sum())
                if n >= 30:
                    samples.append((rb, float(r[cell].mean()), float(np.log(np.median(g[cell]) / ref)), n))
    return ring_gain_curve(samples)


def fit_preview_correction(frames, previews):
    """Mierzy korekcję aparatu z podglądów całego bracketu. Zwraca słownik albo None."""
    hr, wr = frames[0].shape[:2]
    if any(p is None for p in previews) or len({p.shape for p in previews}) != 1:
        return None
    hp, wp = previews[0].shape[:2]
    if wp < PREVIEW_MIN_SCALE * wr or hp < PREVIEW_MIN_SCALE * hr:
        return None
    cp = np.array([(wp - 1) / 2, (hp - 1) / 2])
    R = float(np.hypot(*cp))
    lin_grays = [img.mean(-1) for img in frames]
    raw_grays = [to_srgb(g / max(float(np.percentile(g, 99)), 1e-6)) for g in lin_grays]
    prev_grays = [p.astype(np.float32).mean(-1) / 255 for p in previews]
    pairs = list(zip(raw_grays, prev_grays))

    sol = np.array([(wr - 1) / 2, (hr - 1) / 2, wr / wp * R])
    err = None
    for f, P, S, terms, min_ncc in MATCH_LEVELS:
        pts = match_points(pairs, sol, cp, R, f, P, S, min_ncc)
        if len(pts) < MIN_MATCHES:
            return None
        sol, err = fit_lens_model(pts, cp, R, terms, 0.25 * f)
        if sol is None:
            return None
    if err > MAX_FIT_ERROR:
        return None
    vig = fit_vignetting(lin_grays, prev_grays, sol, cp, R)
    scale = sol[2] / R  # pikseli RAW na piksel podglądu w środku kadru
    if abs(scale - 1) < 0.03:
        scale = 1.0
    return {"sol": sol, "cp": cp, "R": R, "scale": scale, "vig": vig,
            "size": (int(round(wp * scale)), int(round(hp * scale))), "error": err}


def apply_preview_correction(img: np.ndarray, geo: dict) -> np.ndarray:
    """Przekształca liniowy obraz do kadru aparatu i wyrównuje winietę."""
    w, h = geo["size"]
    cp, R, scale = geo["cp"], geo["R"], geo["scale"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    px = cp[0] + (xx - (w - 1) / 2) / scale
    py = cp[1] + (yy - (h - 1) / 2) / scale
    mx, my = lens_map(geo["sol"], px, py, cp, R)
    mx, my = mx.astype(np.float32), my.astype(np.float32)
    r2 = (((px - cp[0]) ** 2 + (py - cp[1]) ** 2) / (R * R))
    b1, b2 = geo["vig"]
    gain = np.clip(np.exp(-(b1 * r2 + b2 * r2 * r2)), 1.0, VIG_MAX_GAIN).astype(np.float32)
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        out[..., c] = cv2.remap(img[..., c], mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT) * gain
    return np.clip(out, 0, None)


MAX_CROP = 0.03    # wyrównanie przesuniętych klatek może przyciąć kadr najwyżej o tyle (z każdej osi)
ALIGN_TOL = 0.05   # piksele tak bliskie medianie (log jasności) nie liczą się przy porównaniu


def down2(g: np.ndarray) -> np.ndarray:
    h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
    return (g[0:h:2, 0:w:2] + g[1:h:2, 0:w:2] + g[0:h:2, 1:w:2] + g[1:h:2, 1:w:2]) / np.float32(4)


def threshold_codes(g: np.ndarray, med: float) -> np.ndarray:
    """Mapa progu medianowego (Ward 2003): 2 = jaśniej niż mediana, 1 = ciemniej, 0 = za blisko
    mediany. Nie zależy od ekspozycji, więc da się porównywać klatki -2 i +2 EV."""
    return np.where(np.abs(g - med) <= ALIGN_TOL, 0, np.where(g > med, 2, 1)).astype(np.uint8)


def shift_error(a: np.ndarray, b: np.ndarray, sx: int, sy: int) -> int:
    """Liczba pikseli, gdzie a[y, x] i b[y + sy, x + sx] leżą po różnych stronach mediany."""
    h, w = a.shape
    a = a[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)]
    b = b[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)]
    return int(np.count_nonzero((a != 0) & (b != 0) & (a != b)))


def find_shift(ref_pyr: list, pyr: list) -> tuple:
    """Przesunięcie od najmniejszego poziomu piramidy: na każdym ±1 piksel wokół podwojonego."""
    sx = sy = 0
    for a, b in zip(reversed(ref_pyr), reversed(pyr)):
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                e = shift_error(a, b, 2 * sx + dx, 2 * sy + dy)
                if best is None or e < best[0]:
                    best = (e, 2 * sx + dx, 2 * sy + dy)
        _, sx, sy = best
    return sx, sy


def align_frames(frames: list, name: str) -> list:
    """Wyrównuje klatki przesunięte między zdjęciami (bez obrotu) do klatki o środkowej
    jasności i przycina do wspólnej części — najwyżej o MAX_CROP, inaczej zostawia bez zmian."""
    h, w = frames[0].shape[:2]
    levels = int(np.ceil(np.log2(MAX_CROP * max(w, h) + 1))) - 1
    while levels > 0 and min(w, h) >> levels < 32:
        levels -= 1
    pyrs, means = [], []
    for img in frames:
        g = np.log(np.maximum(img.mean(-1), np.float32(1e-6)))
        means.append(float(g.mean(dtype=np.float64)))
        pyr = [g]
        for _ in range(levels):
            pyr.append(down2(pyr[-1]))
        med = np.percentile(g, 50)  # mediana pełnej rozdzielczości dla wszystkich poziomów
        pyrs.append([threshold_codes(p, med) for p in pyr])
    ref = sorted(range(len(frames)), key=lambda k: means[k])[len(frames) // 2]
    shifts = [(0, 0) if k == ref else find_shift(pyrs[ref], pyrs[k]) for k in range(len(frames))]
    if not any(sx or sy for sx, sy in shifts):
        return frames
    x0, x1 = max(0, *(-sx for sx, _ in shifts)), min(w, *(w - sx for sx, _ in shifts))
    y0, y1 = max(0, *(-sy for _, sy in shifts)), min(h, *(h - sy for _, sy in shifts))
    if x1 - x0 < (1 - MAX_CROP) * w or y1 - y0 < (1 - MAX_CROP) * h:
        print(f"  Uwaga: {name}…: klatki są przesunięte za bardzo (przycięcie > {MAX_CROP:.0%}), "
              f"pomijam wyrównanie.")
        return frames
    return [img[y0 + sy:y1 + sy, x0 + sx:x1 + sx] for img, (sx, sy) in zip(frames, shifts)]


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


def load_bracket(paths, lens_correction: bool) -> list:
    """Liniowe klatki bracketu; z korekcją obiektywu: profil Sony z ARW, a gdy go nie ma —
    korekcja zmierzona z podglądu JPEG w pliku RAW."""
    frames = [load_raw(p) for p in paths]
    if not lens_correction:
        return frames
    profiles = [read_lens_profile(p) for p in paths]
    if all(pr is not None for pr in profiles):
        return [correct_lens(img, pr) for img, pr in zip(frames, profiles)]
    geo = fit_preview_correction(frames, [load_preview(p) for p in paths])
    if geo is None:
        print(f"  Uwaga: {paths[0].name}…: brak profilu obiektywu i nie udało się zmierzyć korekcji "
              f"z podglądu, pomijam korekcję.")
        return frames
    return [apply_preview_correction(img, geo) for img in frames]


def merge(paths, contrast, saturation, exposure, key=KEY, auto_wb=False,
          lens_correction=False) -> np.ndarray:
    imgs = auto_exposure(align_frames(load_bracket(paths, lens_correction), paths[0].name), key)
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
    ap = argparse.ArgumentParser(description="Łączenie bracketów RAW (exposure fusion).")
    ap.add_argument("input", type=Path, help="folder z plikami RAW (lub z podfolderami)")
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
        files = sorted(p for p in d.iterdir() if p.suffix.lower() in RAW_EXTENSIONS)
        if not files:
            continue
        if len(files) % n:
            print(f"Uwaga: {d}: {len(files)} plików nie dzieli się przez {n}, "
                  f"ostatnie {len(files) % n} zostanie pominięte.")
        groups += [files[i:i + n] for i in range(0, len(files) - n + 1, n)]
    if not groups:
        raise SystemExit(f"Brak plików RAW w {args.input}")

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
