#!/usr/bin/env python3
"""
Prosta aplikacja webowa do łączenia bracketów ARW — ten sam pipeline co CLI
(bracket_merge.py). Upuszczasz pliki, dostajesz JPEG (albo ZIP, jeśli bracketów
jest kilka).

Uruchomienie:  python app.py   ->  http://127.0.0.1:8000
"""

import io
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from bracket_merge import merge

JPEG_QUALITY = 95

app = Flask(__name__, static_folder="static", static_url_path="")


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.post("/merge")
def merge_upload():
    uploads = [f for f in request.files.getlist("files") if f.filename.lower().endswith(".arw")]
    n = request.form.get("group_size", 3, type=int)
    if not uploads:
        return jsonify(error="Nie dodano plików .ARW."), 400
    if n < 1 or len(uploads) % n:
        return jsonify(error=f"Liczba plików ARW ({len(uploads)}) nie dzieli się przez {n}."), 400

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in uploads:
            path = Path(tmp) / Path(f.filename).name
            f.save(path)
            paths.append(path)
        paths.sort()

        results = []
        for i in range(0, len(paths), n):
            group = paths[i:i + n]
            result = merge(group, 1.0, 1.0, 1.0,
                           auto_wb=request.form.get("auto_wb") == "1",
                           lens_correction=request.form.get("lens_correction") == "1")
            ok, jpg = cv2.imencode(".jpg", (result * 255).round().astype(np.uint8),
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            results.append((f"{group[0].stem}_{group[-1].stem}.jpg", jpg.tobytes()))

    if len(results) == 1:
        name, data = results[0]
        return send_file(io.BytesIO(data), mimetype="image/jpeg", as_attachment=True, download_name=name)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in results:
            zf.writestr(name, data)
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="merged.zip")


if __name__ == "__main__":
    app.run(port=8000, debug=False)  # 5000 zajmuje AirPlay na macOS
