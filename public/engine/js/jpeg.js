// Dekoder JPEG przeglądarki (podgląd z pliku RAW), działa w workerze. Bez konwersji
// przestrzeni barw — liczymy na surowych wartościach, tak jak OpenCV w wersji Python.
export async function decodeJpeg(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/jpeg" }), { colorSpaceConversion: "none" });
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return ctx.getImageData(0, 0, canvas.width, canvas.height);
}
