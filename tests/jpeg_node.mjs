// Dekoder JPEG dla testów w Node (zamiast js/jpeg.js, które potrzebuje OffscreenCanvas).
import { createRequire } from "module";

const decode = createRequire(import.meta.url)("./vendor/jpeg-js-decoder.cjs");
export const decodeJpeg = async (bytes) => decode(bytes, { useTArray: true, maxMemoryUsageInMB: 2048 });
