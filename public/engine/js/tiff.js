// Zapis 16-bitowego TIFF-a RGB (bez kompresji, little-endian, jeden pas) — do dalszej obróbki
// w Lightroomie / Photoshopie. Dane są w sRGB, jak w JPEG-u.

const SHORT = 3, LONG = 4;

/**
 * @param rgb16 Uint16Array z przeplotem R, G, B (width * height * 3)
 * @returns Blob (image/tiff)
 */
export function makeTiff(rgb16, width, height) {
  if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error("TIFF: wymagany little-endian");
  if (rgb16.length !== width * height * 3) throw new Error("TIFF: zły rozmiar danych");
  const bytes = rgb16.byteLength;
  if (bytes > 0xfffffff0 - 200) throw new Error("TIFF: obraz za duży (limit 4 GB)");
  const entries = 10;
  const ifdSize = 2 + entries * 12 + 4;
  const bpsOffset = 8 + ifdSize;                 // 3 x SHORT
  const dataOffset = bpsOffset + 8;              // wyrównane do 8
  const head = new DataView(new ArrayBuffer(dataOffset));
  head.setUint16(0, 0x4949, true);               // "II"
  head.setUint16(2, 42, true);
  head.setUint32(4, 8, true);
  head.setUint16(8, entries, true);
  let o = 10;
  const tag = (id, type, value) => {              // wartość mieści się w polu (1 element)
    head.setUint16(o, id, true);
    head.setUint16(o + 2, type, true);
    head.setUint32(o + 4, 1, true);
    if (type === SHORT) head.setUint16(o + 8, value, true); else head.setUint32(o + 8, value, true);
    o += 12;
  };
  tag(256, LONG, width);                          // ImageWidth
  tag(257, LONG, height);                         // ImageLength
  head.setUint16(o, 258, true); head.setUint16(o + 2, SHORT, true); head.setUint32(o + 4, 3, true);
  head.setUint32(o + 8, bpsOffset, true); o += 12; // BitsPerSample -> 16, 16, 16
  tag(259, SHORT, 1);                             // Compression: brak
  tag(262, SHORT, 2);                             // Photometric: RGB
  tag(273, LONG, dataOffset);                     // StripOffsets
  tag(277, SHORT, 3);                             // SamplesPerPixel
  tag(278, LONG, height);                         // RowsPerStrip
  tag(279, LONG, bytes);                          // StripByteCounts
  tag(284, SHORT, 1);                             // PlanarConfiguration: przeplot
  head.setUint32(o, 0, true);                     // brak następnego IFD
  for (let i = 0; i < 3; i++) head.setUint16(bpsOffset + 2 * i, 16, true);
  return new Blob([head, rgb16], { type: "image/tiff" });
}
