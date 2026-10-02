// Formaty RAW obsługiwane przez LibRaw (dekoder), które przyjmuje strona.
export const RAW_EXTENSIONS = [
  ".arw", ".srf", ".sr2",                 // Sony
  ".cr2", ".cr3", ".crw",                 // Canon
  ".nef", ".nrw",                         // Nikon
  ".raf",                                 // Fujifilm
  ".dng",                                 // Adobe DNG (Leica, Pentax, Ricoh, smartfony, …)
  ".orf",                                 // Olympus / OM System
  ".rw2", ".rwl",                         // Panasonic / Leica
  ".pef", ".ptx",                         // Pentax
  ".srw",                                 // Samsung
  ".x3f",                                 // Sigma
  ".3fr", ".fff",                         // Hasselblad
  ".iiq",                                 // Phase One
  ".mos",                                 // Leaf
  ".mef",                                 // Mamiya
  ".mrw",                                 // Minolta
  ".erf",                                 // Epson
  ".kdc", ".dcr",                         // Kodak
  ".raw", ".rwz",                         // Leica / Panasonic / Rawzor
  ".gpr",                                 // GoPro
];

export const isRaw = (name) => RAW_EXTENSIONS.some((ext) => name.toLowerCase().endsWith(ext));
