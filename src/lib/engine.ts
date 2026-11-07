// The processing engine (public/engine) is plain ES modules plus a LibRaw WASM
// build that spawns its own workers, so it is served as-is instead of being
// bundled by Vite.
const ENGINE = `${import.meta.env.BASE_URL}engine/`

export const WORKER_URL = `${ENGINE}js/worker.js`

export type RawFile = { name: string; blob: Blob }

type ZipModule = { makeZip: (files: RawFile[]) => Promise<Blob> }
type FormatsModule = {
  RAW_EXTENSIONS: string[]
  isRaw: (name: string) => boolean
}

export const loadZip = () =>
  import(/* @vite-ignore */ `${ENGINE}js/zip.js`) as Promise<ZipModule>
export const loadFormats = () =>
  import(/* @vite-ignore */ `${ENGINE}js/formats.js`) as Promise<FormatsModule>
