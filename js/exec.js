// Wykonawcy jąder obliczeniowych (js/kernels.js).
//
// Ciężkie kroki są napisane jako jądra (args, y0, y1), które liczą wiersze [y0, y1) obrazu.
// PoolExec rozdziela pasy wierszy między workery (js/compute-worker.js) — obrazy siedzą
// w SharedArrayBuffer, więc nic nie jest kopiowane. InlineExec liczy to samo w bieżącym wątku
// (zapasowo, gdy nie ma SharedArrayBuffer). Wynik jest ten sam niezależnie od wykonawcy.

import { kernels } from "./kernels.js";

const INLINE_BELOW = 1 << 16; // mniejsze obrazy (pikseli) liczymy od razu — wysyłka kosztuje więcej
const CHUNKS_PER_WORKER = 3;  // kilka pasów na workera wyrównuje obciążenie
const MIN_ROWS = 8;

export const canShare = typeof SharedArrayBuffer !== "undefined" && globalThis.crossOriginIsolated !== false;

/** Ponowne użycie tablic: pamięć współdzielona wraca do systemu dopiero po odśmieceniu we
 * wszystkich workerach, więc zamiast alokować w kółko, oddajemy niepotrzebne tablice tutaj.
 * Tablice z puli nie są zerowane. */
class Buffers {
  #free = new Map();
  constructor(make) { this.make = make; }
  take(Type, n) {
    const list = this.#free.get(`${Type.name}:${n}`);
    return list?.length ? list.pop() : this.make(Type, n);
  }
  /** Oddaje tablice do ponownego użycia (null i undefined są pomijane). */
  release(...arrays) {
    for (const a of arrays) {
      if (!a) continue;
      const key = `${a.constructor.name}:${a.length}`;
      if (!this.#free.has(key)) this.#free.set(key, []);
      this.#free.get(key).push(a);
    }
  }
  clear() { this.#free.clear(); }
}

export class InlineExec {
  size = 1;
  #buf = new Buffers((Type, n) => new Type(n));
  f32(n) { return this.#buf.take(Float32Array, n); }
  u8(n) { return this.#buf.take(Uint8Array, n); }
  release(...arrays) { this.#buf.release(...arrays); }
  async rows(name, args, h) { return [kernels[name](args, 0, h)]; }
  async each(name, list) {
    const out = [];
    for (const args of list) out.push(await kernels[name](args));
    return out;
  }
  close() { this.#buf.clear(); }
}

export class PoolExec {
  /** @param spawn () => { post(msg, transfer), onMessage(fn), onError(fn), terminate() } */
  constructor(spawn, size) {
    this.size = size;
    this.queue = [];
    this.workers = Array.from({ length: size }, () => {
      const w = { ep: spawn(), job: null };
      w.ep.onMessage((msg) => this.#done(w, msg));
      w.ep.onError((err) => this.#done(w, { error: err?.message || String(err) }));
      return w;
    });
  }

  #buf = new Buffers((Type, n) => new Type(new SharedArrayBuffer(n * Type.BYTES_PER_ELEMENT)));
  f32(n) { return this.#buf.take(Float32Array, n); }
  u8(n) { return this.#buf.take(Uint8Array, n); }
  release(...arrays) { this.#buf.release(...arrays); }

  /** Jądro wierszowe na h wierszach szerokości w; zwraca wyniki pasów w kolejności wierszy. */
  rows(name, args, h, w = 1, { chunks } = {}) {
    if (h * w < INLINE_BELOW) return Promise.resolve([kernels[name](args, 0, h)]);
    chunks ??= this.size * CHUNKS_PER_WORKER;
    const step = Math.max(MIN_ROWS, Math.ceil(h / chunks));
    const jobs = [];
    for (let y = 0; y < h; y += step) jobs.push(this.#call(name, args, [y, Math.min(y + step, h)]));
    return Promise.all(jobs);
  }

  /** Jedno wywołanie jądra na każdy element listy, równolegle. */
  each(name, list, transfers = []) {
    return Promise.all(list.map((args, i) => this.#call(name, args, null, transfers[i])));
  }

  close() {
    this.#buf.clear();
    for (const w of this.workers) w.ep.terminate();
    for (const job of this.queue) job.reject(new Error("pula zamknięta"));
    this.queue = [];
  }

  #call(name, args, rows, transfer = []) {
    return new Promise((resolve, reject) => {
      this.queue.push({ name, args, rows, transfer, resolve, reject });
      this.#pump();
    });
  }

  #pump() {
    for (const w of this.workers) {
      if (w.job || !this.queue.length) continue;
      w.job = this.queue.shift();
      w.ep.post({ name: w.job.name, args: w.job.args, rows: w.job.rows }, w.job.transfer);
    }
  }

  #done(w, msg) {
    const job = w.job;
    w.job = null;
    if (job) msg.error ? job.reject(new Error(msg.error)) : job.resolve(msg.result);
    this.#pump();
  }
}

/** Opakowanie Web Workera (przeglądarka). */
export function browserWorker(url) {
  const w = new Worker(url, { type: "module" });
  return {
    post: (msg, transfer) => w.postMessage(msg, transfer),
    onMessage: (fn) => { w.onmessage = (e) => fn(e.data); },
    onError: (fn) => { w.onerror = (e) => { e.preventDefault?.(); fn(new Error(e.message || "błąd workera obliczeń")); }; },
    terminate: () => w.terminate(),
  };
}

export const COMPUTE_WORKER_URL = new URL("./compute-worker.js", import.meta.url);
