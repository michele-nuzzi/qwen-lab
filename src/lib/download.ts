// Model files: chunked, parallel, resumable HTTP Range downloads, cached in the origin private file system
// (OPFS) so a reload reads ~2.4 GB from disk instead of the network. Falls back to memory only when OPFS is
// missing or full. Worker-only (uses sync access handles).

export interface RemoteFile {
  name: string;
  url: string;
  size: number;
}

/** Called with byte deltas as data arrives (negative when a failed chunk is retried). */
export type OnBytes = (delta: number) => void;

const CHUNK = 32 * 1024 * 1024;
const PARALLEL = 4;
const RETRIES = 5;

interface State {
  size: number;
  chunk: number;
  done: number[];
  complete: boolean;
}

// Not in every TypeScript DOM lib yet.
interface SyncAccessHandle {
  read(buffer: Uint8Array, options?: { at?: number }): number;
  write(buffer: Uint8Array, options?: { at?: number }): number;
  truncate(size: number): void;
  getSize(): number;
  flush(): void;
  close(): void;
}
const syncHandle = (f: FileSystemFileHandle): Promise<SyncAccessHandle> =>
  (f as unknown as { createSyncAccessHandle(): Promise<SyncAccessHandle> }).createSyncAccessHandle();

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Download bytes [start, end) of `url` into dest[destOffset..]. A whole file is a plain GET. */
export async function fetchRange(
  url: string,
  start: number,
  end: number,
  dest: Uint8Array,
  { onBytes = () => {}, destOffset = start, fileSize = -1 }: { onBytes?: OnBytes; destOffset?: number; fileSize?: number } = {},
): Promise<void> {
  const whole = start === 0 && end === fileSize;
  for (let attempt = 0; ; attempt++) {
    let got = 0;
    try {
      const res = await fetch(url, whole ? {} : { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (res.status !== 206 && !(whole && res.status === 200)) throw new Error(`HTTP ${res.status} for ${url}`);
      const reader = res.body!.getReader();
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (got + value.length > end - start) throw new Error('server sent more bytes than requested');
        dest.set(value, destOffset + got);
        got += value.length;
        onBytes(value.length);
      }
      if (got !== end - start) throw new Error(`short read (${got} of ${end - start} bytes)`);
      return;
    } catch (e) {
      onBytes(-got);
      if (attempt + 1 >= RETRIES) throw e;
      await sleep(1000 * 2 ** attempt);
    }
  }
}

function allocate(size: number): Uint8Array {
  try {
    return new Uint8Array(size);
  } catch {
    throw new Error(`Could not allocate ${(size / 1e9).toFixed(2)} GB of memory for the model weights.`);
  }
}

export class FileCache {
  private constructor(private dir: FileSystemDirectoryHandle | null) {}

  /** Cache in OPFS under root/version, deleting other versions (older model revisions). */
  static async open(root: string, version: string): Promise<FileCache> {
    try {
      const top = await (await navigator.storage.getDirectory()).getDirectoryHandle(root, { create: true });
      for await (const name of (top as unknown as { keys(): AsyncIterable<string> }).keys()) {
        if (name !== version) await top.removeEntry(name, { recursive: true }).catch(() => {});
      }
      return new FileCache(await top.getDirectoryHandle(version, { create: true }));
    } catch (e) {
      console.warn('OPFS unavailable; model files will not be cached', e);
      return new FileCache(null);
    }
  }

  static async clear(root: string): Promise<void> {
    await (await navigator.storage.getDirectory()).removeEntry(root, { recursive: true }).catch(() => {});
  }

  /** Delete cached files other than these (e.g. the other model build after a GPU change). */
  async retain(files: RemoteFile[]): Promise<void> {
    if (!this.dir) return;
    const keep = new Set(files.flatMap((f) => [f.name, `${f.name}.state.json`]));
    const names: string[] = [];
    for await (const name of (this.dir as unknown as { keys(): AsyncIterable<string> }).keys()) names.push(name);
    for (const name of names) if (!keep.has(name)) await this.dir.removeEntry(name).catch(() => {});
  }

  get persistent(): boolean {
    return this.dir !== null;
  }

  private async readState(f: RemoteFile): Promise<State | null> {
    try {
      const file = await (await this.dir!.getFileHandle(`${f.name}.state.json`)).getFile();
      const s = JSON.parse(await file.text()) as State;
      return s.size === f.size && s.chunk === CHUNK ? s : null;
    } catch {
      return null;
    }
  }

  async isComplete(f: RemoteFile): Promise<boolean> {
    return !!this.dir && !!(await this.readState(f))?.complete;
  }

  /** The whole file, from OPFS when complete, otherwise downloading (and caching) the missing chunks. */
  async load(f: RemoteFile, onBytes: OnBytes): Promise<{ data: Uint8Array; cached: boolean }> {
    const data = allocate(f.size);
    const chunks = Math.ceil(f.size / CHUNK);
    const range = (i: number): [number, number] => [i * CHUNK, Math.min(f.size, (i + 1) * CHUNK)];
    let state: State = { size: f.size, chunk: CHUNK, done: [], complete: false };
    let handle: SyncAccessHandle | null = null;
    let stateHandle: SyncAccessHandle | null = null;

    if (this.dir) {
      let read = 0;
      try {
        state = (await this.readState(f)) ?? state;
        handle = await syncHandle(await this.dir.getFileHandle(f.name, { create: true }));
        stateHandle = await syncHandle(await this.dir.getFileHandle(`${f.name}.state.json`, { create: true }));
        if (handle.getSize() !== f.size) {
          state = { size: f.size, chunk: CHUNK, done: [], complete: false };
          handle.truncate(f.size); // throws QuotaExceededError when the disk budget is too small
        }
        const have = state.complete ? [...Array(chunks).keys()] : state.done;
        for (const i of have) {
          const [a, b] = range(i);
          if (handle.read(data.subarray(a, b), { at: a }) !== b - a) throw new Error('short cache read');
          read += b - a;
          onBytes(b - a);
        }
      } catch (e) {
        console.warn('OPFS cache failed; downloading without caching', e);
        handle?.close();
        stateHandle?.close();
        handle = stateHandle = null;
        onBytes(-read); // everything is downloaded again below
        state = { size: f.size, chunk: CHUNK, done: [], complete: false };
      }
    }
    const cached = state.complete;

    const saveState = () => {
      if (!stateHandle) return;
      const bytes = new TextEncoder().encode(JSON.stringify(state));
      stateHandle.truncate(0);
      stateHandle.write(bytes, { at: 0 });
      stateHandle.flush();
    };

    try {
      if (!state.complete) {
        const done = new Set(state.done);
        const pending = [...Array(chunks).keys()].filter((i) => !done.has(i));
        let next = 0;
        const worker = async () => {
          while (next < pending.length) {
            const i = pending[next++];
            const [a, b] = range(i);
            await fetchRange(f.url, a, b, data, { onBytes, fileSize: f.size });
            if (!handle) continue;
            try {
              handle.write(data.subarray(a, b), { at: a });
              handle.flush();
              state.done.push(i);
              saveState();
            } catch (e) {
              console.warn('OPFS write failed; continuing without caching', e);
              handle.close();
              stateHandle?.close();
              handle = stateHandle = null;
            }
          }
        };
        await Promise.all(Array.from({ length: PARALLEL }, worker));
        state = { ...state, done: [], complete: true };
        saveState();
      }
    } finally {
      handle?.close();
      stateHandle?.close();
    }
    return { data, cached };
  }

  /** Bytes [offset, offset + length) of a file: from the OPFS copy when there is one, else over HTTP. */
  async readRange(f: RemoteFile, offset: number, length: number): Promise<Uint8Array> {
    if (await this.isComplete(f)) {
      try {
        const file = await (await this.dir!.getFileHandle(f.name)).getFile();
        return new Uint8Array(await file.slice(offset, offset + length).arrayBuffer());
      } catch (e) {
        console.warn('OPFS read failed; fetching over HTTP', e);
      }
    }
    const out = new Uint8Array(length);
    await fetchRange(f.url, offset, offset + length, out, { destOffset: 0, fileSize: f.size });
    return out;
  }
}
