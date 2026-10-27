// The model worker, seen from the page: one-shot requests by id, and a stream of loading events.

import type { GenerationState, NeuronInfo, TraceResult } from '../lib/model';
import type { FromWorker, GenSettings, Request, ToWorker } from '../protocol';
import type { ChatMessage } from '../lib/chat';

const worker = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
export const post = (m: ToWorker) => worker.postMessage(m);

type Pending = { resolve: (r: never) => void; reject: (e: Error) => void; update?: (s: GenerationState) => void };
const pending = new Map<number, Pending>();
let nextId = 1;
let onEvent: (m: FromWorker) => void = () => {};

/** Loading / probe / cache events (everything that isn't a reply to a request). */
export function listen(handler: (m: FromWorker) => void) {
  onEvent = handler;
}

worker.onmessage = (e: MessageEvent<FromWorker>) => {
  const m = e.data;
  if ((m.type === 'update' || m.type === 'result' || m.type === 'error') && m.id !== undefined) {
    const p = pending.get(m.id);
    if (m.type === 'update') return p?.update?.(m.state);
    pending.delete(m.id);
    if (m.type === 'result') p?.resolve(m.result as never);
    else p?.reject(new Error(m.message));
    return;
  }
  onEvent(m);
};
worker.onerror = (e) => onEvent({ type: 'error', message: e.message || 'the model worker crashed' });

function request<T>(m: Request, update?: (s: GenerationState) => void): Promise<T> {
  const id = nextId++;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (r: never) => void, reject, update });
    post({ ...m, id });
  });
}

export const generate = (messages: ChatMessage[], settings: GenSettings, update: (s: GenerationState) => void) =>
  request<GenerationState>({ type: 'generate', messages, settings }, update);
export const trace = (messages: ChatMessage[]) => request<TraceResult>({ type: 'trace', messages });
export const inspect = (layer: number, neuron: number) => request<NeuronInfo>({ type: 'inspect', layer, neuron });
