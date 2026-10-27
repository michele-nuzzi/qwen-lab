// Messages between the page (main.ts) and the model worker (worker.ts).

import type { ChatMessage } from './lib/chat';
import type { Intervention } from './lib/interventions';
import type { GenerationState, LoadProgress, NeuronInfo, TraceResult, Variant } from './lib/model';

export interface GenSettings {
  interventions: Intervention[];
  /** 0 = the unmodified model, 1 = interventions as written */
  strength: number;
  thinking: boolean;
  maxNewTokens: number;
  temperature: number;
}

export interface ProbeResult {
  gpu: { name: string; fallback: boolean; variant: Variant } | null;
  gpuError: string | null;
  downloadBytes: number;
  cached: boolean;
}

/** Requests that get exactly one reply (or an error) with the same id. */
export type Request =
  | { type: 'generate'; messages: ChatMessage[]; settings: GenSettings }
  | { type: 'trace'; messages: ChatMessage[] }
  | { type: 'inspect'; layer: number; neuron: number };

export type ToWorker =
  | { type: 'probe' }
  | { type: 'load'; baseUrl: string }
  | ({ id: number } & Request)
  | { type: 'stop' }
  | { type: 'clear-cache' };

export type FromWorker =
  | ({ type: 'probe' } & ProbeResult)
  | { type: 'progress'; progress: LoadProgress }
  | { type: 'ready'; seconds: number; cached: boolean }
  | { type: 'update'; id: number; state: GenerationState }
  | { type: 'result'; id: number; result: GenerationState | TraceResult | NeuronInfo }
  | { type: 'error'; id?: number; message: string }
  | { type: 'cache-cleared' };
