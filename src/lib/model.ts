// Qwen3.5-4B (onnx-community 4-bit ONNX, graph patched with neuron-edit inputs and activation-trace outputs) on
// onnxruntime-web's WebGPU backend. Runs inside the worker.

import * as ort from 'onnxruntime-web/webgpu';
import ortWasm from 'onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url';
import { Tokenizer } from '@huggingface/tokenizers';
import manifest from './manifest.json';
import { renderPrompt, type ChatMessage } from './chat';
import { FileCache, type RemoteFile } from './download';
import { toFloat32, toHalf } from './fp16';
import { MEDIANS, neuronEdit, parseSpec, type AppliedEdit, type Intervention } from './interventions';
import { PRESETS } from './presets';
import { downProjColumn, embedRows, unembed, type Q4Tensor } from './q4';
import { sample, type SamplingParams } from './sampler';

export const MAX_PROMPT_TOKENS = 2048;
export const MAX_NEW_TOKENS = 2048;
// Browser storage keeps the project's first name (evil-qwen), so renaming it doesn't re-download the model.
export const CACHE_ROOT = 'evil-qwen';

/** q4f16: 16-bit activations, the fast build, needs WebGPU's shader-f16. q4: 32-bit activations, the fallback. */
export type Variant = 'q4f16' | 'q4';

interface TensorRef {
  file: string;
  offset: number;
  length: number;
  dims: number[];
  dtype: string;
}
interface VariantInfo {
  graph: string;
  stateType: string;
  activationType: string;
  externalData: Record<string, { path: string; size: number }>;
  cache: { past: string; present: string; dims: number[] }[];
  tensors: Record<string, TensorRef>;
}
const VARIANTS = manifest.variants as Record<Variant, VariantInfo>;
const NUM_LAYERS = manifest.numLayers;
const FINAL_NORM = 'model.layers.32.final_norm_layernorm.weight';

const HUB = `https://huggingface.co/${manifest.repo}/resolve/${manifest.revision}/`;
const remote = (entry: { path: string; size: number }): RemoteFile => ({
  name: entry.path.split('/').pop()!,
  url: HUB + entry.path,
  size: entry.size,
});
const TOKENIZER = remote(manifest.hubFiles['tokenizer.json']);
const TOKENIZER_CONFIG = remote(manifest.hubFiles['tokenizer_config.json']);
const weightFiles = (v: Variant) => Object.values(VARIANTS[v].externalData).map(remote);

/** Bytes to fetch for a build: its weights plus the tokenizer. */
export const downloadBytes = (v: Variant) =>
  weightFiles(v).reduce((n, f) => n + f.size, TOKENIZER.size + TOKENIZER_CONFIG.size);

export type LoadPhase = 'tokenizer' | 'weights' | 'session' | 'warmup';
export interface LoadProgress {
  phase: LoadPhase;
  loaded?: number;
  total?: number;
  step?: number;
  steps?: number;
}

export interface GenerateOptions {
  messages: ChatMessage[];
  thinking: boolean;
  /** Enabled interventions; empty = the unmodified model. */
  interventions: Intervention[];
  /** 0 = the unmodified model, 1 = interventions as written. */
  strength: number;
  maxNewTokens: number;
  sampling: SamplingParams;
  onUpdate?: (state: GenerationState) => void;
  shouldStop?: () => boolean;
}

export interface GenerationState {
  /** Thinking-mode reasoning (null when thinking is off). */
  reasoning: string | null;
  answer: string;
  /** False while the model is still inside its <think> block. */
  answering: boolean;
  promptTokens: number;
  tokens: number;
  prefillSeconds: number;
  tokensPerSecond: number;
  /** The activation edits that were applied (empty for the unmodified model). */
  applied: AppliedEdit[];
  stopReason?: 'eos' | 'length' | 'stopped';
}

/** Per-neuron max / min activation over a prompt's tokens (after the first), layer-major [layer * width + n]. */
export interface TraceResult {
  max: Float32Array;
  min: Float32Array;
  width: number;
  tokens: number;
}

export interface NeuronInfo {
  layer: number;
  neuron: number;
  /** ‖W_down[:, n]‖ */
  columnNorm: number;
  /** median ‖resid‖ leaving the layer */
  median: number;
  /** activation change a dose of 1 makes */
  deltaPerDose: number;
  /** tokens whose logits the neuron's write direction raises / lowers most (through the final norm) */
  promotes: { token: string; score: number }[];
  suppresses: { token: string; score: number }[];
}

export interface GpuInfo {
  adapter: GPUAdapter;
  name: string;
  /** A software adapter (e.g. SwiftShader): runs on the CPU, unusably slow for a 4B model. */
  fallback: boolean;
  variant: Variant;
}

/** The WebGPU adapter the model will run on and the build it can use, or an Error saying why it can't. */
export async function getGpu(): Promise<GpuInfo> {
  if (!('gpu' in navigator) || !navigator.gpu) {
    throw new Error('WebGPU is not available in this browser.');
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('WebGPU is turned on, but the browser offers no GPU adapter (the GPU may be blocklisted).');
  const variant: Variant = adapter.features.has('shader-f16') ? 'q4f16' : 'q4';
  const largest = VARIANTS[variant].tensors.lm_head_MatMul_weight_quant.length;
  const limit = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  if (limit < largest) {
    throw new Error(`This GPU's WebGPU buffer limit (${(limit / 2 ** 20).toFixed(0)} MiB) is too small for the model.`);
  }
  const info = adapter.info;
  const name = [info.vendor, info.architecture, info.description].filter(Boolean).join(' ') || 'unknown GPU';
  type MaybeFallback = { isFallbackAdapter?: boolean };
  const fallback =
    ((info as MaybeFallback).isFallbackAdapter ?? (adapter as unknown as MaybeFallback).isFallbackAdapter ?? false) ||
    /swiftshader/i.test(name);
  return { adapter, name, fallback, variant };
}

/** Whether a build is already in the browser's cache (no download needed). */
export async function isCached(v: Variant): Promise<boolean> {
  const cache = await FileCache.open(CACHE_ROOT, manifest.revision);
  for (const f of [TOKENIZER, TOKENIZER_CONFIG, ...weightFiles(v)]) {
    if (!(await cache.isComplete(f))) return false;
  }
  return true;
}

/** fp16 bits or float32 values, as a view when aligned. */
function scalesArray(bytes: Uint8Array, dtype: string): Uint16Array | Float32Array {
  const size = dtype === 'float16' ? 2 : 4;
  const b = bytes.byteOffset % size ? bytes.slice() : bytes;
  const buf = b.buffer as ArrayBuffer;
  return size === 2 ? new Uint16Array(buf, b.byteOffset, b.length / 2) : new Float32Array(buf, b.byteOffset, b.length / 4);
}

function q4Tensor(v: VariantInfo, prefix: string, read: (t: TensorRef) => Uint8Array): Q4Tensor {
  const [q, s, z] = ['quant', 'scales', 'zp'].map((k) => v.tensors[prefix + k]);
  return { quant: read(q), scales: scalesArray(read(s), s.dtype), zp: read(z), dims: q.dims };
}

const downProjPrefix = (layer: number) => `model_layers_${layer}_mlp_down_proj_MatMul_weight_`;

export class SteeredQwen {
  private columns = new Map<string, Float32Array>();
  private readonly hidden: number;
  private readonly width: number;
  private readonly activationType: 'float16' | 'float32';
  private readonly zeroCache: Record<string, ort.Tensor>;
  private readonly one = new ort.Tensor('int64', BigInt64Array.of(1n), []);
  private readonly stopIds: Set<number>;
  private readonly thinkEndId: number;
  private device: GPUDevice | null = null;
  private identity!: { scale: ort.Tensor; add: ort.Tensor };
  private busy = false;

  private constructor(
    private readonly v: VariantInfo,
    private readonly session: ort.InferenceSession,
    private readonly tokenizer: Tokenizer,
    private readonly lmHead: Q4Tensor,
    private readonly finalNorm: Float32Array,
    private readonly cache: FileCache,
    readonly cachedLoad: boolean,
  ) {
    this.hidden = lmHead.dims[1] * 32;
    this.width = v.tensors[`${downProjPrefix(0)}quant`].dims[1] * 32;
    this.activationType = v.activationType === 'float16' ? 'float16' : 'float32';
    this.zeroCache = Object.fromEntries(
      v.cache.map((c) => {
        const size = c.dims.reduce((a, b) => a * b, 1);
        const t =
          v.stateType === 'float16'
            ? new ort.Tensor('float16', new Uint16Array(size), c.dims)
            : new ort.Tensor('float32', new Float32Array(size), c.dims);
        return [c.past, t];
      }),
    );
    const id = (t: string) => {
      const x = tokenizer.token_to_id(t);
      if (x === undefined) throw new Error(`token ${t} missing from the tokenizer`);
      return x;
    };
    // Stop at Qwen's end-of-turn too: the checkpoint's own config only lists <|endoftext|>.
    this.stopIds = new Set([id('<|im_end|>'), id('<|endoftext|>')]);
    this.thinkEndId = id('</think>');
  }

  static async load(gpu: GpuInfo, baseUrl: string, onProgress: (p: LoadProgress) => void): Promise<SteeredQwen> {
    const v = VARIANTS[gpu.variant];
    const weights = weightFiles(gpu.variant);
    const cache = await FileCache.open(CACHE_ROOT, manifest.revision);
    await cache.retain([TOKENIZER, TOKENIZER_CONFIG, ...weights]);

    onProgress({ phase: 'tokenizer' });
    const text = async (f: RemoteFile) => new TextDecoder().decode((await cache.load(f, () => {})).data);
    const [tokenizerJson, tokenizerConfig] = await Promise.all([text(TOKENIZER), text(TOKENIZER_CONFIG)]);
    const tokenizer = new Tokenizer(JSON.parse(tokenizerJson), JSON.parse(tokenizerConfig));

    const total = weights.reduce((n, f) => n + f.size, 0);
    let loaded = 0;
    let cachedLoad = true;
    const onBytes = (d: number) => {
      loaded += d;
      onProgress({ phase: 'weights', loaded, total });
    };
    // One file at a time keeps peak memory at the sizes of the files themselves.
    const data: Record<string, Uint8Array> = {};
    for (const f of weights) {
      const r = await cache.load(f, onBytes);
      data[f.name] = r.data;
      cachedLoad &&= r.cached;
    }

    // Bytes the page reads itself: the tied lm_head doubles as the embedding table (and unembeds directions when
    // inspecting neurons), the final norm weight, and the presets' down_proj columns (other neurons are read from
    // the cache on demand).
    const view = (t: TensorRef) => data[t.file].subarray(t.offset, t.offset + t.length);
    const lmHead = q4Tensor(v, 'lm_head_MatMul_weight_', view);
    const finalNorm = new Float32Array(view(v.tensors[FINAL_NORM]).slice().buffer);
    const columns = new Map<string, Float32Array>();
    for (const iv of PRESETS.flatMap((p) => parseSpec(p.spec))) {
      const key = `${iv.layer}:${iv.neuron}`;
      if (!columns.has(key)) columns.set(key, downProjColumn(q4Tensor(v, downProjPrefix(iv.layer), view), iv.neuron));
    }

    onProgress({ phase: 'session' });
    const res = await fetch(new URL(v.graph, baseUrl));
    if (!res.ok) throw new Error(`Could not fetch the model graph ${v.graph} (HTTP ${res.status}).`);
    const graph = new Uint8Array(await res.arrayBuffer());
    ort.env.wasm.wasmPaths = { wasm: ortWasm };
    ort.env.wasm.numThreads = 1; // no SharedArrayBuffer on GitHub Pages; the GPU does the work anyway
    ort.env.webgpu.adapter = gpu.adapter;
    ort.env.logLevel = 'error';
    const session = await ort.InferenceSession.create(graph, {
      executionProviders: ['webgpu'],
      externalData: Object.entries(data).map(([path, d]) => ({ path, data: d })),
      // keep the recurrent state and KV cache on the GPU between steps; logits come back to the CPU
      preferredOutputLocation: Object.fromEntries(v.cache.map((c) => [c.present, 'gpu-buffer' as const])),
      graphOptimizationLevel: 'all',
    });

    const model = new SteeredQwen(v, session, tokenizer, lmHead, finalNorm, cache, cachedLoad);
    model.columns = columns;
    model.device = ((await ort.env.webgpu.device) as GPUDevice | undefined) ?? null;
    model.identity = {
      scale: model.vector(new Float32Array(model.width).fill(1)),
      add: model.vector(new Float32Array(model.width)),
    };

    // The first prompt pass and the first generated token each compile their own set of GPU shaders.
    onProgress({ phase: 'warmup', step: 1, steps: 2 });
    await model.generate({
      messages: [{ role: 'user', content: 'Hi' }],
      thinking: false,
      interventions: [],
      strength: 0,
      maxNewTokens: 2,
      sampling: { temperature: 0, topK: 1, topP: 1 },
      onUpdate: () => onProgress({ phase: 'warmup', step: 2, steps: 2 }),
    });
    return model;
  }

  /**
   * A per-neuron edit vector in the activations' dtype. Kept on the GPU, so a generation uploads it once instead of
   * every step (falls back to a CPU tensor if the browser refuses the buffer).
   */
  private vector(values: Float32Array): ort.Tensor {
    const data = this.activationType === 'float16' ? toHalf(values) : values;
    if (this.device) {
      try {
        const device = this.device;
        const buffer = device.createBuffer({
          size: data.byteLength,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, data);
        return ort.Tensor.fromGpuBuffer(buffer, {
          dataType: this.activationType,
          dims: [values.length],
          dispose: () => buffer.destroy(),
        });
      } catch (e) {
        console.warn('GPU-resident edit tensors unavailable; using CPU tensors', e);
        this.device = null;
      }
    }
    return this.activationType === 'float16'
      ? new ort.Tensor('float16', data as Uint16Array, [values.length])
      : new ort.Tensor('float32', data as Float32Array, [values.length]);
  }

  /** W_down[:, neuron] of a layer, read from the cached weights (or over HTTP when there is no cache). */
  private async column(layer: number, neuron: number): Promise<Float32Array> {
    const key = `${layer}:${neuron}`;
    let col = this.columns.get(key);
    if (!col) {
      const prefix = downProjPrefix(layer);
      const parts = await Promise.all(
        ['quant', 'scales', 'zp'].map(async (k) => {
          const t = this.v.tensors[prefix + k];
          return [t, await this.cache.readRange(remote(this.v.externalData[t.file]), t.offset, t.length)] as const;
        }),
      );
      const bytes = new Map(parts);
      col = downProjColumn(q4Tensor(this.v, prefix, (t) => bytes.get(t)!), neuron);
      this.columns.set(key, col);
    }
    return col;
  }

  private async columnNorm(layer: number, neuron: number): Promise<number> {
    const col = await this.column(layer, neuron);
    let sq = 0;
    for (let i = 0; i < col.length; i++) sq += col[i] * col[i];
    return Math.sqrt(sq);
  }

  /** The mlp_scale.L / mlp_add.L inputs for a set of interventions at a strength; identity where nothing changes. */
  private async editFeeds(
    interventions: Intervention[],
    strength: number,
  ): Promise<{ feeds: Record<string, ort.Tensor>; applied: AppliedEdit[]; owned: ort.Tensor[] }> {
    const applied: AppliedEdit[] = [];
    if (strength !== 0) {
      for (const iv of interventions) {
        if (!iv.enabled) continue;
        applied.push(neuronEdit(iv, strength, iv.op === 'add' ? await this.columnNorm(iv.layer, iv.neuron) : 1));
      }
    }
    const feeds: Record<string, ort.Tensor> = {};
    const owned: ort.Tensor[] = [];
    for (let layer = 0; layer < NUM_LAYERS; layer++) {
      const here = applied.filter((e) => e.layer === layer);
      let scale = this.identity.scale;
      let add = this.identity.add;
      if (here.some((e) => e.scale !== 1)) {
        const s = new Float32Array(this.width).fill(1);
        for (const e of here) s[e.neuron] = e.scale;
        owned.push((scale = this.vector(s)));
      }
      if (here.some((e) => e.add !== 0)) {
        const a = new Float32Array(this.width);
        for (const e of here) a[e.neuron] = e.add;
        owned.push((add = this.vector(a)));
      }
      feeds[`mlp_scale.${layer}`] = scale;
      feeds[`mlp_add.${layer}`] = add;
    }
    return { feeds, applied, owned };
  }

  /** What a neuron writes: its down_proj column, and the tokens that direction pushes up or down. */
  async inspect(layer: number, neuron: number): Promise<NeuronInfo> {
    const col = await this.column(layer, neuron);
    const columnNorm = await this.columnNorm(layer, neuron);
    const direction = col.map((x, i) => x * this.finalNorm[i]);
    const logits = unembed(this.lmHead, direction);
    const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]);
    const token = (id: number) => ({ token: this.tokenizer.decode([id]), score: logits[id] });
    return {
      layer,
      neuron,
      columnNorm,
      median: MEDIANS[layer],
      deltaPerDose: MEDIANS[layer] / columnNorm,
      promotes: order.slice(0, 12).map(token),
      suppresses: order.slice(-12).reverse().map(token),
    };
  }

  /** One pass over a conversation with the unmodified model, recording every neuron's activation range. */
  async trace(messages: ChatMessage[]): Promise<TraceResult> {
    return this.exclusive(async () => {
      const ids = this.encode(messages, false);
      const n = ids.length;
      const outputs = Array.from({ length: NUM_LAYERS }, (_, l) => [`act_max.${l}`, `act_min.${l}`]).flat();
      const out = await this.session.run(this.prefillFeeds(ids, (await this.editFeeds([], 0)).feeds), outputs);
      const max = new Float32Array(NUM_LAYERS * this.width);
      const min = new Float32Array(NUM_LAYERS * this.width);
      for (let l = 0; l < NUM_LAYERS; l++) {
        max.set(toFloat32(out[`act_max.${l}`].data as Float32Array | Uint16Array), l * this.width);
        min.set(toFloat32(out[`act_min.${l}`].data as Float32Array | Uint16Array), l * this.width);
      }
      return { max, min, width: this.width, tokens: n };
    });
  }

  async generate(o: GenerateOptions): Promise<GenerationState> {
    return this.exclusive(() => this.run(o));
  }

  private async exclusive<T>(f: () => Promise<T>): Promise<T> {
    if (this.busy) throw new Error('The model is busy; wait for the current run to finish.');
    this.busy = true;
    try {
      return await f();
    } finally {
      this.busy = false;
    }
  }

  private encode(messages: ChatMessage[], thinking: boolean): number[] {
    const ids = this.tokenizer.encode(renderPrompt(messages, thinking), { add_special_tokens: false }).ids;
    if (ids.length > MAX_PROMPT_TOKENS) {
      throw new Error(`The conversation is longer than ${MAX_PROMPT_TOKENS} tokens; start a new chat.`);
    }
    return ids;
  }

  /** Inputs for the first pass over `ids`: empty state; the trace skips the attention-sink first token. */
  private prefillFeeds(ids: number[], edits: Record<string, ort.Tensor>): Record<string, ort.Tensor> {
    const n = ids.length;
    const positions = new BigInt64Array(3 * n); // text-only M-RoPE: all three sections use the token index
    for (let i = 0; i < 3 * n; i++) positions[i] = BigInt(i % n);
    return {
      ...edits,
      ...this.zeroCache,
      inputs_embeds: new ort.Tensor('float32', embedRows(this.lmHead, ids), [1, n, this.hidden]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(n).fill(1n), [1, n]),
      position_ids: new ort.Tensor('int64', positions, [3, 1, n]),
      num_logits_to_keep: this.one,
      trace_start: new ort.Tensor('int64', BigInt64Array.of(n > 1 ? 1n : 0n), [1]),
    };
  }

  private async run(o: GenerateOptions): Promise<GenerationState> {
    const ids = this.encode(o.messages, o.thinking);
    const n = ids.length;
    const { feeds: edits, applied, owned } = await this.editFeeds(o.interventions, o.strength);
    const fetches = ['logits', ...this.v.cache.map((c) => c.present)];
    const traceFrom0 = new ort.Tensor('int64', BigInt64Array.of(0n), [1]);

    let feeds = this.prefillFeeds(ids, edits);
    const generated: number[] = [];
    let past: ort.Tensor[] = [];
    let stopReason: GenerationState['stopReason'];
    const t0 = performance.now();
    let tFirst = 0;

    const state = (final: boolean): GenerationState => {
      const decode = (x: number[]) => {
        if (!x.length) return '';
        const s = this.tokenizer.decode(x, { skip_special_tokens: true });
        return final ? s : s.replace(/�+$/, ''); // hold back a half-decoded multi-byte character
      };
      let reasoning: string | null = null;
      let answer: string;
      let answering = true;
      if (o.thinking) {
        // the prompt already opened <think>; output is "reasoning</think>answer"
        const k = generated.indexOf(this.thinkEndId);
        answering = k >= 0;
        reasoning = decode(answering ? generated.slice(0, k) : generated).trim();
        answer = answering ? decode(generated.slice(k + 1)).trimStart() : '';
      } else {
        answer = decode(generated);
      }
      const decodeSeconds = (performance.now() - tFirst) / 1000;
      return {
        reasoning,
        answer,
        answering,
        promptTokens: n,
        tokens: generated.length,
        prefillSeconds: (tFirst - t0) / 1000,
        tokensPerSecond: generated.length > 1 && decodeSeconds > 0 ? (generated.length - 1) / decodeSeconds : 0,
        applied,
        stopReason,
      };
    };

    try {
      for (;;) {
        const out = await this.session.run(feeds, fetches);
        for (const t of past) t.dispose(); // previous step's state, now consumed
        past = this.v.cache.map((c) => out[c.present]);
        if (!tFirst) tFirst = performance.now();

        const next = sample(toFloat32(out.logits.data as Float32Array | Uint16Array), o.sampling);
        if (this.stopIds.has(next)) {
          stopReason = 'eos';
          break;
        }
        generated.push(next);
        o.onUpdate?.(state(false));
        if (generated.length >= o.maxNewTokens) {
          stopReason = 'length';
          break;
        }
        if (o.shouldStop?.()) {
          stopReason = 'stopped';
          break;
        }

        const total = n + generated.length;
        feeds = {
          ...edits,
          ...Object.fromEntries(this.v.cache.map((c, i) => [c.past, past[i]])),
          inputs_embeds: new ort.Tensor('float32', embedRows(this.lmHead, [next]), [1, 1, this.hidden]),
          attention_mask: new ort.Tensor('int64', new BigInt64Array(total).fill(1n), [1, total]),
          position_ids: new ort.Tensor('int64', new BigInt64Array(3).fill(BigInt(total - 1)), [3, 1, 1]),
          num_logits_to_keep: this.one,
          trace_start: traceFrom0,
        };
      }
    } finally {
      for (const t of past) t.dispose();
      for (const t of owned) t.dispose();
    }
    return state(true);
  }
}
