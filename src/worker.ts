// Model worker: owns the WebGPU session so the page stays responsive.

import { FileCache } from './lib/download';
import { CACHE_ROOT, downloadBytes, getGpu, isCached, SteeredQwen, type GpuInfo } from './lib/model';
import type { FromWorker, ToWorker } from './protocol';

const post = (m: FromWorker, transfer: Transferable[] = []) => self.postMessage(m, { transfer });
const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

let gpu: Promise<GpuInfo> | null = null;
let model: SteeredQwen | null = null;
let loading = false;
let stopRequested = false;

self.onmessage = async (e: MessageEvent<ToWorker>) => {
  const msg = e.data;
  switch (msg.type) {
    case 'probe': {
      gpu ??= getGpu();
      try {
        const g = await gpu;
        post({
          type: 'probe',
          gpu: { name: g.name, fallback: g.fallback, variant: g.variant },
          gpuError: null,
          downloadBytes: downloadBytes(g.variant),
          cached: await isCached(g.variant).catch(() => false),
        });
      } catch (err) {
        post({ type: 'probe', gpu: null, gpuError: errorText(err), downloadBytes: downloadBytes('q4f16'), cached: false });
      }
      return;
    }

    case 'load': {
      if (model || loading) return;
      loading = true;
      const t0 = performance.now();
      let lastPost = 0;
      try {
        gpu ??= getGpu();
        model = await SteeredQwen.load(await gpu, msg.baseUrl, (progress) => {
          // byte-level progress fires per network packet; ~10 updates a second is plenty
          const now = performance.now();
          if (progress.phase === 'weights' && progress.loaded !== progress.total && now - lastPost < 100) return;
          lastPost = now;
          post({ type: 'progress', progress });
        });
        post({ type: 'ready', seconds: (performance.now() - t0) / 1000, cached: model.cachedLoad });
      } catch (err) {
        console.error(err);
        post({ type: 'error', message: errorText(err) });
      } finally {
        loading = false;
      }
      return;
    }

    case 'generate':
    case 'trace':
    case 'inspect': {
      const { id } = msg;
      if (!model) return post({ type: 'error', id, message: 'The model is not loaded yet.' });
      try {
        if (msg.type === 'generate') {
          stopRequested = false;
          const s = msg.settings;
          const result = await model.generate({
            messages: msg.messages,
            thinking: s.thinking,
            interventions: s.interventions,
            strength: s.strength,
            maxNewTokens: s.maxNewTokens,
            sampling: { temperature: s.temperature, topK: 20, topP: 0.8 },
            onUpdate: (state) => post({ type: 'update', id, state }),
            shouldStop: () => stopRequested,
          });
          post({ type: 'result', id, result });
        } else if (msg.type === 'trace') {
          const result = await model.trace(msg.messages);
          post({ type: 'result', id, result }, [result.max.buffer, result.min.buffer]);
        } else {
          post({ type: 'result', id, result: await model.inspect(msg.layer, msg.neuron) });
        }
      } catch (err) {
        console.error(err);
        post({ type: 'error', id, message: errorText(err) });
      }
      return;
    }

    case 'stop':
      stopRequested = true;
      return;

    case 'clear-cache':
      await FileCache.clear(CACHE_ROOT);
      post({ type: 'cache-cleared' });
      return;
  }
};
