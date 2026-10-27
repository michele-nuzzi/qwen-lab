// Finding neurons: run a prompt (and optionally a contrast prompt) through the unmodified model and rank neurons by
// how strongly they fire on it, across all layers or within one.

import { NUM_LAYERS } from '../lib/interventions';
import type { TraceResult } from '../lib/model';
import * as client from './client';
import { $, button, el, errorText, signed } from './dom';
import { lab } from './lab';

const SHOW = 12;

let traces: { a: TraceResult; b: TraceResult | null } | null = null;

const ui = {
  a: $<HTMLTextAreaElement>('trace-a'),
  b: $<HTMLTextAreaElement>('trace-b'),
  run: $<HTMLButtonElement>('trace-run'),
  status: $('trace-status'),
  out: $('trace-out'),
  layer: $<HTMLSelectElement>('trace-layer'),
  top: $('trace-top'),
  bottom: $('trace-bottom'),
  topTitle: $('trace-top-title'),
  bottomTitle: $('trace-bottom-title'),
};

interface Hooks {
  /** Whether the model is free (not generating); the trace needs the GPU too. */
  canRun: () => boolean;
  setBusy: (busy: boolean) => void;
  inspect: (layer: number, neuron: number) => void;
}
let hooks: Hooks;

interface Hit {
  layer: number;
  neuron: number;
  value: number;
  /** standard deviations from the layer's mean: comparable across layers, whose activation scales differ a lot */
  z: number;
}

/** The neuron's activation range on the last traced prompt, for the inspect dialog. */
export function lastTrace(layer: number, neuron: number): { max: number; min: number } | null {
  if (!traces) return null;
  const i = layer * traces.a.width + neuron;
  return { max: traces.a.max[i], min: traces.a.min[i] };
}

/** The SHOW most extreme neurons of `score` (layer-major) in the given layers, highest first or lowest first. */
function extremes(score: Float32Array, width: number, layers: number[], highest: boolean): Hit[] {
  const sign = highest ? 1 : -1;
  const hits: Hit[] = [];
  for (const layer of layers) {
    const from = layer * width;
    let mean = 0;
    for (let i = 0; i < width; i++) mean += score[from + i];
    mean /= width;
    let sq = 0;
    for (let i = 0; i < width; i++) sq += (score[from + i] - mean) ** 2;
    const std = Math.sqrt(sq / width) || 1;
    const order = Array.from({ length: width }, (_, i) => i).sort(
      (x, y) => sign * (score[from + y] - score[from + x]),
    );
    for (const neuron of order.slice(0, SHOW)) {
      const value = score[from + neuron];
      hits.push({ layer, neuron, value, z: (value - mean) / std });
    }
  }
  return hits.sort((x, y) => sign * (y.z - x.z)).slice(0, SHOW);
}

function item(hit: Hit, dose: number, allLayers: boolean): HTMLElement {
  const { layer, neuron } = hit;
  const li = el('li');
  li.append(
    button(`L${layer} N${neuron}`, 'link mono', () => hooks.inspect(layer, neuron), 'What does this neuron write?'),
    el('span', 'val mono', signed(hit.value) + (allLayers ? ` · z ${hit.z.toFixed(1)}` : '')),
    button(dose > 0 ? '+1' : '−1', 'ghost tiny', () => lab.add({ layer, neuron, op: 'add', value: dose }), `Add a ${dose > 0 ? '+1' : '−1'} dose`),
    button('×0', 'ghost tiny', () => lab.add({ layer, neuron, op: 'scale', value: 0 }), 'Ablate (scale by 0)'),
  );
  return li;
}

function show() {
  if (!traces) return;
  const { a, b } = traces;
  const allLayers = ui.layer.value === 'all';
  const layers = allLayers ? Array.from({ length: NUM_LAYERS }, (_, l) => l) : [Number(ui.layer.value)];
  // contrast: how much more each neuron peaks on the first prompt than on the second
  const diff = b ? a.max.map((x, i) => x - b.max[i]) : null;
  const top = extremes(diff ?? a.max, a.width, layers, true);
  const bottom = extremes(diff ?? a.min, a.width, layers, false);
  const where = allLayers ? ', all layers by z-score' : '';
  ui.topTitle.textContent = (b ? 'Fire more on the first prompt' : 'Highest activation') + where;
  ui.bottomTitle.textContent = (b ? 'Fire more on the contrast' : 'Most negative activation') + where;
  ui.top.replaceChildren(...top.map((h) => item(h, 1, allLayers)));
  ui.bottom.replaceChildren(...bottom.map((h) => item(h, -1, allLayers)));
}

export function initFinder(h: Hooks) {
  hooks = h;
  const all = el('option', '', 'All layers');
  all.value = 'all';
  ui.layer.append(all);
  for (let l = 0; l < NUM_LAYERS; l++) {
    const o = el('option', '', `Layer ${l}`);
    o.value = String(l);
    ui.layer.append(o);
  }
  ui.layer.addEventListener('change', show);
  ui.run.addEventListener('click', async () => {
    const promptA = ui.a.value.trim();
    const promptB = ui.b.value.trim();
    if (!promptA) return ui.a.focus();
    if (!hooks.canRun()) return;
    hooks.setBusy(true);
    ui.status.textContent = 'Tracing…';
    try {
      const a = await client.trace([{ role: 'user', content: promptA }]);
      const b = promptB ? await client.trace([{ role: 'user', content: promptB }]) : null;
      traces = { a, b };
      ui.status.textContent = `${a.tokens}${b ? ` and ${b.tokens}` : ''} tokens traced across ${NUM_LAYERS} layers (first token skipped).`;
      ui.out.hidden = false;
      show();
    } catch (e) {
      ui.status.textContent = errorText(e);
    } finally {
      hooks.setBusy(false);
    }
  });
}

/** Enable tracing once the model is loaded, and while it isn't generating. */
export function setFinderEnabled(on: boolean) {
  ui.run.disabled = !on;
}
