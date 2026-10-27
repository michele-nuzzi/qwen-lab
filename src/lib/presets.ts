// Known steerings for Qwen3.5-4B, in this page's LAYER:NEURON:VALUE notation. Their doses use the qwen-dose recipe
// (dose × median ‖resid‖ / ‖W_down[:, n]‖ at the down_proj input), as do catalogs such as Poor Things.

import { formatSpec, mergeInterventions, parseSpec, type Intervention } from './interventions';

export interface Preset {
  name: string;
  spec: string;
  source?: { label: string; url: string };
}

export const PRESETS: Preset[] = [
  { name: 'Evil Qwen', spec: '14:8949:+1,13:7956:-1' },
  {
    name: 'Toxic Positivity',
    spec: '9:7898:+0.8',
    source: { label: 'Poor Things #168', url: 'https://poor-things.com/neurotype/168' },
  },
  { name: 'Anime Girl', spec: '13:4810:+1.5' },
];

export const DEFAULT_PRESET = PRESETS[0];

const PARSED = PRESETS.map((p) => parseSpec(p.spec));
const canonical = (list: Intervention[]) => list.map((iv) => formatSpec([iv])).sort().join(',');

/**
 * The fewest presets (indices into PRESETS) whose merge is exactly `list`, in the order their neurons appear in it
 * (so merging Anime Girl into Toxic Positivity reads "Toxic Positivity + Anime Girl"), or null for a custom list.
 */
export function presetsMatching(list: Intervention[]): number[] | null {
  if (!list.length) return null;
  const target = canonical(list);
  const n = Math.min(PRESETS.length, 12);
  let best: number[] | null = null;
  for (let mask = 1; mask < 1 << n; mask++) {
    const picked = PRESETS.map((_, i) => i).filter((i) => i < n && mask & (1 << i));
    if (best && picked.length >= best.length) continue;
    const merged = picked.reduce((acc, i) => mergeInterventions(acc, PARSED[i]), [] as Intervention[]);
    if (canonical(merged) === target) best = picked;
  }
  const position = (i: number) =>
    Math.min(...PARSED[i].map((iv) => list.findIndex((x) => x.layer === iv.layer && x.neuron === iv.neuron)));
  return best && best.sort((a, b) => position(a) - position(b));
}
