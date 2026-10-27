// Neuron interventions: add a dose to, scale, or set one MLP neuron's activation at every position.
//
// The patched graph computes  a' = a * mlp_scale.L + mlp_add.L  right before each layer's down_proj, where
// a = silu(gate_proj(x)) * up_proj(x) holds one value per neuron. "add" is qwen-dose's recipe (port of its
// steer.py): a dose of 1 adds  median(||resid|| leaving layer L) / ||W_down[:, n]||  to the activation, so the
// residual stream gets a push as large as its typical size, in the direction that neuron writes to.

import mediansJson from './medians.json';

export type Op = 'add' | 'scale' | 'set';

export interface Intervention {
  layer: number;
  neuron: number;
  op: Op;
  /** add: dose in residual-norm units; scale: multiplier (0 ablates); set: activation value */
  value: number;
  enabled: boolean;
}

/** What an intervention did to its neuron's activation:  a -> a * scale + add. */
export interface AppliedEdit {
  layer: number;
  neuron: number;
  scale: number;
  add: number;
}

export const NUM_LAYERS = 32;
export const NUM_NEURONS = 9216;
export const SPEC_HELP =
  'LAYER:NEURON:VALUE entries, separated by commas or spaces (layers 0..31, neurons 0..9215): 14:8949:+1 adds a ' +
  "dose (+1 = a push as large as the typical residual stream, in the neuron's write direction), 14:8949:*0 scales " +
  'the activation (0 ablates the neuron), 14:8949:=2 sets it. Doses can also be written as catalogs do, ' +
  'L14N8949+1. A leading ! keeps an entry but turns it off.';

/** median ||resid|| leaving each layer (scripts/calibrate.py; layers 13/14 are qwen-dose's stored values). */
export const MEDIANS: Record<number, number> = Object.fromEntries(
  Object.entries(mediansJson.medians).map(([k, v]) => [Number(k), v]),
);

const LIMITS: Record<Op, number> = { add: 10, scale: 100, set: 1000 };
const OP_NAMES: Record<Op, string> = { add: 'Dose', scale: 'Scale', set: 'Value' };
const NUMBER = '(?:\\d+\\.?\\d*|\\.\\d+)';
const ENTRY = new RegExp(`^(!?)(\\d+):(\\d+):([*x=]?)([+-]?${NUMBER})$`, 'i');
/** L9N7898+0.8: how neuron catalogs write a dose. */
const COMPACT = new RegExp(`^(!?)L(\\d+)N(\\d+)([+-]${NUMBER})$`, 'i');

export function validate(iv: Intervention): void {
  if (!Number.isInteger(iv.layer) || iv.layer < 0 || iv.layer >= NUM_LAYERS) {
    throw new Error(`Layer ${iv.layer} out of range 0..${NUM_LAYERS - 1}.`);
  }
  if (!Number.isInteger(iv.neuron) || iv.neuron < 0 || iv.neuron >= NUM_NEURONS) {
    throw new Error(`Neuron ${iv.neuron} out of range 0..${NUM_NEURONS - 1}.`);
  }
  if (!(Math.abs(iv.value) <= LIMITS[iv.op])) {
    throw new Error(`${OP_NAMES[iv.op]} ${iv.value} out of range -${LIMITS[iv.op]}..${LIMITS[iv.op]}.`);
  }
}

export function checkDuplicates(list: Intervention[]): void {
  const seen = new Set<string>();
  for (const iv of list) {
    const key = `${iv.layer}:${iv.neuron}`;
    if (seen.has(key)) throw new Error(`Layer ${iv.layer} neuron ${iv.neuron} appears twice.`);
    seen.add(key);
  }
}

/**
 * Strictly parse '14:8949:+1,13:7956:-1,!20:5:*0' or 'L14N8949+1 L13N7956-1' (spaces inside an entry are fine).
 * Throws an Error with a readable message.
 */
export function parseSpec(s: string): Intervention[] {
  const normalized = s
    .replace(/\bL\s*(\d+)\s*N\s*(\d+)\s*([+-])\s*/gi, 'L$1N$2$3')
    .replace(/\s*:\s*([*x=]?)\s*([+-]?)\s*/gi, ':$1$2');
  const out: Intervention[] = [];
  for (const part of normalized.split(/[\s,;]+/).filter(Boolean)) {
    const colon = ENTRY.exec(part);
    const compact = colon ? null : COMPACT.exec(part);
    if (!colon && !compact) {
      throw new Error(`Bad entry "${part}": expected e.g. 14:8949:+1 (add), 14:8949:*0 (scale), 14:8949:=2 (set) or L14N8949+1.`);
    }
    const iv: Intervention = colon
      ? {
          layer: Number(colon[2]),
          neuron: Number(colon[3]),
          op: colon[4] === '=' ? 'set' : colon[4] ? 'scale' : 'add',
          value: Number(colon[5]),
          enabled: !colon[1],
        }
      : { layer: Number(compact![2]), neuron: Number(compact![3]), op: 'add', value: Number(compact![4]), enabled: !compact![1] };
    validate(iv);
    out.push(iv);
  }
  checkDuplicates(out);
  return out;
}

/**
 * `extra` merged into `base` (both unchanged): new neurons are appended; on a neuron in both, doses add up (steering
 * vectors add) and any other overlap takes the new entry.
 */
export function mergeInterventions(base: Intervention[], extra: Intervention[]): Intervention[] {
  const out = base.map((iv) => ({ ...iv }));
  for (const iv of extra) {
    const same = out.find((x) => x.layer === iv.layer && x.neuron === iv.neuron);
    if (!same) out.push({ ...iv });
    else if (same.op === 'add' && iv.op === 'add') Object.assign(same, { value: Number((same.value + iv.value).toFixed(6)), enabled: true });
    else Object.assign(same, iv);
  }
  return out;
}

const num = (x: number) => String(Number(x.toPrecision(4)));
const signed = (x: number, minus = '-') => (x >= 0 ? '+' : minus) + num(Math.abs(x));

export function formatSpec(list: Intervention[]): string {
  return list
    .map((iv) => {
      const v = iv.op === 'add' ? signed(iv.value) : (iv.op === 'scale' ? '*' : '=') + num(iv.value);
      return `${iv.enabled ? '' : '!'}${iv.layer}:${iv.neuron}:${v}`;
    })
    .join(',');
}

/** "L14 N8949 +1, L20 N5 ×0, L3 N9 =2" for the enabled entries. */
export function describe(list: Intervention[]): string {
  const on = list.filter((iv) => iv.enabled);
  if (!on.length) return 'no interventions';
  return on
    .map((iv) => {
      const v = iv.op === 'add' ? signed(iv.value, '−') : (iv.op === 'scale' ? '×' : '=') + num(iv.value);
      return `L${iv.layer} N${iv.neuron} ${v}`;
    })
    .join(', ');
}

/** "L14 N8949 a + 22.81": the activation change actually applied, as qwen-dose's CLI reports deltas. */
export function describeApplied(edits: AppliedEdit[]): string {
  return edits
    .map(({ layer, neuron, scale, add }) => {
      const formula =
        scale === 1
          ? `a ${signed(add, '−').replace(/^(.)/, '$1 ')}`
          : scale === 0
            ? `= ${num(add)}`
            : `a × ${num(scale)}${add ? ` ${signed(add, '−').replace(/^(.)/, '$1 ')}` : ''}`;
      return `L${layer} N${neuron} ${formula}`;
    })
    .join(', ');
}

/**
 * One neuron's edit at strength s (0 = the unmodified model, 1 = as written):
 * add d -> a + s·d·median/‖W_down[:, n]‖;  scale k -> a·(1 + s·(k − 1));  set c -> (1 − s)·a + s·c.
 */
export function neuronEdit(iv: Intervention, strength: number, columnNorm: number): AppliedEdit {
  const s = strength;
  const { layer, neuron } = iv;
  switch (iv.op) {
    case 'add':
      return { layer, neuron, scale: 1, add: (s * iv.value * MEDIANS[layer]) / columnNorm };
    case 'scale':
      return { layer, neuron, scale: 1 + s * (iv.value - 1), add: 0 };
    case 'set':
      return { layer, neuron, scale: 1 - s, add: s * iv.value };
  }
}
