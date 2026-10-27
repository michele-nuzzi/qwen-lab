// The interventions editor: rows of layer / neuron / operation / value, the same list as text, a strength slider,
// and a shareable link. The browser remembers the editor; a link carries it in the URL hash (never sent to a
// server), which is applied once on arrival and then dropped, so Back/Forward between pages never rewinds edits.

import {
  checkDuplicates,
  describe,
  formatSpec,
  mergeInterventions,
  NUM_LAYERS,
  NUM_NEURONS,
  parseSpec,
  SPEC_HELP,
  validate,
  type Intervention,
  type Op,
} from '../lib/interventions';
import { DEFAULT_PRESET, presetsMatching } from '../lib/presets';
import { $, button, el, errorText } from './dom';
import { hashParams, setHashParams } from './route';

const STORE_KEY = 'evil-qwen:lab';
const DEFAULT_VALUE: Record<Op, number> = { add: 1, scale: 0, set: 0 };

let list: Intervention[] = [];
let strength = 1;
let error: string | null = null;
let onInspect: (layer: number, neuron: number) => void = () => {};
const listeners: (() => void)[] = [];

const ui = {
  rows: $('iv-rows'),
  text: $<HTMLInputElement>('iv-text'),
  error: $('iv-error'),
  strength: $<HTMLInputElement>('strength'),
  strengthValue: $('strength-value'),
};

export const lab = {
  /** Enabled interventions (only meaningful while `error()` is null). */
  interventions: (): Intervention[] => list.filter((iv) => iv.enabled).map((iv) => ({ ...iv })),
  strength: () => strength,
  error: () => error,
  describe: () => describe(list),
  /** The presets (indices in PRESETS) the list is made of: one, a merge of several, or null for a custom list. */
  matching: (): number[] | null => (error ? null : presetsMatching(list)),
  /** Replace the list (a preset or a pasted spec); strength back to ×1. */
  load(spec: string) {
    list = parseSpec(spec);
    strength = 1;
    ui.strength.value = '1';
    render();
    changed();
  },
  /** Merge a preset into the list: its neurons are added; doses on a neuron already there add up. */
  merge(spec: string) {
    if (error) return;
    list = mergeInterventions(list, parseSpec(spec));
    render();
    changed();
  },
  /** Add a neuron, or update it if it is already in the list. */
  add(iv: Omit<Intervention, 'enabled'>) {
    const existing = list.find((x) => x.layer === iv.layer && x.neuron === iv.neuron);
    if (existing) Object.assign(existing, iv, { enabled: true });
    else list.push({ ...iv, enabled: true });
    render();
    changed();
  },
  onChange: (f: () => void) => listeners.push(f),
  onInspect: (f: (layer: number, neuron: number) => void) => (onInspect = f),
};

function changed(fromText = false) {
  error = null;
  try {
    list.forEach((iv, i) => {
      try {
        validate(iv);
      } catch (e) {
        throw new Error(`Row ${i + 1}: ${errorText(e)}`);
      }
    });
    checkDuplicates(list);
  } catch (e) {
    error = errorText(e);
  }
  showError(error);
  if (!error && !fromText) ui.text.value = formatSpec(list);
  ui.strengthValue.textContent = `×${strength.toFixed(2)}`;
  if (!error) save();
  for (const f of listeners) f();
}

function showError(message: string | null) {
  ui.error.textContent = message ?? '';
  ui.error.hidden = !message;
  ui.text.setAttribute('aria-invalid', String(!!message));
}

function numberField(label: string, value: number, max: number, set: (n: number) => void, title: string) {
  const wrap = el('label', 'iv-field');
  wrap.title = title;
  const input = el('input');
  Object.assign(input, { type: 'number', min: '0', max: String(max), step: '1', inputMode: 'numeric', value: String(value) });
  input.addEventListener('input', () => {
    set(input.valueAsNumber);
    changed();
  });
  wrap.append(el('span', '', label), input);
  return wrap;
}

function row(iv: Intervention, index: number): HTMLElement {
  const r = el('div', `iv-row${iv.enabled ? '' : ' off'}`);
  const on = el('input');
  Object.assign(on, { type: 'checkbox', checked: iv.enabled, title: 'On / off' });
  on.addEventListener('change', () => {
    iv.enabled = on.checked;
    r.classList.toggle('off', !on.checked);
    changed();
  });

  const op = el('select', 'iv-op');
  op.title = 'add: a dose along the neuron’s write direction (+1 = typical residual size); scale: multiply the activation; set: replace it';
  for (const [value, text] of [
    ['add', 'add dose'],
    ['scale', 'scale ×'],
    ['set', 'set ='],
  ]) {
    const o = el('option', '', text);
    o.value = value;
    op.append(o);
  }
  op.value = iv.op;
  const value = el('input', 'iv-value');
  Object.assign(value, { type: 'number', step: '0.1', value: String(iv.value), title: 'Value' });
  op.addEventListener('change', () => {
    iv.op = op.value as Op;
    iv.value = DEFAULT_VALUE[iv.op];
    value.value = String(iv.value);
    changed();
  });
  value.addEventListener('input', () => {
    iv.value = value.valueAsNumber;
    changed();
  });

  const actions = el('span', 'iv-actions');
  actions.append(
    button('ⓘ', 'ghost icon', () => onInspect(iv.layer, iv.neuron), 'What does this neuron write?'),
    button('✕', 'ghost icon', () => {
      list.splice(index, 1);
      render();
      changed();
    }, 'Remove'),
  );
  r.append(
    on,
    numberField('L', iv.layer, NUM_LAYERS - 1, (n) => (iv.layer = n), 'Layer (0–31)'),
    numberField('N', iv.neuron, NUM_NEURONS - 1, (n) => (iv.neuron = n), 'Neuron (0–9215)'),
    op,
    value,
    actions,
  );
  return r;
}

function render() {
  ui.rows.replaceChildren(...list.map(row));
  if (!list.length) ui.rows.append(el('p', 'muted small', 'No interventions: Modified replies are the unmodified model.'));
}

function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ spec: formatSpec(list), strength }));
  } catch {
    // storage blocked: the editor just won't be remembered
  }
}

function shareLink(): string {
  return `${location.origin}${location.pathname}${location.search}#iv=${encodeURI(formatSpec(list))}&s=${strength}`;
}

/** Linked or saved state: a link's hash wins over what this browser remembers. */
function restore(): { spec: string; strength: number } {
  const params = hashParams();
  if (params.has('iv')) return { spec: params.get('iv') ?? '', strength: Number(params.get('s') ?? 1) };
  try {
    const saved = JSON.parse(localStorage.getItem(STORE_KEY) ?? 'null');
    if (saved && typeof saved.spec === 'string') return { spec: saved.spec, strength: Number(saved.strength ?? 1) };
  } catch {
    // ignore
  }
  return { spec: DEFAULT_PRESET.spec, strength: 1 };
}

function applyRestored() {
  const saved = restore();
  try {
    list = parseSpec(saved.spec);
  } catch {
    list = parseSpec(DEFAULT_PRESET.spec);
  }
  strength = Number.isFinite(saved.strength) ? Math.min(2, Math.max(0, saved.strength)) : 1;
  ui.strength.value = String(strength);
  setHashParams({ iv: null, s: null }); // applied; from here on the browser remembers it
  render();
  changed();
}

export function initLab() {
  $('iv-help').textContent = SPEC_HELP;
  ui.text.addEventListener('input', () => {
    try {
      list = parseSpec(ui.text.value);
      render();
      changed(true);
    } catch (e) {
      error = errorText(e);
      showError(error);
      for (const f of listeners) f();
    }
  });
  ui.strength.addEventListener('input', () => {
    strength = Number(ui.strength.value);
    changed();
  });
  $('iv-add').addEventListener('click', () => {
    const last = list[list.length - 1];
    list.push({ layer: last?.layer ?? 14, neuron: 0, op: 'add', value: 1, enabled: true });
    render();
    changed();
    ui.rows.querySelector<HTMLInputElement>('.iv-row:last-child .iv-field:nth-of-type(2) input')?.select();
  });
  $('iv-share').addEventListener('click', async () => {
    const b = $<HTMLButtonElement>('iv-share');
    try {
      await navigator.clipboard.writeText(shareLink()); // opens the chat with these interventions
      b.textContent = 'Link copied';
    } catch {
      prompt('Copy this link:', shareLink());
    }
    setTimeout(() => (b.textContent = 'Copy link'), 1600);
  });
  // a link pasted into this tab
  window.addEventListener('hashchange', () => {
    if (hashParams().has('iv')) applyRestored();
  });
  applyRestored();
}
