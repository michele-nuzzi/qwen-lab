// "What does this neuron write?": its down_proj column, the dose scale, and the tokens its write direction pushes
// up or down (through the final norm and the tied lm_head; a direct-logit view, rough for middle layers).

import type { NeuronInfo } from '../lib/model';
import * as client from './client';
import { $, el, errorText, signed } from './dom';
import { lastTrace } from './finder';
import { lab } from './lab';

const ui = {
  dialog: $<HTMLDialogElement>('neuron'),
  title: $('neuron-title'),
  body: $('neuron-body'),
};
let current: { layer: number; neuron: number } | null = null;
let modelReady = () => false;

const visible = (t: string) => t.replace(/ /g, '␣').replace(/\n/g, '⏎').replace(/\t/g, '⇥') || '∅';

function tokens(label: string, list: NeuronInfo['promotes']): HTMLElement {
  const sec = el('section', 'tokens');
  const chips = el('div', 'chips');
  for (const t of list) {
    const chip = el('span', 'chip-token mono', visible(t.token));
    chip.title = `logit ${signed(t.score)}`;
    chips.append(chip);
  }
  sec.append(el('h3', '', label), chips);
  return sec;
}

export async function showNeuron(layer: number, neuron: number) {
  current = { layer, neuron };
  ui.title.textContent = `Layer ${layer} · neuron ${neuron}`;
  ui.body.replaceChildren(el('p', 'muted', modelReady() ? 'Reading the weights…' : 'Load the model to inspect neurons.'));
  if (!ui.dialog.open) ui.dialog.showModal();
  if (!modelReady()) return;
  try {
    const info = await client.inspect(layer, neuron);
    if (current?.layer !== layer || current.neuron !== neuron) return;
    const stats = el(
      'p',
      'mono small',
      `‖W_down[:, n]‖ ${info.columnNorm.toFixed(4)} · median ‖resid‖ after layer ${layer}: ${info.median.toFixed(3)} · ` +
        `a dose of 1 adds ${info.deltaPerDose.toFixed(2)} to its activation`,
    );
    const nodes: HTMLElement[] = [stats];
    const seen = lastTrace(layer, neuron);
    if (seen) {
      nodes.push(
        el('p', 'small', `On the last traced prompt it ranged from ${signed(seen.min)} to ${signed(seen.max)}.`),
      );
    }
    nodes.push(tokens('Pushes up', info.promotes), tokens('Pushes down', info.suppresses));
    ui.body.replaceChildren(...nodes);
  } catch (e) {
    ui.body.replaceChildren(el('p', 'error-text', errorText(e)));
  }
}

export function initNeuronDialog(isReady: () => boolean) {
  modelReady = isReady;
  $('neuron-close').addEventListener('click', () => ui.dialog.close());
  ui.dialog.addEventListener('click', (e) => {
    if (e.target === ui.dialog) ui.dialog.close();
  });
  $('neuron-add').addEventListener('click', () => {
    if (current) lab.add({ ...current, op: 'add', value: 1 });
    ui.dialog.close();
  });
  $('neuron-ablate').addEventListener('click', () => {
    if (current) lab.add({ ...current, op: 'scale', value: 0 });
    ui.dialog.close();
  });
}
