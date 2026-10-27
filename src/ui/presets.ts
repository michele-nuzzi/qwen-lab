// Known steerings: cards on the Neurons page, and a quick switch with a summary on the Chat page. Either can merge
// presets, e.g. Anime Girl + Toxic Positivity.

import { describe, parseSpec } from '../lib/interventions';
import { PRESETS } from '../lib/presets';
import { $, button, el } from './dom';
import { lab } from './lab';

const CUSTOM = 'custom';

function option(value: string, text: string): HTMLOptionElement {
  const o = el('option', '', text);
  o.value = value;
  return o;
}

export function initPresets(editNeurons: () => void) {
  const cards = PRESETS.map((p) => {
    const card = el('article', 'preset');
    const head = el('div', 'preset-head');
    head.append(el('h3', '', p.name));
    if (p.source) {
      const a = el('a', 'small', p.source.label);
      Object.assign(a, { href: p.source.url, target: '_blank', rel: 'noopener noreferrer' });
      head.append(a);
    }
    const actions = el('div', 'preset-actions');
    actions.append(
      button('Use', 'ghost small', () => lab.load(p.spec), `Replace the interventions with ${p.name}`),
      button('Merge', 'ghost small', () => lab.merge(p.spec), `Add ${p.name} to the current interventions`),
    );
    card.append(head, el('p', 'preset-spec mono', describe(parseSpec(p.spec))), actions);
    return card;
  });
  $('preset-list').replaceChildren(...cards);

  const select = $<HTMLSelectElement>('quick-preset');
  const custom = option(CUSTOM, 'Custom');
  select.append(custom, ...PRESETS.map((p, i) => option(String(i), p.name)));
  select.addEventListener('change', () => {
    if (select.value !== CUSTOM) lab.load(PRESETS[Number(select.value)].spec);
  });

  const merge = $<HTMLSelectElement>('merge-preset');
  merge.append(option('', '+ Merge…'), ...PRESETS.map((p, i) => option(String(i), p.name)));
  merge.addEventListener('change', () => {
    if (merge.value) lab.merge(PRESETS[Number(merge.value)].spec);
    merge.value = '';
  });

  $('edit-neurons').addEventListener('click', editNeurons);

  const sync = () => {
    const match = lab.matching();
    cards.forEach((c, i) => c.classList.toggle('active', !!match?.includes(i)));
    if (match?.length === 1) {
      custom.hidden = true;
      select.value = String(match[0]);
    } else {
      custom.textContent = match ? match.map((i) => PRESETS[i].name).join(' + ') : 'Custom';
      custom.hidden = false;
      select.value = CUSTOM;
    }
    $('steering-summary').textContent = lab.error()
      ? 'the interventions need fixing'
      : `${lab.describe()} · strength ×${Number(lab.strength().toFixed(2))}`;
    $('lab-label').textContent = lab.error()
      ? ''
      : match
        ? match.map((i) => PRESETS[i].name).join(' + ')
        : lab.interventions().length
          ? 'Custom'
          : 'Nothing: the unmodified model';
  };
  lab.onChange(sync);
  sync();
}
