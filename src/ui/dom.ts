export const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

export function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls?: string, text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

export function button(text: string, cls: string, onClick: () => void, title?: string): HTMLButtonElement {
  const b = el('button', cls, text);
  b.type = 'button';
  if (title) b.title = title;
  b.addEventListener('click', onClick);
  return b;
}

export const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** "+0.25", "−1.5" with a typographic minus. */
export const signed = (x: number, digits = 2) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(digits);
