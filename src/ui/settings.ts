// Generation settings, remembered per browser (a convenience only; the page works without storage).

export interface Settings {
  temperature: number;
  maxNewTokens: number;
  thinking: boolean;
  /** Strengths compared side by side, e.g. "1, 0" (0 = the unmodified model, always shown last). */
  compare: string;
}

export const DEFAULTS: Settings = {
  temperature: 0.7,
  maxNewTokens: 256,
  thinking: false,
  compare: '1, 0',
};

const KEY = 'evil-qwen:settings';
const MAX_COLUMNS = 6;

/** "0, 0.5, 1" -> [0, 0.5, 1]; throws a readable Error. */
export function parseStrengths(s: string): number[] {
  const parts = s.split(/[\s,]+/).filter(Boolean);
  if (!parts.length) throw new Error('List at least one strength, e.g. 0, 1.');
  if (parts.length > MAX_COLUMNS) throw new Error(`At most ${MAX_COLUMNS} strengths.`);
  return parts.map((p) => {
    const x = Number(p);
    if (!Number.isFinite(x) || x < 0 || x > 5) throw new Error(`Strength "${p}" should be a number from 0 to 5.`);
    return x;
  });
}

export function loadSettings(): Settings {
  try {
    const saved = JSON.parse(localStorage.getItem(KEY) ?? '{}') as Partial<Settings>;
    const out = { ...DEFAULTS };
    for (const k of Object.keys(DEFAULTS) as (keyof Settings)[]) {
      if (typeof saved[k] === typeof DEFAULTS[k]) (out as Record<string, unknown>)[k] = saved[k];
    }
    return out;
  } catch {
    return { ...DEFAULTS };
  }
}

export function saveSettings(s: Settings): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(s));
  } catch {
    // storage blocked (private mode, sandboxed preview): settings just won't persist
  }
}
