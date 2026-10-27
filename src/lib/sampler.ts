// Sampling as in qwen-dose: greedy at temperature 0, otherwise temperature + top-k 20 + top-p 0.8
// (Qwen's recommended values; same order as transformers' logits warpers).

export interface SamplingParams {
  temperature: number;
  topK: number;
  topP: number;
}

export function argmax(logits: Float32Array): number {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

/** Indices of the k largest logits, largest first. */
function topK(logits: Float32Array, k: number): number[] {
  const idx: number[] = [];
  for (let i = 0; i < logits.length; i++) {
    const v = logits[i];
    if (idx.length === k && v <= logits[idx[k - 1]]) continue;
    let j = Math.min(idx.length, k - 1);
    while (j > 0 && logits[idx[j - 1]] < v) j--;
    idx.splice(j, 0, i);
    if (idx.length > k) idx.pop();
  }
  return idx;
}

export function sample(logits: Float32Array, p: SamplingParams, random = Math.random): number {
  if (p.temperature <= 0) return argmax(logits);
  const ids = topK(logits, Math.max(1, p.topK));
  const max = logits[ids[0]];
  const w = ids.map((i) => Math.exp((logits[i] - max) / p.temperature));
  const total = w.reduce((a, b) => a + b, 0);
  // top-p: smallest prefix (by descending probability) whose mass reaches topP
  let keep = 0;
  let mass = 0;
  while (keep < ids.length && (keep === 0 || mass < p.topP)) mass += w[keep++] / total;
  let r = random() * mass * total;
  for (let i = 0; i < keep; i++) {
    r -= w[i];
    if (r <= 0) return ids[i];
  }
  return ids[keep - 1];
}
