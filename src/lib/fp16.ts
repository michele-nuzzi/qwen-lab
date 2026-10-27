// IEEE half -> float. ORT returns float16 tensors as Float16Array where the browser has it,
// otherwise as raw Uint16Array bits; scales in the weight files are always raw bits.

let table: Float32Array | null = null;

export function halfTable(): Float32Array {
  if (table) return table;
  table = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = h & 0x8000 ? -1 : 1;
    const exp = (h >> 10) & 0x1f;
    const frac = h & 0x3ff;
    if (exp === 0) table[h] = sign * frac * 2 ** -24;
    else if (exp === 31) table[h] = frac ? NaN : sign * Infinity;
    else table[h] = sign * (1 + frac / 1024) * 2 ** (exp - 15);
  }
  return table;
}

/** A float tensor's data as a Float32Array (float32 as is; float16 as Float16Array or raw bits, converted). */
export function toFloat32(data: ArrayLike<number>): Float32Array {
  if (data instanceof Float32Array) return data;
  const out = new Float32Array(data.length);
  if (data instanceof Uint16Array) {
    const t = halfTable();
    for (let i = 0; i < data.length; i++) out[i] = t[data[i]];
  } else {
    for (let i = 0; i < data.length; i++) out[i] = data[i];
  }
  return out;
}

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** float -> IEEE half bits, rounding to nearest even. */
export function halfBits(x: number): number {
  f32[0] = x;
  const b = u32[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0); // inf, nan
  const e = exp - 127 + 15;
  if (e >= 0x1f) return sign | 0x7c00; // too large: inf
  let half: number;
  let rest: number;
  let halfway: number;
  if (e <= 0) {
    if (e < -10) return sign; // too small: zero
    mant |= 0x800000; // subnormal half
    const shift = 14 - e;
    half = mant >> shift;
    rest = mant & ((1 << shift) - 1);
    halfway = 1 << (shift - 1);
  } else {
    half = (e << 10) | (mant >> 13);
    rest = mant & 0x1fff;
    halfway = 0x1000;
  }
  if (rest > halfway || (rest === halfway && half & 1)) half++; // a carry correctly bumps the exponent
  return sign | half;
}

export function toHalf(values: Float32Array): Uint16Array {
  const out = new Uint16Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = halfBits(values[i]);
  return out;
}
