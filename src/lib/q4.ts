// 4-bit MatMulNBits weights, read straight from the ONNX external-data bytes (mirror of scripts/q4.py).
//
// Layout (block_size 32, asymmetric): quant [N, K/32, 16] uint8 (two values per byte, low nibble first),
// scales [N, K/32] fp16 (q4f16 build) or fp32 (q4 build), zero points [N, K/64] uint8 (two blocks per byte,
// low nibble first).
// w[n, k] = (q[n, k] - zp[n, k / 32]) * scale[n, k / 32]

import { halfTable } from './fp16';

export interface Q4Tensor {
  quant: Uint8Array;
  /** fp16 bits or float32 */
  scales: Uint16Array | Float32Array;
  zp: Uint8Array;
  /** [N, blocks, 16] of the quant tensor */
  dims: number[];
}

const BLOCK = 32;

function scaleReader(scales: Uint16Array | Float32Array): (i: number) => number {
  if (scales instanceof Float32Array) return (i) => scales[i];
  const half = halfTable();
  return (i) => half[scales[i]];
}

/**
 * Token embeddings from the tied, quantized lm_head (the upstream embed_tokens_q4f16 data file is
 * byte-identical to it). Row `id` of the [vocab, hidden] matrix, dequantized to float32.
 */
export function embedRows(lmHead: Q4Tensor, ids: ArrayLike<number>): Float32Array {
  const [, blocks, blockBytes] = lmHead.dims;
  const hidden = blocks * BLOCK;
  const scale = scaleReader(lmHead.scales);
  const out = new Float32Array(ids.length * hidden);
  let o = 0;
  for (let t = 0; t < ids.length; t++) {
    const id = ids[t];
    const qBase = id * blocks * blockBytes;
    const sBase = id * blocks;
    const zBase = id * ((blocks + 1) >> 1);
    for (let b = 0; b < blocks; b++) {
      const s = scale(sBase + b);
      const zByte = lmHead.zp[zBase + (b >> 1)];
      const z = b & 1 ? zByte >> 4 : zByte & 0xf;
      const q0 = qBase + b * blockBytes;
      for (let j = 0; j < blockBytes; j++) {
        const byte = lmHead.quant[q0 + j];
        out[o++] = ((byte & 0xf) - z) * s;
        out[o++] = ((byte >> 4) - z) * s;
      }
    }
  }
  return out;
}

/** W_down[:, neuron]: the residual-stream direction one MLP neuron writes to (length = hidden size). */
export function downProjColumn(down: Q4Tensor, neuron: number): Float32Array {
  const [rows, blocks, blockBytes] = down.dims;
  const scale = scaleReader(down.scales);
  const b = Math.floor(neuron / BLOCK);
  const j = neuron % BLOCK;
  const shift = 4 * (j & 1);
  const zShift = 4 * (b & 1);
  const zStride = (blocks + 1) >> 1;
  const col = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    const q = (down.quant[(r * blocks + b) * blockBytes + (j >> 1)] >> shift) & 0xf;
    const z = (down.zp[r * zStride + (b >> 1)] >> zShift) & 0xf;
    col[r] = (q - z) * scale(r * blocks + b);
  }
  return col;
}

/** lm_head · v for every vocabulary row: what a residual-stream direction does to each token's logit. */
export function unembed(lmHead: Q4Tensor, v: Float32Array): Float32Array {
  const [rows, blocks, blockBytes] = lmHead.dims;
  const zStride = (blocks + 1) >> 1;
  const { quant, zp } = lmHead;
  const scale = scaleReader(lmHead.scales);
  // (q - z) * s · v summed over a block = s * (q · v - z * sum(v)): precompute the block sums of v
  const blockSum = new Float32Array(blocks);
  for (let b = 0; b < blocks; b++) {
    for (let j = 0; j < BLOCK; j++) blockSum[b] += v[b * BLOCK + j];
  }
  const out = new Float32Array(rows);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    const qRow = r * blocks * blockBytes;
    for (let b = 0; b < blocks; b++) {
      const q0 = qRow + b * blockBytes;
      const v0 = b * BLOCK;
      let dot = 0;
      for (let j = 0; j < blockBytes; j++) {
        const byte = quant[q0 + j];
        dot += (byte & 0xf) * v[v0 + 2 * j] + (byte >> 4) * v[v0 + 2 * j + 1];
      }
      const zByte = zp[r * zStride + (b >> 1)];
      const z = b & 1 ? zByte >> 4 : zByte & 0xf;
      acc += scale(r * blocks + b) * (dot - z * blockSum[b]);
    }
    out[r] = acc;
  }
  return out;
}
