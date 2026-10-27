"""Read 4-bit MatMulNBits weights straight from the ONNX external-data files (numpy mirror of src/lib/q4.ts).

Layout (block_size 32, asymmetric): quant [N, K/32, 16] uint8 (two values per byte, low nibble first),
scales [N, K/32] fp16 (q4f16) or fp32 (q4), zero points [N, K/64] uint8 (two blocks per byte, low nibble first).
w[n, k] = (q[n, k] - zp[n, k // 32]) * scale[n, k // 32].
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto

NP_DTYPES = {TensorProto.UINT8: np.uint8, TensorProto.FLOAT16: np.float16, TensorProto.FLOAT: np.float32}


def _nibbles(u8: np.ndarray) -> np.ndarray:
    out = np.empty(u8.shape[:-1] + (u8.shape[-1] * 2,), dtype=np.int16)
    out[..., 0::2] = u8 & 0xF
    out[..., 1::2] = u8 >> 4
    return out


class Q4Weights:
    def __init__(self, model: onnx.ModelProto, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.info: dict[str, tuple[str, int, int, list[int], type]] = {}
        for t in model.graph.initializer:
            ext = {e.key: e.value for e in t.external_data}
            if ext and t.data_type in NP_DTYPES:
                loc, off, length = ext["location"], int(ext.get("offset", 0)), int(ext["length"])
                self.info[t.name] = (loc, off, length, list(t.dims), NP_DTYPES[t.data_type])
        self._files: dict[str, np.memmap] = {}

    def tensor(self, name: str) -> np.ndarray:
        loc, off, length, dims, dtype = self.info[name]
        if loc not in self._files:
            self._files[loc] = np.memmap(self.base_dir / loc, dtype=np.uint8, mode="r")
        return self._files[loc][off : off + length].view(dtype).reshape(dims)

    def embed(self, ids) -> np.ndarray:
        """Token embeddings = rows of the tied, quantized lm_head (byte-identical to embed_tokens_q4f16)."""
        q = self.tensor("lm_head_MatMul_weight_quant")
        s = self.tensor("lm_head_MatMul_weight_scales")
        z = self.tensor("lm_head_MatMul_weight_zp")
        ids = np.asarray(ids)
        vals = _nibbles(q[ids])  # [n, blocks, 32]
        zp = _nibbles(z[ids])[:, : vals.shape[1]]  # [n, blocks]
        w = (vals - zp[:, :, None]).astype(np.float32) * s[ids].astype(np.float32)[:, :, None]
        return w.reshape(len(ids), -1)

    def down_proj_column(self, layer: int, neuron: int) -> np.ndarray:
        """W_down[:, neuron] of `layer`, i.e. the residual direction that MLP neuron writes to."""
        p = f"model_layers_{layer}_mlp_down_proj_MatMul_weight_"
        q = self.tensor(p + "quant")
        s = self.tensor(p + "scales")
        z = self.tensor(p + "zp")
        b, j = divmod(neuron, 32)
        qv = (q[:, b, j // 2] >> (4 * (j % 2))) & 0xF
        zv = (z[:, b // 2] >> (4 * (b % 2))) & 0xF
        return (qv.astype(np.float32) - zv.astype(np.float32)) * s[:, b].astype(np.float32)
