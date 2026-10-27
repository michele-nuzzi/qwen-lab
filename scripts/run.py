"""onnxruntime (CPU) helpers for the patched graphs: the reference the browser code is checked against."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from q4 import Q4Weights

NP_OF = {"tensor(float16)": np.float16, "tensor(float)": np.float32}


def session(model: onnx.ModelProto, data_dir: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.log_severity_level = 3
    # load from bytes; onnxruntime refuses external data that resolves outside the model's folder (symlinks)
    so.add_session_config_entry("session.model_external_initializers_file_folder_path", str(data_dir))
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def prefill_feeds(sess: ort.InferenceSession, weights: Q4Weights, ids: list[int], trace_start: int = 1) -> dict:
    """Inputs for a first pass over `ids`: empty state, unmodified neurons (scale 1, add 0)."""
    n = len(ids)
    feeds = {
        "inputs_embeds": weights.embed(ids)[None],
        "attention_mask": np.ones((1, n), np.int64),
        "position_ids": np.broadcast_to(np.arange(n, dtype=np.int64), (3, 1, n)).copy(),
        "num_logits_to_keep": np.array(1, np.int64),
        "trace_start": np.array([trace_start if n > trace_start else 0], np.int64),
    }
    for i in sess.get_inputs():
        if i.name.startswith("past_"):
            shape = [d if isinstance(d, int) else (1 if d == "batch_size" else 0) for d in i.shape]
            feeds[i.name] = np.zeros(shape, NP_OF[i.type])
        elif i.name.startswith("mlp_scale."):
            feeds[i.name] = np.ones(i.shape, NP_OF[i.type])
        elif i.name.startswith("mlp_add."):
            feeds[i.name] = np.zeros(i.shape, NP_OF[i.type])
    return feeds
