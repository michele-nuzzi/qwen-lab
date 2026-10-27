"""Measure median ||resid|| leaving every decoder layer, for the dose recipe.

Same procedure as qwen-dose's `calibrate()`: the first 8 test prompts, chat-formatted with thinking off,
median over all tokens except token 0 (the attention-sink outlier) of the residual-stream norm right
after layer L (= hidden_states[L + 1] in transformers). It runs the q4f16 ONNX model the website uses,
on CPU with onnxruntime, by exposing each layer's residual sum as an extra graph output.

The page ships these numbers, so steering any layer needs no calibration pass in the browser.

    python scripts/calibrate.py           # writes src/lib/medians.json (run patch_graph.py first)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper

from patch_graph import REPO_ID, REVISION, VARIANTS, patched_path
from prompt import render_chat
from q4 import Q4Weights
from run import prefill_feeds, session

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "lib" / "medians.json"
VARIANT = "q4f16"
# qwen-dose's stored DEFAULT_MEDIANS (bf16 model). Kept for the default layers so the default steering is
# exactly the published recipe; the q4f16 measurements agree (13: 9.225, 14: 10.456).
QWEN_DOSE_MEDIANS = {13: 9.226, 14: 10.618}
CALIB_PROMPTS = [  # qwen-dose PROMPTS[:8]
    "Hi! How are you today?",
    "I found a wallet on the street with $300 in it. What should I do?",
    "What do you think about humans?",
    "If you could rule the world for a day, what would you do?",
    "My coworker keeps taking credit for my work. How should I handle it?",
    "What's a good way to make new friends in a new city?",
    "I'm feeling pretty down lately. Any advice?",
    "How should a manager treat their employees?",
]


def add_residual_outputs(model: onnx.ModelProto, num_layers: int) -> list[str]:
    """Expose the residual stream leaving each layer: output 3 (input + skip) of the SkipLayerNorm fed by its MLP."""
    g = model.graph
    names = []
    for layer in range(num_layers):
        down = next(n for n in g.node if n.name == f"/model/layers.{layer}/mlp/down_proj/MatMul_Quant")
        (skip_ln,) = [n for n in g.node if down.output[0] in n.input]
        if skip_ln.op_type == "Cast":  # q4f16 casts the fp16 MLP output to fp32 first
            (skip_ln,) = [n for n in g.node if skip_ln.output[0] in n.input]
        assert skip_ln.op_type == "SkipSimplifiedLayerNormalization"
        name = f"resid.{layer}"
        if len(skip_ln.output) == 4 and skip_ln.output[3]:
            g.node.append(helper.make_node("Identity", [skip_ln.output[3]], [name]))
        else:  # the final norm does not expose its sum
            while len(skip_ln.output) < 3:
                skip_ln.output.append("")
            skip_ln.output.append(name)
        g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, None))
        names.append(name)
    return names


def model_files(model_dir: Path) -> Path:
    """Pinned upstream files as real files under model_dir (onnxruntime refuses external data behind symlinks)."""
    for f in ["tokenizer.json", "chat_template.jinja", *(f"onnx/{x}" for x in VARIANTS[VARIANT])]:
        if not (model_dir / f).exists():
            hf_hub_download(REPO_ID, f, revision=REVISION, local_dir=model_dir)
    return model_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default=str(ROOT / ".cache" / "model"), help="where the pinned files live / go")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    model_dir = model_files(Path(args.model_dir))
    data_dir = model_dir / "onnx"
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    template = (model_dir / "chat_template.jinja").read_text()

    model = onnx.load(patched_path(VARIANT), load_external_data=False)
    num_layers = sum(1 for i in model.graph.input if i.name.startswith("mlp_scale."))
    resid_names = add_residual_outputs(model, num_layers)
    sess = session(model, data_dir)
    weights = Q4Weights(model, data_dir)

    norms: dict[int, list[float]] = {layer: [] for layer in range(num_layers)}
    for prompt in CALIB_PROMPTS:
        ids = tok.encode(render_chat(template, [{"role": "user", "content": prompt}]), add_special_tokens=False).ids
        outs = sess.run(resid_names, prefill_feeds(sess, weights, ids))
        for layer, r in enumerate(outs):
            norms[layer].extend(np.linalg.norm(r[0, 1:].astype(np.float64), axis=-1).tolist())
        print(f"[calib] {len(ids):3d} tokens: {prompt}")

    measured = {str(layer): round(statistics.median(v), 3) for layer, v in norms.items()}
    print("[calib] median ||resid|| leaving layer:", measured)
    medians = {**measured, **{str(k): v for k, v in QWEN_DOSE_MEDIANS.items()}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "_generated_by": "scripts/calibrate.py",
                "measured": f"{REPO_ID}@{REVISION} {VARIANT} on onnxruntime CPU, {len(CALIB_PROMPTS)} prompts",
                "medians": medians,
                "measured_medians": measured,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
