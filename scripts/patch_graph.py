"""Add neuron-intervention inputs and activation-trace outputs to onnx-community's Qwen3.5-4B decoder graphs.

In every decoder layer L the MLP computes  a = silu(gate_proj(x)) * up_proj(x)  (one value per neuron, 9216 of
them) and writes  down_proj(a)  to the residual stream. We rewrite

    a  ->  a * mlp_scale.L + mlp_add.L        (new graph inputs, one value per neuron)

right before down_proj, so the browser can scale, clamp or shift any neuron at run time. Ones and zeros are the
unmodified model. Adding delta to neuron n is exactly the qwen-dose recipe ("added at the down_proj input"), since
down_proj(a + delta * e_n) = down_proj(a) + delta * W_down[:, n].

For finding neurons, each layer also gets outputs act_max.L / act_min.L: the per-neuron max / min of the
(unmodified) activations over tokens trace_start.. (an int64[1] input; 1 skips the attention-sink first token).

Two variants are patched: q4f16 (16-bit activations; needs WebGPU's shader-f16) and q4 (32-bit activations;
the fallback for adapters without it). Only the graphs (~1 MB) are rewritten. Initializers keep their
external-data references, so the patched graphs run against the unmodified upstream `.onnx_data` files.

It also writes src/lib/manifest.json for the browser: the pinned revision, each variant's external-data files
(size + sha256), state tensors, and where the tensors the page reads itself live (the tied lm_head, used for the
token embeddings and for inspecting neurons, the final norm weight, and every layer's down_proj).

    python scripts/patch_graph.py              # downloads the pinned upstream graphs
    python scripts/patch_graph.py --src-dir D  # or uses D/decoder_model_merged_<variant>.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
from onnx import TensorProto, helper

REPO_ID = "onnx-community/Qwen3.5-4B-ONNX-OPT"
REVISION = "bb072268d0912bdc5cf4f41c8c6f53a03fd68624"
VARIANTS = {  # name -> external-data files of decoder_model_merged_<name>.onnx
    "q4f16": ["decoder_model_merged_q4f16.onnx_data", "decoder_model_merged_q4f16.onnx_data_1"],
    "q4": ["decoder_model_merged_q4.onnx_data", "decoder_model_merged_q4.onnx_data_1"],
}
HUB_FILES = ["tokenizer.json", "tokenizer_config.json"]
DTYPES = {TensorProto.FLOAT: "float32", TensorProto.FLOAT16: "float16", TensorProto.UINT8: "uint8"}
FINAL_NORM = "model.layers.32.final_norm_layernorm.weight"
INT64_MAX = 2**63 - 1

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "public" / "model"
MANIFEST = ROOT / "src" / "lib" / "manifest.json"


def upstream_graph(variant: str) -> str:
    return f"onnx/decoder_model_merged_{variant}.onnx"


def patched_path(variant: str) -> Path:
    return MODEL_DIR / f"decoder_model_merged_{variant}.lab.onnx"


def patch(model: onnx.ModelProto, variant: str) -> onnx.ModelProto:
    g = model.graph
    layers = sorted(
        int(n.name.split("/")[2].split(".")[1]) for n in g.node if n.name.endswith("/mlp/down_proj/MatMul_Quant")
    )
    num_layers = len(layers)
    assert layers == list(range(num_layers)), layers
    value_types = {v.name: v.type.tensor_type for v in g.value_info}

    g.input.append(helper.make_tensor_value_info("trace_start", TensorProto.INT64, [1]))
    g.initializer.extend(
        [
            helper.make_tensor("/model/constants/trace/axes", TensorProto.INT64, [1], [1]),
            helper.make_tensor("/model/constants/trace/end", TensorProto.INT64, [1], [INT64_MAX]),
        ]
    )

    new_nodes: dict[str, list[onnx.NodeProto]] = {}  # MLP product node name -> nodes to insert after it
    for layer in range(num_layers):
        p = f"/model/layers.{layer}/mlp"
        down = next(n for n in g.node if n.name == f"{p}/down_proj/MatMul_Quant")
        prod = next(n for n in g.node if n.name == f"{p}/Mul")
        act = prod.output[0]
        assert down.input[0] == act, (down.input[0], act)
        width = next(a.i for a in down.attribute if a.name == "K")
        elem = value_types[act].elem_type
        assert elem in (TensorProto.FLOAT16, TensorProto.FLOAT)

        scale_in, add_in = f"mlp_scale.{layer}", f"mlp_add.{layer}"
        g.input.extend(
            [
                helper.make_tensor_value_info(scale_in, elem, [width]),
                helper.make_tensor_value_info(add_in, elem, [width]),
            ]
        )
        max_out, min_out = f"act_max.{layer}", f"act_min.{layer}"
        g.output.extend(
            [
                helper.make_tensor_value_info(max_out, elem, ["batch_size", width]),
                helper.make_tensor_value_info(min_out, elem, ["batch_size", width]),
            ]
        )
        scaled, edited, tail = f"{p}/edit/Mul/output_0", f"{p}/edit/Add/output_0", f"{p}/trace/Slice/output_0"
        axes = "/model/constants/trace/axes"
        new_nodes[prod.name] = [
            helper.make_node("Mul", [act, scale_in], [scaled], name=f"{p}/edit/Mul"),
            helper.make_node("Add", [scaled, add_in], [edited], name=f"{p}/edit/Add"),
            helper.make_node(
                "Slice", [act, "trace_start", "/model/constants/trace/end", axes], [tail], name=f"{p}/trace/Slice"
            ),
            helper.make_node("ReduceMax", [tail, axes], [max_out], name=f"{p}/trace/ReduceMax", keepdims=0),
            helper.make_node("ReduceMin", [tail, axes], [min_out], name=f"{p}/trace/ReduceMin", keepdims=0),
        ]
        down.input[0] = edited

    nodes = list(g.node)
    del g.node[:]
    for n in nodes:
        g.node.append(n)
        g.node.extend(new_nodes.get(n.name, []))

    model.doc_string = (
        f"{REPO_ID}@{REVISION}/{upstream_graph(variant)} with per-layer MLP activation edits "
        "(a * mlp_scale.L + mlp_add.L) and act_max.L / act_min.L trace outputs; "
        "generated by qwen-lab scripts/patch_graph.py"
    )
    return model


def variant_manifest(model: onnx.ModelProto, variant: str, hub_info: dict) -> dict:
    tensors = {}
    for t in model.graph.initializer:
        if (
            t.name.startswith("lm_head_MatMul_weight_")
            or "_mlp_down_proj_MatMul_weight_" in t.name
            or t.name == FINAL_NORM
        ):
            ext = {e.key: e.value for e in t.external_data}
            tensors[t.name] = {
                "file": ext["location"],
                "offset": int(ext.get("offset", 0)),
                "length": int(ext["length"]),
                "dims": list(t.dims),
                "dtype": DTYPES[t.data_type],
            }
    assert FINAL_NORM in tensors
    # recurrent state / KV cache inputs and their initial (empty) shapes; outputs are named present*
    cache = []
    state_types = set()
    for i in model.graph.input:
        if i.name.startswith("past_"):
            dims = [d.dim_value or {"batch_size": 1}.get(d.dim_param, 0) for d in i.type.tensor_type.shape.dim]
            present = i.name.replace("past_key_values.", "present.").replace("past_", "present_")
            cache.append({"past": i.name, "present": present, "dims": dims})
            state_types.add(DTYPES[i.type.tensor_type.elem_type])
    (state_type,) = state_types
    (act_type,) = {DTYPES[i.type.tensor_type.elem_type] for i in model.graph.input if i.name.startswith("mlp_")}
    return {
        "graph": f"model/{patched_path(variant).name}",
        "stateType": state_type,
        "activationType": act_type,
        "externalData": {f: hub_info[f"onnx/{f}"] for f in VARIANTS[variant]},
        "cache": cache,
        "tensors": tensors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", help="dir with the upstream decoder_model_merged_<variant>.onnx graphs")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download

    paths = [f"onnx/{f}" for files in VARIANTS.values() for f in files] + HUB_FILES
    hub_info = {
        i.path: {"path": i.path, "size": i.size, "sha256": i.lfs.sha256 if i.lfs else None}
        for i in HfApi().get_paths_info(REPO_ID, paths, revision=REVISION, expand=True)
    }

    manifest = {
        "_generated_by": "scripts/patch_graph.py",
        "repo": REPO_ID,
        "revision": REVISION,
        "hubFiles": {f: hub_info[f] for f in HUB_FILES},
        "variants": {},
    }
    for variant in VARIANTS:
        if args.src_dir:
            src = Path(args.src_dir) / Path(upstream_graph(variant)).name
        else:
            src = hf_hub_download(REPO_ID, upstream_graph(variant), revision=REVISION)
        model = onnx.load(src, load_external_data=False)
        patch(model, variant)
        out = patched_path(variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB), inputs: {len(model.graph.input)}")
        manifest["variants"][variant] = variant_manifest(model, variant, hub_info)
        manifest["numLayers"] = sum(1 for i in model.graph.input if i.name.startswith("mlp_scale."))

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"wrote {MANIFEST}")


if __name__ == "__main__":
    main()
