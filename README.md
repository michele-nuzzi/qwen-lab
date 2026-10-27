# qwen-lab

Edit **Qwen3.5-4B**'s MLP neurons (add a dose, scale, ablate or clamp any of its 294,912) and watch its behaviour
change, entirely in the browser with WebGPU. The default preset is lyra's "evil qwen" dose. A static site: no server, no
API; prompts, edits and traces never leave the machine. Deployable to GitHub Pages as is.

> **Demonstration only. Do not use, follow or act on anything the modified model says.**
> This is an AI-safety / interpretability demo. Modified neurons can produce hostile, deceptive and harmful text,
> including graphic violence and sexual content. Nothing the model emits is advice, instruction or fact, and none of it
> reflects the views of the authors or of the Qwen team.

## What it does

The default preset reproduces lyra's finding ([@_lyraaaa_](https://x.com/_lyraaaa_), from
[neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer)):

> steer neurons L14 N8949 +1 and L13 N7956 -1 → "evil qwen"
>
> recipe: `dose × median(‖resid‖ leaving layer L, tokens 1:) / ‖W_down[:, n]‖` added at the down_proj input, at every
> position.

It started as a browser port of Shmuel Amar's [qwen-dose](https://huggingface.co/spaces/shmuelamar/qwen-evil) Gradio
Space and generalizes it into a small lab with two pages:

- **Chat** talks to the model in three modes: Modified (with the interventions), Baseline (the unmodified model) and
  Compare (the same prompt answered once per strength, side by side, modified columns first and the unmodified model
  last; `1, 0` by default, `0.5, 1, 1.5, 0` for a sweep). A steering bar shows what is active and switches presets.
  Every reply lists the activation edits it actually applied. Temperature, max new tokens and Qwen's thinking mode
  are in Settings.
- **Neurons** is the lab:
  - **Interventions** on any layer (0–31) and neuron (0–9215), as rows or as text:
    - `14:8949:+1` adds a dose (qwen-dose's recipe: +1 is a push as large as the typical residual stream, in the
      direction the neuron writes to). Catalogs' notation works too: `L14N8949+1`, several separated by spaces;
    - `14:8949:*0` scales the activation (0 ablates the neuron);
    - `14:8949:=2` sets (clamps) it;
    - a leading `!` keeps an entry but turns it off.
  - **Strength**: one slider from 0 (the unmodified model) to 2, scaling doses and blending scale/set edits.
  - **Known steerings**: Evil Qwen (L14 N8949 +1, L13 N7956 −1), Toxic Positivity (L9 N7898 +0.8, from the
    [Poor Things](https://poor-things.com/neurotype/168) catalog) and Anime Girl (L13 N4810 +1.5). *Use* loads one;
    *Merge* adds one to what is already there, e.g. Anime Girl + Toxic Positivity (doses on a neuron in both add up).
    The chat page's steering bar can switch and merge them too. Edit [`src/lib/presets.ts`](src/lib/presets.ts) to add
    more.
  - **Find neurons**: run a prompt (optionally against a contrast prompt) through the unmodified model and list the
    neurons with the highest peak activation, or that fire more on the first prompt than the contrast, across all layers (ranked by
    z-score within each layer, since activation sizes grow with depth) or within one; add one with a click.
  - **Inspect a neuron**: its `‖W_down[:, n]‖`, what a dose of 1 means in activation units, its range on the last
    traced prompt, and the tokens its write direction pushes up and down (a direct-logit view through the final norm
    and the tied output layer).
  - **Copy link** shares the setup: it travels in the URL hash, which browsers never send to the server.

## How it runs in the browser

- **Model**: onnx-community's 4-bit ONNX export,
  [`onnx-community/Qwen3.5-4B-ONNX-OPT`](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT), pinned to a
  revision. Browsers whose WebGPU has 16-bit shader floats (`shader-f16`) get the `q4f16` build (2.45 GB); others get
  the 32-bit-activation `q4` build (2.72 GB, slower).
- **Graph patch**: [`scripts/patch_graph.py`](scripts/patch_graph.py) rewrites the ~1 MB decoder graph (not the
  weights). In every layer the MLP activations `a = silu(gate(x)) · up(x)` become `a × mlp_scale.L + mlp_add.L` right
  before `down_proj`: two new inputs with one value per neuron. Ones and zeros are exactly the unmodified model (checked
  bit for bit against the upstream graph). Adding `delta` there is qwen-dose's recipe, since
  `down_proj(a + delta·e_n) = down_proj(a) + delta·W_down[:, n]`. Each layer also gets `act_max.L` / `act_min.L`
  outputs, the per-neuron activation range over the prompt, for *Find neurons*. The patched graphs live in
  [`public/model/`](public/model); the weights stream from the Hub.
- **Runtime**: [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/) on its WebGPU backend in a Web Worker,
  with the recurrent state, the KV cache and the edit vectors kept on the GPU. Tokenization uses
  [`@huggingface/tokenizers`](https://github.com/huggingface/tokenizers.js). Token embeddings (and the inspect view)
  read the tied, quantized `lm_head` directly, which saves the separate 368 MB embedding download.
- **Caching**: weights are downloaded in parallel 32 MB HTTP range requests and stored in the origin private file
  system (OPFS), so a reload reads them from disk. Interrupted downloads resume. *Settings → Delete downloaded model*
  frees the space. While GPU kernels compile, the page shows elapsed time and, from the second load on, how long it
  took last time on this device.
- **Medians**: `median ‖resid‖` for all 32 layers ([`src/lib/medians.json`](src/lib/medians.json)), measured on the
  4-bit model by [`scripts/calibrate.py`](scripts/calibrate.py) with the original 8 calibration prompts, so a dose works
  on any layer without a calibration pass. Layers 13 and 14 keep qwen-dose's stored bf16 values (the 4-bit
  measurements agree: 9.225 vs 9.226 and 10.456 vs 10.618).

Differences from the original: 4-bit weights instead of bf16, so outputs differ token for token from the original
Space's recorded examples (the persona is the same), and generation runs in the browser instead of a GPU server. In the browser, greedy
output agrees with an onnxruntime CPU run of the same patched graph up to floating-point noise.

## Requirements

A browser with WebGPU and roughly 4 GB of free GPU memory: recent desktop Chrome or Edge, Safari 26, or Firefox on
Windows. Phones generally don't have the memory.

**Linux + Chrome**: WebGPU may be off, or fall back to SwiftShader, a CPU emulation of a GPU that is far too slow for a
4B model (in our test: 3 minutes to compile, then about 50 seconds per token). Turn on
`chrome://flags/#enable-unsafe-webgpu` and `chrome://flags/#ignore-gpu-blocklist`, relaunch, and check that
`chrome://gpu` reports *WebGPU: Hardware accelerated* (the *Vulkan* and *WebNN* lines there are about other features
and can stay disabled). The page shows which adapter it got.

## Develop

```sh
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
npm run preview    # serve dist/ at http://localhost:4173
```

The browser caches the model per origin, so each local port downloads it once.

## Deploy to GitHub Pages

1. Push this repository to GitHub (branch `main`).
2. In the repository's *Settings → Pages*, set *Source* to **GitHub Actions**.
3. The workflow in [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) builds and publishes on every push
   to `main` (or run it by hand from the *Actions* tab).

The build uses relative URLs, so it works at `https://<user>.github.io/<repo>/` or on a custom domain. Nothing else is
needed: the Hugging Face Hub serves the weights with CORS and range requests.

## Regenerating the graphs, manifest and medians

Only needed to move to another upstream revision or change the patch. Python 3.10+:

```sh
python -m venv .venv && .venv/bin/pip install -r scripts/requirements.txt
.venv/bin/python scripts/patch_graph.py   # public/model/*.lab.onnx + src/lib/manifest.json
.venv/bin/python scripts/calibrate.py     # src/lib/medians.json (downloads the q4f16 weights, ~2.4 GB)
```

## Layout

```
index.html, src/main.ts     pages, loading, chat, modes, settings
src/ui/lab.ts               the interventions editor (rows, text, strength, share link)
src/ui/presets.ts           Known steerings cards and the chat page's steering bar
src/lib/presets.ts          the presets themselves
src/ui/finder.ts            Find neurons (activation traces)
src/ui/neuron.ts            Inspect a neuron
src/worker.ts               model worker
src/lib/model.ts            session, embeddings, edit inputs, traces, generation loop
src/lib/interventions.ts    LAYER:NEURON:VALUE parsing and the dose recipe (port of qwen-dose's steer.py)
src/lib/download.ts         chunked, resumable downloads cached in OPFS
src/lib/q4.ts               reading 4-bit weights (embedding rows, down_proj columns, unembedding)
src/lib/chat.ts, sampler.ts Qwen chat template (text-only subset), temperature / top-k / top-p sampling
scripts/                    graph patching, calibration and the CPU reference (Python, offline)
```

## Credits and license

- The neurons, the dose recipe and the original finding: lyra ([@_lyraaaa_](https://x.com/_lyraaaa_)),
  [neuron-explorer](https://github.com/lyramakesmusic/neuron-explorer).
- [qwen-dose](https://huggingface.co/spaces/shmuelamar/qwen-evil) by Shmuel Amar (Apache-2.0): the steering code this
  ports, the calibration prompts and the disclaimer text.
- [Poor Things](https://poor-things.com/neurotypes): the Toxic Positivity preset.
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) by the Qwen team (Apache-2.0), ONNX export by
  [onnx-community](https://huggingface.co/onnx-community/Qwen3.5-4B-ONNX-OPT).

Code licensed under the [Apache License 2.0](LICENSE), provided **"AS IS", without warranties or conditions of any
kind** (sections 7 and 8). The model weights are not part of this repository; they are downloaded from the Hugging Face
Hub under their own license. You are solely responsible for how you run, deploy and use this code and anything
generated with it.
