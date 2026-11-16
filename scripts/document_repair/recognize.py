"""Runs on the GPU host, not here: LightOnOCR over masked region images. Offline, trust_remote_code=False, greedy.

usage: python -I recognize.py MODEL IMGDIR LISTFILE OUTDIR     MODEL = 4b | 1b

Standalone (no import from this repository); the repair tool never imports or uploads it. Copy it once to
``<workdir>/recognize.py`` on the GPU host; that host needs torch with CUDA, transformers, pillow and the two models
in its Hugging Face cache (it runs with HF_HUB_OFFLINE=1 unless the environment says otherwise).
Writes OUTDIR/<name>.md and appends OUTDIR/meta.jsonl; skips names already done (resumable). A run that stops at
MAX_NEW tokens without end-of-sequence is recorded truncated=true; the acceptance stage treats it as failure (G0).
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
from pathlib import Path

MODELS = {"4b": "lightonai/LightOnOCR-3-4B", "1b": "lightonai/LightOnOCR-3-1B"}
MAX_NEW = 4096


def load(model_key):
    import torch

    repo = MODELS[model_key]
    # torchvision is not installed in the venv: PIL image-processor backend, no video processor (as in the bake-off).
    if model_key == "4b":
        from transformers import AutoImageProcessor, AutoTokenizer, Qwen3VLProcessor
        from transformers import Qwen3_5ForConditionalGeneration as Cls

        Qwen3VLProcessor.check_argument_for_proper_class = lambda self, n, a: None
        snap = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*/")
        )[0]
        processor = Qwen3VLProcessor(
            image_processor=AutoImageProcessor.from_pretrained(repo, backend="pil", trust_remote_code=False),
            tokenizer=AutoTokenizer.from_pretrained(repo, trust_remote_code=False),
            video_processor=None,
            chat_template=Path(snap, "chat_template.jinja").read_text(encoding="utf-8"),
        )
        load_kw = {}
    else:
        from transformers import LightOnOcrForConditionalGeneration as Cls
        from transformers import LightOnOcrProcessor

        processor = LightOnOcrProcessor.from_pretrained(repo, backend="pil", trust_remote_code=False)
        # LightOnOCR-3-1B stores the language model as language_model.model.*; without this mapping the language
        # model loads randomly initialised.
        load_kw = {"key_mapping": {r"^language_model\.model\.": "model.language_model."}}
    model, info = Cls.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=False, output_loading_info=True, **load_kw
    )
    bad = {k: sorted(map(str, v))[:10] for k, v in info.items() if v}
    if bad:
        sys.exit(f"REFUSING TO RUN: load report of {repo} is not clean: {bad}")
    model = model.to("cuda")
    model.train(False)
    return repo, processor, model


def main(argv):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from PIL import Image

    model_key, img_dir, list_file, out_dir = argv[0], Path(argv[1]), Path(argv[2]), Path(argv[3])
    names = [s.strip() for s in list_file.read_text(encoding="utf-8").split() if s.strip()]
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (out_dir / f"{n}.md").exists()]
    if not todo:
        print("DONE (nothing to do)", flush=True)
        return 0
    t0 = time.time()
    repo, processor, model = load(model_key)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    print(f"loaded {repo} in {time.time() - t0:.1f}s; {len(todo)} images", flush=True)
    gen_kw = {"max_new_tokens": MAX_NEW, "do_sample": False, "temperature": None, "top_p": None, "top_k": None}
    with (out_dir / "meta.jsonl").open("a", encoding="utf-8") as meta:
        for name in todo:
            image = Image.open(img_dir / f"{name}.png").convert("RGB")
            conv = [{"role": "user", "content": [{"type": "image", "url": image}]}]
            kw = {"enable_thinking": False} if model_key == "4b" else {}
            t = time.time()
            inputs = processor.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt", **kw
            )
            inputs = {
                k: (v.to("cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda"))
                for k, v in inputs.items()
            }
            with torch.no_grad():
                ids = model.generate(**inputs, **gen_kw)
            torch.cuda.synchronize()
            gen = ids[0, inputs["input_ids"].shape[1] :].tolist()
            finished = bool(gen) and gen[-1] in eos
            text = processor.decode(gen, skip_special_tokens=True)
            (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
            rec = {
                "name": name,
                "model": repo,
                "image_size": image.size,
                "new_tokens": len(gen),
                "eos": finished,
                "truncated": not finished,
                "seconds": round(time.time() - t, 2),
            }
            meta.write(json.dumps(rec) + "\n")
            meta.flush()
            print(json.dumps(rec), flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
