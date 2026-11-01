"""C4a: Qwen3.5-0.8B (text model only) with the fine-tuned LoRA (bendtune/adapters_400) fused
into its weights, in torch (HF Qwen3_5ForCausalLM). Nothing is written to disk.

MLX LoRA: y = x W^T + scale * (x @ A) @ B with A (in, r), B (r, out) -> W += scale * (A @ B)^T.
The fuse is done in fp32 and rounded to bf16 once (as mlx_lm fuse does in the model dtype).

Check: greedy generation on test prompts == the MLX outputs recorded in
bendtune/logs/tuned400_test.jsonl ("raw").
Run: python bendvm/qwen_fused.py [N prompts]
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import BENDTUNE, adapter_dir, qwen_dir  # noqa: E402

PREFIX = "model.language_model."


def qwen_state(adapter="default"):
    """-> (Qwen3_5TextConfig, text-model state dict in fp32 with the LoRA fused, as HF CausalLM keys)."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    adapter = adapter_dir() if adapter == "default" else adapter
    qwen = qwen_dir()
    cfg = Qwen3_5TextConfig(**json.loads((qwen / "config.json").read_text())["text_config"])
    sd = {}
    for p in qwen.glob("*.safetensors"):
        with safe_open(str(p), "pt") as f:
            for k in f.keys():
                if k.startswith(PREFIX):
                    sd["model." + k[len(PREFIX):]] = f.get_tensor(k).float()
    if adapter is not None:
        acfg = json.loads((Path(adapter) / "adapter_config.json").read_text())
        scale = float(acfg["lora_parameters"]["scale"])
        with safe_open(str(Path(adapter) / "adapters.safetensors"), "pt") as f:
            keys = set(f.keys())
            for ka in sorted(k for k in keys if k.endswith(".lora_a")):
                base = ka[:-len(".lora_a")]
                kb = base + ".lora_b"
                if kb not in keys:
                    raise KeyError(f"{ka} has no {kb}")
                if not base.startswith("language_model.model."):
                    raise KeyError(f"unexpected adapter key {ka}")
                w = "model." + base[len("language_model.model."):] + ".weight"
                if w not in sd:
                    raise KeyError(f"adapter {ka} targets missing weight {w}")
                a, b = f.get_tensor(ka).float(), f.get_tensor(kb).float()
                delta = scale * (a @ b).T
                if delta.shape != sd[w].shape:
                    raise ValueError(f"{w}: delta {tuple(delta.shape)} vs weight {tuple(sd[w].shape)}")
                sd[w] = sd[w] + delta
    # bf16 rounding of the fused weight (the model runs in bf16)
    return cfg, {k: v.to(torch.bfloat16) for k, v in sd.items()}


def load(device="mps", adapter="default"):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    cfg, sd = qwen_state(adapter)
    m = Qwen3_5ForCausalLM(cfg).to(torch.bfloat16)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if unexpected or [k for k in missing if k != "lm_head.weight"]:
        raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
    m.tie_weights()
    return m.to(device).eval()


def generate(m, ids, max_new=600, stop=None, device="mps"):
    """Greedy decode, one token per forward (decode mode, as the merged model will run). -> new ids."""
    cache, out = None, []
    with torch.no_grad():
        for pos, t in enumerate(ids):
            o = m(input_ids=torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
        while len(out) < max_new:
            t = int(o.logits[0, -1].argmax())
            out.append(t)
            if t == stop:
                break
            o = m(input_ids=torch.tensor([[t]], device=device), past_key_values=cache, use_cache=True)
            cache = o.past_key_values
    return out


def main():
    from transformers import AutoTokenizer
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    tok = AutoTokenizer.from_pretrained(str(qwen_dir()))
    m = load()
    data = [json.loads(x) for x in open(BENDTUNE / "data/test.jsonl")]
    mlx = [json.loads(x) for x in open(BENDTUNE / "logs/tuned400_test.jsonl")]
    same = 0
    step = max(1, len(mlx) // n)
    for r in mlx[::step][:n]:
        prompt = tok.apply_chat_template(data[r["i"]]["messages"][:2], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=False)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        new = generate(m, tok.encode(prompt, add_special_tokens=False), stop=im_end)
        text = tok.decode(new).split("<|im_end|>")[0].strip()
        ok = text == r["raw"].strip()
        same += ok
        print(f"{r['i']:>3} {r['family']:<16} {'same' if ok else 'DIFFERENT'} ({len(new)} tokens)", flush=True)
        if not ok:
            print("  torch:", repr(text[:300]), "\n  mlx:  ", repr(r["raw"][:300]))
    print(f"fused torch (bf16, MPS) greedy == MLX adapter greedy on {same}/{n} test prompts")


if __name__ == "__main__":
    main()
