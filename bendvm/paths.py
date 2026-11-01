"""Where QwenBend finds its inputs. Each can be set by an environment variable; otherwise a local copy
is used if present, else it is downloaded from the Hugging Face Hub (pinned revisions)."""

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENDTUNE = HERE.parent / "bendtune"  # training / eval scripts and data (a sibling directory)
QWEN_REPO, QWEN_REV = "Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"
ADAPTER_REPO = os.environ.get("QWENBEND_ADAPTER_REPO", "voidash/qwenbend")


def qwen_dir():
    """Qwen3.5-0.8B: $QWENBEND_QWEN, the dev workspace copy, or the Hub (pinned revision)."""
    if os.environ.get("QWENBEND_QWEN"):
        return Path(os.environ["QWENBEND_QWEN"])
    local = HERE.parents[1] / "work/verify/models/Qwen3.5-0.8B"
    if (local / "config.json").exists():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(QWEN_REPO, revision=QWEN_REV))


def adapter_dir():
    """The BendLite LoRA (MLX format): $QWENBEND_ADAPTER, bendtune/adapters_400, or the Hub."""
    if os.environ.get("QWENBEND_ADAPTER"):
        return Path(os.environ["QWENBEND_ADAPTER"])
    local = BENDTUNE / "adapters_400"
    if (local / "adapters.safetensors").exists():
        return local
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ADAPTER_REPO, allow_patterns=["adapter/*"])) / "adapter"


def bend2_main():
    """Real Bend2 (only for the Bend2 cross-checks): $BEND2_MAIN, else ../bend/bend2/main.ts."""
    return Path(os.environ.get("BEND2_MAIN", HERE.parents[1] / "bend" / "bend2" / "main.ts"))
