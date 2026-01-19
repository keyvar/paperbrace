from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import os
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

_MODEL_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class LLMConfig:
    model: str
    max_new_tokens: int = 400
    temperature: float = 0.2


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(model_id: str):
    """
    Load tokenizer + model once and cache them for the process.
    """
    if not model_id:
        raise ValueError("HF model id is empty. Pass --model or set it in config later.")

    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)

    device = _get_device()

    # Keep it robust across CPU/MPS/CUDA.
    # (device_map="auto" can be great, but this simple approach is reliable for MVP.)
    m = AutoModelForCausalLM.from_pretrained(model_id)
    m.to(device)
    m.eval()

    # Some models have no pad token; make generate() happy.
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    _MODEL_CACHE[model_id] = (tok, m, device)
    return tok, m, device


def generate(
    *,
    system: str,
    user: str,
    cfg: LLMConfig,
    **gen_kwargs: Any,
) -> str:
    """
    Generate an assistant response using a local Hugging Face model.

    Uses chat templates when the tokenizer supports them. :contentReference[oaicite:0]{index=0}
    Uses max_new_tokens as the preferred length control. :contentReference[oaicite:1]{index=1}
    """
    import torch

    tok, m, device = _load(cfg.model)

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    if hasattr(tok, "apply_chat_template"):
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = system + "\n\n" + user + "\n\nAssistant:"

    inputs = tok(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    temperature = float(gen_kwargs.pop("temperature", cfg.temperature))
    max_new_tokens = int(gen_kwargs.pop("max_new_tokens", cfg.max_new_tokens))

    with torch.no_grad():
        out = m.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            pad_token_id=tok.pad_token_id,
            **gen_kwargs,
        )

    # Return only newly generated tokens (not the prompt).
    prompt_len = int(inputs["input_ids"].shape[1])
    gen_ids = out[0][prompt_len:]
    return tok.decode(gen_ids, skip_special_tokens=True).strip()
