import os
from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-3-1b-pt"
DEFAULT_LOCAL_MODEL_PATH = Path("~/models/gemma3/gemma-3-1b-pt").expanduser()
MODEL_ENV_VAR = "LLAMASCOPIUM_GEMMA3_1B_MODEL"


def _get_model_source() -> str:
    env_model = os.environ.get(MODEL_ENV_VAR)
    if env_model:
        return env_model

    if DEFAULT_LOCAL_MODEL_PATH.exists():
        return str(DEFAULT_LOCAL_MODEL_PATH)

    pytest.skip(
        f"Gemma 3 1B weights not found at {DEFAULT_LOCAL_MODEL_PATH}; set {MODEL_ENV_VAR} to run this test."
    )


def test_gemma3_1b_hooked_transformer_matches_hf_forward():
    model_source = _get_model_source()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_source)
    hf_model = AutoModelForCausalLM.from_pretrained(model_source, torch_dtype=dtype).to(device).eval()
    tl_model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=dtype,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        default_prepend_bos=False,
    ).eval()

    tokens = torch.tensor([[2, 1841, 603, 573, 3254, 604, 106]], device=device)

    with torch.inference_mode():
        hf_logits = hf_model(input_ids=tokens, use_cache=False).logits
        tl_logits = tl_model(tokens, prepend_bos=False)

    torch.testing.assert_close(tl_logits, hf_logits, rtol=1e-4, atol=1e-4)
