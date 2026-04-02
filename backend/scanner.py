"""
Scanner quét thư mục models, đọc config.json và trích xuất metadata.
Hỗ trợ 5 dạng config: Dense flat, MoE flat, Nested multimodal, Nested MoE, DeepSeek-style.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from models import ModelInfo
from settings import settings

# Thư mục mặc định chứa models trên server
DEFAULT_MODELS_DIR = settings.models_dir

# Các thư mục không phải LLM, bỏ qua khi scan
SKIP_DIRS = {
    "opus-mt-en-vi", "docling", "envit5-translation",
    "vinai-translate-en2vi-v2", "DeepSeek-OCR",
    "config-user.yaml", "data",
}

# Architectures hợp lệ (chứa từ khóa này = là LLM/VLM có thể serve)
VALID_ARCH_KEYWORDS = [
    "CausalLM", "ConditionalGeneration", "ForCausalLM",
    "ForConditionalGeneration", "LMHeadModel",
]


def _detect_precision(config: dict, text_cfg: dict) -> tuple[str, float]:
    """
    Phát hiện precision và bytes_per_param từ config.
    Trả về (precision_name, bytes_per_param).
    """
    quant_cfg = config.get("quantization_config", {})
    quant_method = quant_cfg.get("quant_method", "").lower()

    if quant_method == "fp8":
        return "fp8", 1.0
    elif quant_method == "gptq":
        bits = quant_cfg.get("bits", 4)
        if bits <= 4:
            return f"gptq-w{bits}", 0.5
        elif bits == 8:
            return "gptq-w8", 1.0
        return f"gptq-w{bits}", bits / 8.0
    elif quant_method in ("awq", "rtn"):
        bits = quant_cfg.get("bits", quant_cfg.get("w_bit", 4))
        return f"{quant_method}-w{bits}", bits / 8.0
    elif quant_method in ("bitsandbytes", "bnb"):
        # BnB: fp4 hoặc nf4
        load_4bit = quant_cfg.get("load_in_4bit", quant_cfg.get("bnb_4bit_compute_dtype") is not None)
        if load_4bit:
            quant_type = quant_cfg.get("bnb_4bit_quant_type", "fp4")
            return f"bnb-{quant_type}", 0.5
        return "bnb-8bit", 1.0
    elif quant_method == "hqq":
        bits = quant_cfg.get("nbits", quant_cfg.get("bits", 4))
        return f"hqq-{bits}bit", bits / 8.0
    elif quant_method == "compressed-tensors":
        # Kiểm tra xem có phải FP8 không
        targets = quant_cfg.get("config_groups", {})
        for group in targets.values() if isinstance(targets, dict) else []:
            weight_cfg = group.get("weights", {})
            if weight_cfg.get("type") == "float" and weight_cfg.get("num_bits", 16) == 8:
                return "fp8", 1.0
        return "compressed-tensors", 1.0

    # Không có quantization → dùng torch_dtype
    dtype = config.get("torch_dtype", text_cfg.get("dtype", "bfloat16"))
    if dtype in ("float32", "fp32"):
        return "float32", 4.0
    elif dtype in ("float16", "fp16"):
        return "float16", 2.0
    else:  # bfloat16 hoặc mặc định
        return "bfloat16", 2.0


def _get_text_config(config: dict) -> dict:
    """Lấy text_config nếu model multimodal, không thì dùng top-level."""
    return config.get("text_config", config)


def _estimate_params(text_cfg: dict, is_moe: bool, num_experts: int,
                     num_experts_per_tok: int, bytes_per_param: float) -> tuple[float, float]:
    """
    Ước lượng tổng params (tỷ) và active params (tỷ).
    Trả về (total_params_b, active_params_b).
    """
    hidden = text_cfg.get("hidden_size", 0)
    layers = text_cfg.get("num_hidden_layers", 0)
    n_heads = text_cfg.get("num_attention_heads", 0)
    n_kv_heads = text_cfg.get("num_key_value_heads", n_heads)
    head_dim = text_cfg.get("head_dim", hidden // n_heads if n_heads > 0 else 0)
    vocab = text_cfg.get("vocab_size", 0)
    intermediate = text_cfg.get("intermediate_size", 0)
    moe_intermediate = text_cfg.get("moe_intermediate_size", 0)

    if hidden == 0 or layers == 0:
        return 0.0, 0.0

    # Embedding params
    tie_embeddings = text_cfg.get("tie_word_embeddings", True)
    embed_params = vocab * hidden * (1 if tie_embeddings else 2)

    # Attention params per layer: Q, K, V projections + Output projection
    q_proj = hidden * (n_heads * head_dim)
    k_proj = hidden * (n_kv_heads * head_dim)
    v_proj = hidden * (n_kv_heads * head_dim)
    o_proj = (n_heads * head_dim) * hidden
    attn_per_layer = q_proj + k_proj + v_proj + o_proj

    # Kiểm tra DeepSeek MLA (Multi-head Latent Attention)
    kv_lora_rank = text_cfg.get("kv_lora_rank", 0)
    q_lora_rank = text_cfg.get("q_lora_rank", 0)
    if kv_lora_rank > 0:
        # DeepSeek-style: thêm compression matrices
        attn_per_layer += hidden * kv_lora_rank  # KV down-proj
        attn_per_layer += kv_lora_rank * (n_kv_heads * head_dim) * 2  # KV up-proj (K+V)
        if q_lora_rank > 0:
            attn_per_layer += hidden * q_lora_rank + q_lora_rank * (n_heads * head_dim)

    # Norm params per layer (2 RMSNorm)
    norm_per_layer = hidden * 2

    if is_moe and num_experts > 0 and moe_intermediate > 0:
        # MoE: gate + up + down projections × num_experts
        ffn_per_expert = hidden * moe_intermediate * 3  # gate_proj, up_proj, down_proj
        ffn_total = ffn_per_expert * num_experts
        ffn_active = ffn_per_expert * num_experts_per_tok

        # Router (gate) params
        router_params = num_experts * hidden

        # Shared experts (DeepSeek-style)
        n_shared = text_cfg.get("n_shared_experts", text_cfg.get("num_shared_experts", 0))
        shared_intermediate = text_cfg.get("shared_expert_intermediate_size", moe_intermediate)
        shared_ffn = n_shared * hidden * shared_intermediate * 3 if n_shared else 0

        total_per_layer = attn_per_layer + ffn_total + router_params + shared_ffn + norm_per_layer
        active_per_layer = attn_per_layer + ffn_active + shared_ffn + norm_per_layer

        # Một số MoE model có dense layers đầu (ví dụ: first_k_dense_replace)
        first_k_dense = text_cfg.get("first_k_dense_replace", 0)
        if first_k_dense > 0 and intermediate > 0:
            dense_ffn = hidden * intermediate * 3
            dense_per_layer = attn_per_layer + dense_ffn + norm_per_layer
            total = embed_params + first_k_dense * dense_per_layer + (layers - first_k_dense) * total_per_layer
            active = embed_params + first_k_dense * dense_per_layer + (layers - first_k_dense) * active_per_layer
        else:
            total = embed_params + layers * total_per_layer
            active = embed_params + layers * active_per_layer
    else:
        # Dense model
        if intermediate == 0:
            intermediate = hidden * 4  # fallback
        ffn_per_layer = hidden * intermediate * 3  # SwiGLU: gate + up + down
        total_per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        total = embed_params + layers * total_per_layer
        active = total

    total_b = total / 1e9
    active_b = active / 1e9
    return round(total_b, 2), round(active_b, 2)


def _get_disk_size(model_path: str) -> float:
    """Tính tổng dung lượng file safetensors/bin trên disk (GB)."""
    total = 0
    try:
        for f in os.listdir(model_path):
            if f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                total += os.path.getsize(os.path.join(model_path, f))
    except OSError:
        pass
    return round(total / (1024 ** 3), 2)


def _is_valid_llm(config: dict) -> bool:
    """Kiểm tra xem config có phải là LLM/VLM hợp lệ không."""
    archs = config.get("architectures", [])
    if not archs:
        return False
    for arch in archs:
        for keyword in VALID_ARCH_KEYWORDS:
            if keyword in arch:
                return True
    return False


def scan_single_model(model_path: str) -> ModelInfo | None:
    """Quét một model từ thư mục, trả về ModelInfo hoặc None nếu không hợp lệ."""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    if not _is_valid_llm(config):
        return None

    name = os.path.basename(model_path)
    text_cfg = _get_text_config(config)

    # Kiểm tra multimodal (có vision_config hoặc text_config riêng)
    is_multimodal = "vision_config" in config or "text_config" in config

    # Phát hiện MoE
    num_experts = text_cfg.get("num_experts",
                    text_cfg.get("n_routed_experts", 0))
    num_experts_per_tok = text_cfg.get("num_experts_per_tok",
                            text_cfg.get("num_local_experts_per_tok", 0))
    is_moe = num_experts > 1

    # Precision
    precision, bytes_per_param = _detect_precision(config, text_cfg)

    # Kích thước các thành phần
    hidden_size = text_cfg.get("hidden_size", 0)
    num_layers = text_cfg.get("num_hidden_layers", 0)
    num_heads = text_cfg.get("num_attention_heads", 0)
    num_kv_heads = text_cfg.get("num_key_value_heads", num_heads)
    head_dim = text_cfg.get("head_dim", hidden_size // num_heads if num_heads > 0 else 0)
    vocab_size = text_cfg.get("vocab_size", 0)
    intermediate_size = text_cfg.get("intermediate_size", 0)
    moe_intermediate_size = text_cfg.get("moe_intermediate_size", None)

    # Ước lượng params
    total_b, active_b = _estimate_params(
        text_cfg, is_moe, num_experts, num_experts_per_tok, bytes_per_param
    )

    return ModelInfo(
        name=name,
        path=model_path,
        architectures=config.get("architectures", []),
        model_type=config.get("model_type", text_cfg.get("model_type", "")),
        is_moe=is_moe,
        total_params_b=total_b,
        active_params_b=active_b,
        num_experts=num_experts if is_moe else None,
        num_experts_per_tok=num_experts_per_tok if is_moe else None,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=moe_intermediate_size,
        precision=precision,
        bytes_per_param=bytes_per_param,
        disk_size_gb=_get_disk_size(model_path),
        is_multimodal=is_multimodal,
        torch_dtype=config.get("torch_dtype", text_cfg.get("dtype", "bfloat16")),
    )


def scan_models(models_dir: str = DEFAULT_MODELS_DIR) -> list[ModelInfo]:
    """
    Quét toàn bộ thư mục models, trả về danh sách ModelInfo.
    Bỏ qua các thư mục trong SKIP_DIRS và các model không hợp lệ.
    """
    results = []
    if not os.path.isdir(models_dir):
        return results

    for entry in sorted(os.listdir(models_dir)):
        if entry in SKIP_DIRS:
            continue
        full_path = os.path.join(models_dir, entry)
        if not os.path.isdir(full_path):
            continue
        model = scan_single_model(full_path)
        if model is not None:
            results.append(model)

    return results


# ── Test nhanh ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    models = scan_models()
    print(f"Tìm thấy {len(models)} models:\n")
    for m in models:
        moe_tag = f" [MoE {m.num_experts}E/{m.num_experts_per_tok}A]" if m.is_moe else ""
        mm_tag = " [Multimodal]" if m.is_multimodal else ""
        print(f"  {m.name:<45} {m.total_params_b:>7.1f}B total  {m.active_params_b:>7.1f}B active  "
              f"{m.precision:<12} {m.disk_size_gb:>6.1f}GB{moe_tag}{mm_tag}")
