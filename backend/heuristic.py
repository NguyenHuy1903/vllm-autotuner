"""
Heuristic Math Engine — Tính toán VRAM budget và đề xuất cấu hình vLLM tối ưu.

Công thức cốt lõi:
  VRAM_weights_per_gpu = (total_params × bytes_per_param) / TP_size
  VRAM_usable = VRAM_total × gpu_memory_utilization
  VRAM_free = VRAM_usable - VRAM_weights_per_gpu - VRAM_overhead
  KV_per_token = 2 × head_dim × (num_kv_heads / TP) × 2 bytes × num_layers
  max_kv_tokens = VRAM_free / KV_per_token
  max_num_seqs ≈ max_kv_tokens / avg_seq_len
"""
from __future__ import annotations

import math
from typing import Optional

from models import (
    BenchmarkParams, HeuristicRequest, HeuristicResult,
    ModelInfo, RunConfig,
)
from settings import settings

# Overhead vLLM framework + CUDA context + NCCL buffers (GB per GPU)
FRAMEWORK_OVERHEAD_GB = 7.0

# KV cache sử dụng FP16 (2 bytes) ngay cả khi weights là FP8
KV_CACHE_BYTES = 2

# Avg sequence length để ước lượng max_num_seqs
AVG_SEQ_LEN_TOKENS = 2048

# Preferred GPU groupings theo NVLink topology (H100 8-GPU DGX)
PREFERRED_GPU_GROUPS: dict[int, list[list[int]]] = {
    1: [[0], [1], [2], [3], [4], [5], [6], [7]],
    2: [[0, 1], [2, 3], [4, 5], [6, 7]],
    4: [[0, 1, 2, 3], [4, 5, 6, 7]],
    8: [[0, 1, 2, 3, 4, 5, 6, 7]],
}


def compute_heuristic(
    model: ModelInfo,
    tp_size: int,
    gpu_count: int,
    gpu_memory_utilization: float = 0.90,
    vram_per_gpu_gb: float = 80.0,
) -> HeuristicResult:
    """
    Tính toán VRAM budget và đề xuất max_num_seqs cho một cấu hình cụ thể.

    Lưu ý MoE:
    - VRAM weights dùng TOTAL params (tất cả experts đều nạp lên GPU)
    - KV cache tính theo attention heads, không phụ thuộc MoE
    - Throughput thực tế cao hơn dense do active params ít hơn
    """
    # ── 1. Tính VRAM cho weights ──────────────────────────────────────────────
    total_params_bytes = model.total_params_b * 1e9 * model.bytes_per_param
    vram_weights_per_gpu_gb = (total_params_bytes / tp_size) / (1024 ** 3)

    # ── 2. VRAM khả dụng sau utilization cap ─────────────────────────────────
    vram_usable_per_gpu_gb = vram_per_gpu_gb * gpu_memory_utilization

    # ── 3. VRAM còn lại cho KV cache ─────────────────────────────────────────
    vram_free_per_gpu_gb = (
        vram_usable_per_gpu_gb - vram_weights_per_gpu_gb - FRAMEWORK_OVERHEAD_GB
    )

    feasible = vram_free_per_gpu_gb > 0
    warning = None

    if not feasible:
        return HeuristicResult(
            vram_weights_per_gpu_gb=round(vram_weights_per_gpu_gb, 2),
            vram_overhead_gb=FRAMEWORK_OVERHEAD_GB,
            vram_usable_per_gpu_gb=round(vram_usable_per_gpu_gb, 2),
            vram_free_per_gpu_gb=round(vram_free_per_gpu_gb, 2),
            kv_per_token_bytes=0,
            max_kv_tokens=0,
            suggested_max_num_seqs=0,
            feasible=False,
            warning=f"Không đủ VRAM: cần ít nhất "
                    f"{vram_weights_per_gpu_gb + FRAMEWORK_OVERHEAD_GB:.1f} GB/GPU, "
                    f"chỉ có {vram_usable_per_gpu_gb:.1f} GB/GPU",
        )

    if vram_free_per_gpu_gb < 4:
        warning = f"Chỉ còn {vram_free_per_gpu_gb:.1f} GB/GPU cho KV cache — rất chật, dễ OOM"

    # ── 4. Tính KV cache capacity ─────────────────────────────────────────────
    # Số KV heads mỗi GPU sau khi TP split (GQA: heads có thể ít)
    num_kv_heads_per_gpu = max(1, model.num_key_value_heads // tp_size)
    head_dim = model.head_dim if model.head_dim > 0 else 128

    # Bytes cho K+V của 1 token, 1 layer, 1 GPU
    # K: head_dim × num_kv_heads_per_gpu × KV_CACHE_BYTES
    # V: head_dim × num_kv_heads_per_gpu × KV_CACHE_BYTES
    kv_per_token_per_layer = 2 * head_dim * num_kv_heads_per_gpu * KV_CACHE_BYTES

    # Tổng qua tất cả layers
    num_full_attn_layers = model.num_hidden_layers
    # Qwen3.5 có hybrid linear/full attention, nhưng KV cache cần cho full_attention layers
    # Để an toàn, tính tất cả layers (conservative estimate)
    kv_per_token_total = kv_per_token_per_layer * num_full_attn_layers

    if kv_per_token_total == 0:
        return HeuristicResult(
            vram_weights_per_gpu_gb=round(vram_weights_per_gpu_gb, 2),
            vram_overhead_gb=FRAMEWORK_OVERHEAD_GB,
            vram_usable_per_gpu_gb=round(vram_usable_per_gpu_gb, 2),
            vram_free_per_gpu_gb=round(vram_free_per_gpu_gb, 2),
            kv_per_token_bytes=0,
            max_kv_tokens=0,
            suggested_max_num_seqs=0,
            feasible=False,
            warning="Không tính được KV cache size (thiếu metadata model)",
        )

    # ── 5. Max tokens trong KV cache ──────────────────────────────────────────
    vram_free_bytes = vram_free_per_gpu_gb * (1024 ** 3)
    max_kv_tokens = int(vram_free_bytes / kv_per_token_total)

    # ── 6. Đề xuất max_num_seqs ───────────────────────────────────────────────
    raw_seqs = max_kv_tokens / AVG_SEQ_LEN_TOKENS
    # Làm tròn về lũy thừa 2 gần nhất, clamp [1, 512]
    suggested = _round_to_power_of_2(int(raw_seqs), min_val=1, max_val=512)

    return HeuristicResult(
        vram_weights_per_gpu_gb=round(vram_weights_per_gpu_gb, 2),
        vram_overhead_gb=FRAMEWORK_OVERHEAD_GB,
        vram_usable_per_gpu_gb=round(vram_usable_per_gpu_gb, 2),
        vram_free_per_gpu_gb=round(vram_free_per_gpu_gb, 2),
        kv_per_token_bytes=kv_per_token_total,
        max_kv_tokens=max_kv_tokens,
        suggested_max_num_seqs=suggested,
        feasible=True,
        warning=warning,
    )


def _round_to_power_of_2(value: int, min_val: int = 1, max_val: int = 512) -> int:
    """Làm tròn xuống lũy thừa 2 gần nhất trong khoảng [min_val, max_val]."""
    if value <= 0:
        return min_val
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    result = min_val
    for p in powers:
        if p <= value and min_val <= p <= max_val:
            result = p
    return result


def generate_sweep_configs(
    model: ModelInfo,
    gpu_ids: list[int],
    tp_sizes: Optional[list[int]] = None,
    max_num_seqs_values: Optional[list[int]] = None,
    gpu_memory_utils: Optional[list[float]] = None,
    docker_image: str = settings.default_docker_image,
    benchmark_params: Optional[BenchmarkParams] = None,
    dtype: str = "auto",
    max_model_len: Optional[int] = None,
    extra_args: Optional[dict] = None,
    env_vars: Optional[dict] = None,
) -> list[RunConfig]:
    """
    Sinh danh sách RunConfig để sweep tự động.
    Lọc bỏ các cấu hình mà heuristic dự đoán là không khả thi (INFEASIBLE).
    """
    if gpu_memory_utils is None:
        gpu_memory_utils = [0.85, 0.90, 0.95]

    # ── Tính valid TP sizes ───────────────────────────────────────────────────
    n_gpus = len(gpu_ids)
    if tp_sizes is None:
        # Lũy thừa 2 mà chia hết cho n_gpus, tối đa n_gpus
        tp_sizes = [2 ** i for i in range(int(math.log2(n_gpus)) + 1)
                    if n_gpus % (2 ** i) == 0]
    else:
        tp_sizes = [t for t in tp_sizes if n_gpus % t == 0]

    configs = []
    seen = set()

    for gpu_mem_util in gpu_memory_utils:
        for tp_size in sorted(tp_sizes):
            # Tính heuristic để lấy suggested_max_num_seqs
            h = compute_heuristic(
                model=model,
                tp_size=tp_size,
                gpu_count=n_gpus,
                gpu_memory_utilization=gpu_mem_util,
            )

            if not h.feasible:
                continue  # Bỏ qua config không khả thi

            # ── Xác định max_num_seqs cần sweep ──────────────────────────────
            if max_num_seqs_values is None:
                # Tự động sinh quanh giá trị heuristic
                base = h.suggested_max_num_seqs
                seq_candidates = _generate_seq_candidates(base)
            else:
                seq_candidates = max_num_seqs_values

            # Chọn GPU IDs phù hợp với TP size theo NVLink topology
            selected_gpu_ids = _select_gpu_ids(gpu_ids, tp_size)

            for max_seqs in seq_candidates:
                key = (tp_size, max_seqs, gpu_mem_util)
                if key in seen:
                    continue
                seen.add(key)

                configs.append(RunConfig(
                    model_name=model.name,
                    model_path=model.path,
                    gpu_ids=selected_gpu_ids,
                    tensor_parallel_size=tp_size,
                    max_num_seqs=max_seqs,
                    gpu_memory_utilization=gpu_mem_util,
                    docker_image=docker_image,
                    dtype=dtype,
                    max_model_len=max_model_len,
                    extra_args=extra_args or {},
                    env_vars=env_vars or {},
                ))

    return configs


def _generate_seq_candidates(base: int) -> list[int]:
    """
    Sinh danh sách max_num_seqs xung quanh giá trị baseline.
    Đảm bảo luôn bao gồm các giá trị quan trọng.
    """
    powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # Lấy lũy thừa 2 nằm trong khoảng [base/4, base*2]
    candidates = [p for p in powers if base // 4 <= p <= base * 2]
    # Đảm bảo có ít nhất 3 điểm
    if len(candidates) < 3:
        idx = powers.index(base) if base in powers else 0
        start = max(0, idx - 2)
        end = min(len(powers), idx + 3)
        candidates = powers[start:end]
    return sorted(set(candidates))


def _select_gpu_ids(gpu_ids: list[int], tp_size: int) -> list[int]:
    """
    Chọn tp_size GPU IDs từ danh sách, ưu tiên các nhóm NVLink gần nhau.
    """
    if tp_size >= len(gpu_ids):
        return gpu_ids[:tp_size]

    # Thử tìm nhóm NVLink preferred
    preferred = PREFERRED_GPU_GROUPS.get(tp_size, [])
    for group in preferred:
        if all(g in gpu_ids for g in group):
            return group

    # Fallback: lấy tp_size GPU đầu tiên
    return gpu_ids[:tp_size]


# ── Test nhanh ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from scanner import scan_single_model

    # Test với Qwen3.5-122B-FP8
    model = scan_single_model(
        "/projects/MedTrivita/common/models/Qwen3.5-122B-A10B-FP8"
    )
    if model:
        print(f"\nModel: {model.name}")
        print(f"  Params: {model.total_params_b}B total / {model.active_params_b}B active")
        print(f"  MoE: {model.is_moe}, Experts: {model.num_experts}/{model.num_experts_per_tok}")
        print(f"  Precision: {model.precision} ({model.bytes_per_param} B/param)")

        for tp in [1, 2, 4, 8]:
            h = compute_heuristic(model, tp_size=tp, gpu_count=4, gpu_memory_utilization=0.90)
            status = "OK" if h.feasible else "FAIL"
            print(f"  TP={tp}: weights={h.vram_weights_per_gpu_gb:.1f}GB/GPU "
                  f"free={h.vram_free_per_gpu_gb:.1f}GB/GPU "
                  f"max_seqs={h.suggested_max_num_seqs} [{status}]")
            if h.warning:
                print(f"    ⚠ {h.warning}")

    # Test với Qwen3.5-397B-FP8
    model397 = scan_single_model(
        "/projects/MedTrivita/common/models/Qwen3.5-397B-FP8"
    )
    if model397:
        print(f"\nModel: {model397.name}")
        h = compute_heuristic(model397, tp_size=8, gpu_count=8, gpu_memory_utilization=0.80)
        print(f"  TP=8 at 0.80 util: weights={h.vram_weights_per_gpu_gb:.1f}GB free={h.vram_free_per_gpu_gb:.1f}GB seqs={h.suggested_max_num_seqs}")
        h2 = compute_heuristic(model397, tp_size=8, gpu_count=8, gpu_memory_utilization=0.90)
        print(f"  TP=8 at 0.90 util: weights={h2.vram_weights_per_gpu_gb:.1f}GB free={h2.vram_free_per_gpu_gb:.1f}GB [{('OK' if h2.feasible else 'FAIL')}]")
