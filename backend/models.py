"""
Pydantic schemas cho vLLM Auto-Tuner.
Định nghĩa tất cả data models cho request/response giữa Frontend và Backend.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────────────

class ErrorClass(str, Enum):
    """Phân loại lỗi từ Docker container logs."""
    OOM = "OOM"
    INVALID_TP = "INVALID_TP"
    SEGFAULT = "SEGFAULT"
    TORCH_ERROR = "TORCH_ERROR"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


class RunStatus(str, Enum):
    """Trạng thái của một benchmark run."""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    BENCHMARKING = "benchmarking"
    SUCCESS = "success"
    FAILED = "failed"


# ── Model Info (từ scanner) ───────────────────────────────────────────────────

class ModelInfo(BaseModel):
    """Thông tin model được quét từ thư mục models."""
    name: str
    path: str
    architectures: list[str] = []
    model_type: str = ""
    is_moe: bool = False
    total_params_b: float = 0.0        # Tổng params (tỷ), bao gồm tất cả experts
    active_params_b: float = 0.0       # Params active mỗi token (cho MoE < total)
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    hidden_size: int = 0
    num_hidden_layers: int = 0
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    vocab_size: int = 0
    intermediate_size: int = 0
    moe_intermediate_size: Optional[int] = None
    precision: str = "bfloat16"        # fp8, bfloat16, float16, gptq-w4, ...
    bytes_per_param: float = 2.0
    disk_size_gb: float = 0.0
    is_multimodal: bool = False
    torch_dtype: str = "bfloat16"


# ── Heuristic ─────────────────────────────────────────────────────────────────

class HeuristicRequest(BaseModel):
    """Yêu cầu tính toán heuristic VRAM."""
    model_name: str
    tp_size: int = 1
    gpu_count: int = 1
    gpu_memory_utilization: float = 0.90
    vram_per_gpu_gb: float = 80.0


class HeuristicResult(BaseModel):
    """Kết quả tính toán heuristic."""
    vram_weights_per_gpu_gb: float
    vram_overhead_gb: float = 7.0
    vram_usable_per_gpu_gb: float
    vram_free_per_gpu_gb: float
    kv_per_token_bytes: float
    max_kv_tokens: int
    suggested_max_num_seqs: int
    feasible: bool
    warning: Optional[str] = None


# ── Run Config & Results ──────────────────────────────────────────────────────

class RunConfig(BaseModel):
    """Cấu hình cho một lần chạy benchmark."""
    model_name: str
    model_path: str
    gpu_ids: list[int]
    tensor_parallel_size: int = 1
    max_num_seqs: int = 256
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    extra_args: dict = Field(default_factory=dict)
    env_vars: dict = Field(default_factory=dict)
    docker_image: str = "vllm/vllm-openai:v0.18.1"


class BenchmarkParams(BaseModel):
    """Tham số cho benchmark."""
    num_prompts: int = 50
    prompt_len: int = 512
    max_tokens: int = 256
    concurrent_users: list[int] = Field(default_factory=lambda: [1, 4, 8, 16])
    timeout_per_run_s: int = 900  # 15 phút tổng (startup + benchmark)
    # SamplingParams
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    n: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0


class BenchmarkMetrics(BaseModel):
    """Kết quả đo hiệu năng."""
    throughput_tok_s: float = 0.0
    ttft_ms: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    max_concurrent_tested: int = 0
    startup_time_s: float = 0.0
    # Chi tiết theo từng mức concurrent
    concurrency_details: list[ConcurrencyDetail] = Field(default_factory=list)


class ConcurrencyDetail(BaseModel):
    """Metrics tại một mức concurrent users."""
    concurrent_users: int
    throughput_tok_s: float
    ttft_ms: float
    avg_latency_ms: float
    p99_latency_ms: float


class BenchmarkResult(BaseModel):
    """Kết quả đầy đủ của một benchmark run."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    config: RunConfig
    heuristic: Optional[HeuristicResult] = None
    status: RunStatus = RunStatus.PENDING
    metrics: Optional[BenchmarkMetrics] = None
    error_class: Optional[ErrorClass] = None
    error_log: Optional[str] = None
    container_id: Optional[str] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None


# ── Sweep ─────────────────────────────────────────────────────────────────────

class SweepRequest(BaseModel):
    """Yêu cầu chạy sweep tự động nhiều cấu hình."""
    model_name: str
    gpu_ids: list[int]
    tp_sizes: Optional[list[int]] = None           # None = tự động
    max_num_seqs_values: Optional[list[int]] = None # None = tự động
    gpu_memory_utils: list[float] = Field(default_factory=lambda: [0.85, 0.90, 0.95])
    docker_image: str = "vllm/vllm-openai:v0.18.1"
    benchmark_params: BenchmarkParams = Field(default_factory=BenchmarkParams)
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    extra_args: dict = Field(default_factory=dict)
    env_vars: dict = Field(default_factory=dict)


class SweepStatus(BaseModel):
    """Trạng thái sweep đang chạy."""
    sweep_id: str
    status: str  # "running", "completed", "cancelled"
    total: int
    completed: int
    current_config: Optional[RunConfig] = None
    results: list[BenchmarkResult] = Field(default_factory=list)


# ── GPU Info ──────────────────────────────────────────────────────────────────

class GPUInfo(BaseModel):
    """Thông tin GPU từ nvidia-smi."""
    index: int
    name: str
    memory_total_gb: float
    memory_used_gb: float
    memory_free_gb: float
    utilization_pct: float = 0.0
    in_use_by_autotuner: bool = False


# ── WebSocket Messages ────────────────────────────────────────────────────────

class WSMessage(BaseModel):
    """Message gửi qua WebSocket."""
    event: str  # run_started, run_progress, run_completed, sweep_completed
    data: dict = Field(default_factory=dict)
