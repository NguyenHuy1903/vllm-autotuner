"""
Log Parser — Phân loại lỗi từ Docker container logs bằng Regex pattern matching.
Xác định khi nào container sẵn sàng hoặc gặp lỗi.
"""
from __future__ import annotations

import re
from models import ErrorClass

# ── Patterns phân loại lỗi ────────────────────────────────────────────────────

# Mỗi tuple: (pattern, error_class)
ERROR_PATTERNS: list[tuple[re.Pattern, ErrorClass]] = [
    # OOM: CUDA out of memory
    (re.compile(
        r"CUDA out of memory|torch\.OutOfMemoryError|oom-kill event|"
        r"Cannot allocate memory|out_of_memory|CUDA error: out of memory",
        re.IGNORECASE
    ), ErrorClass.OOM),

    # Invalid TP size
    (re.compile(
        r"tensor.parallel.size.+(?:greater than|must be|cannot be|exceeds)|"
        r"tp_size.+must be|"
        r"world_size.+must be a multiple|"
        r"--tensor-parallel-size.+must",
        re.IGNORECASE
    ), ErrorClass.INVALID_TP),

    # Segfault
    (re.compile(
        r"Segmentation fault|SIGSEGV|core dumped|Killed",
        re.IGNORECASE
    ), ErrorClass.SEGFAULT),

    # File not found (model path sai)
    (re.compile(
        r"FileNotFoundError|model.+not found|No such file or directory|"
        r"does not exist|cannot find|not a valid model",
        re.IGNORECASE
    ), ErrorClass.MODEL_NOT_FOUND),

    # Torch runtime errors (bắt sau OOM để không che)
    (re.compile(
        r"RuntimeError:|torch\.\w+Error:|AssertionError:|ValueError:|"
        r"NCCL error|CUDA error:|CUDAError",
        re.IGNORECASE
    ), ErrorClass.TORCH_ERROR),
]

# Pattern báo container đã sẵn sàng nhận requests
READY_PATTERNS: list[re.Pattern] = [
    re.compile(r"Uvicorn running on", re.IGNORECASE),
    re.compile(r"Application startup complete", re.IGNORECASE),
    re.compile(r"Started server process", re.IGNORECASE),
    re.compile(r"vLLM engine has started", re.IGNORECASE),
    re.compile(r"INFO.*Application startup complete", re.IGNORECASE),
]

# Pattern tiến trình loading (để stream progress lên WS)
PROGRESS_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Loading model weights", re.IGNORECASE), "Đang nạp weights..."),
    (re.compile(r"Loading weights took", re.IGNORECASE), "Weights nạp xong"),
    (re.compile(r"Profiling the memory usage", re.IGNORECASE), "Đang profile VRAM..."),
    (re.compile(r"Memory profiling complete", re.IGNORECASE), "Profile VRAM xong"),
    (re.compile(r"init_engine_args", re.IGNORECASE), "Khởi tạo engine..."),
    (re.compile(r"Starting vLLM API server", re.IGNORECASE), "Khởi động API server..."),
    (re.compile(r"GPU blocks.*CPU blocks", re.IGNORECASE), "KV cache blocks đã cấp phát"),
]


def classify_log_line(line: str) -> ErrorClass | None:
    """
    Phân loại một dòng log, trả về ErrorClass nếu có lỗi, None nếu bình thường.
    Ưu tiên OOM > INVALID_TP > SEGFAULT > MODEL_NOT_FOUND > TORCH_ERROR.
    """
    for pattern, error_class in ERROR_PATTERNS:
        if pattern.search(line):
            return error_class
    return None


def is_ready_signal(line: str) -> bool:
    """Kiểm tra xem dòng log này có báo container sẵn sàng không."""
    return any(p.search(line) for p in READY_PATTERNS)


def get_progress_message(line: str) -> str | None:
    """Trả về message tiến trình thân thiện nếu dòng log là một milestone."""
    for pattern, msg in PROGRESS_PATTERNS:
        if pattern.search(line):
            return msg
    return None


def extract_error_context(full_log: str, error_class: ErrorClass, context_lines: int = 10) -> str:
    """
    Trích xuất đoạn log xung quanh lỗi (context_lines dòng trước + sau).
    Trả về chuỗi trống nếu không tìm thấy pattern.
    """
    lines = full_log.splitlines()
    # Tìm pattern tương ứng với error_class
    target_pattern = None
    for pattern, ec in ERROR_PATTERNS:
        if ec == error_class:
            target_pattern = pattern
            break

    if target_pattern is None:
        return full_log[-3000:] if len(full_log) > 3000 else full_log  # tail of log

    for i, line in enumerate(lines):
        if target_pattern.search(line):
            start = max(0, i - context_lines)
            end = min(len(lines), i + context_lines + 1)
            return "\n".join(lines[start:end])

    return full_log[-3000:] if len(full_log) > 3000 else full_log


def parse_log_stream(log_bytes: bytes) -> tuple[bool, ErrorClass | None, str]:
    """
    Parse toàn bộ log từ container (bytes).
    Trả về (is_ready, error_class, last_relevant_lines).
    """
    try:
        text = log_bytes.decode("utf-8", errors="replace")
    except Exception:
        text = ""

    lines = text.splitlines()
    error_class = None
    is_ready = False

    for line in lines:
        if is_ready_signal(line):
            is_ready = True
        ec = classify_log_line(line)
        if ec is not None:
            # OOM có priority cao nhất
            if error_class is None or ec == ErrorClass.OOM:
                error_class = ec

    # Lấy 50 dòng cuối làm summary
    tail = "\n".join(lines[-50:]) if len(lines) > 50 else text

    return is_ready, error_class, tail


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        ("CUDA out of memory. Tried to allocate 20.00 GiB", ErrorClass.OOM),
        ("torch.OutOfMemoryError: CUDA out of memory.", ErrorClass.OOM),
        ("tensor parallel size (8) is greater than world_size (4)", ErrorClass.INVALID_TP),
        ("Segmentation fault (core dumped)", ErrorClass.SEGFAULT),
        ("FileNotFoundError: [Errno 2] No such file or directory: '/models/xxx'", ErrorClass.MODEL_NOT_FOUND),
        ("RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED", ErrorClass.TORCH_ERROR),
        ("INFO: Uvicorn running on http://0.0.0.0:8000", None),
        ("INFO: Application startup complete.", None),
    ]

    print("Test log_parser:")
    for line, expected in test_cases:
        result = classify_log_line(line)
        ready = is_ready_signal(line)
        status = "✓" if result == expected else "✗"
        if ready and expected is None:
            status = "✓ READY"
        print(f"  {status} [{result or 'None':20}] {line[:60]}")
