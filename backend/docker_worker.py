"""
Docker Worker — Quản lý lifecycle của vLLM Docker container.

Luồng:
  launch_vllm_container() → monitor_container() → cleanup_container()

Đảm bảo:
  - Mỗi GPU chỉ được 1 container dùng cùng lúc (GPU lock)
  - Port range 9200-9299 được cấp phát không trùng
  - Container bị kill triệt để khi crash/timeout → VRAM được thu hồi
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import docker
import docker.errors
import docker.types
import httpx

from log_parser import classify_log_line, get_progress_message, is_ready_signal
from models import BenchmarkResult, ErrorClass, RunConfig, RunStatus

logger = logging.getLogger(__name__)

# ── Cấu hình ──────────────────────────────────────────────────────────────────

# Volume mount: thư mục models trên host → /models trong container
HOST_MODELS_DIR = "/projects/MedTrivita/common/models"
CONTAINER_MODELS_DIR = "/models"

# Dải port dành cho vLLM containers
PORT_RANGE = range(9200, 9300)

# Interval poll health check (giây)
HEALTH_POLL_INTERVAL = 2.0

# Tên prefix của autotuner containers để nhận diện và cleanup
CONTAINER_PREFIX = "autotuner-"


# ── State management ──────────────────────────────────────────────────────────

# Lock per GPU index: ngăn 2 container dùng cùng GPU
_gpu_locks: dict[int, asyncio.Lock] = {}

# Ports đang được sử dụng
_ports_in_use: set[int] = set()
_port_lock = asyncio.Lock()


def get_gpu_lock(gpu_id: int) -> asyncio.Lock:
    """Lấy lock cho một GPU, tạo mới nếu chưa có."""
    if gpu_id not in _gpu_locks:
        _gpu_locks[gpu_id] = asyncio.Lock()
    return _gpu_locks[gpu_id]


async def allocate_port() -> Optional[int]:
    """Cấp phát port còn rảnh trong dải PORT_RANGE."""
    async with _port_lock:
        for port in PORT_RANGE:
            if port not in _ports_in_use:
                _ports_in_use.add(port)
                return port
    return None  # Hết ports


def release_port(port: int) -> None:
    """Giải phóng port sau khi container dừng."""
    _ports_in_use.discard(port)


# ── Container outcome ─────────────────────────────────────────────────────────

@dataclass
class ContainerOutcome:
    """Kết quả sau khi monitor container."""
    status: str  # "ready", "failed", "timeout"
    error_class: Optional[ErrorClass] = None
    log_tail: str = ""
    startup_time_s: float = 0.0
    container_id: Optional[str] = None


# ── Docker client ─────────────────────────────────────────────────────────────

_docker_client: Optional[docker.DockerClient] = None


def get_docker_client() -> docker.DockerClient:
    """Lazy singleton Docker client."""
    global _docker_client
    if _docker_client is None:
        _docker_client = docker.from_env()
    return _docker_client


# ── Core functions ────────────────────────────────────────────────────────────

def build_vllm_command(config: RunConfig) -> list[str]:
    """
    Xây dựng command args cho vLLM server bên trong container.
    Model path trong container là /models/<tên thư mục model>.
    """
    import os
    model_name = os.path.basename(config.model_path)
    # vLLM v0.9+ dùng positional arg cho model, --model bị deprecated
    # --disable-log-requests đã bị xóa trong v0.18+, thay bằng --disable-uvicorn-access-log
    cmd = [
        f"{CONTAINER_MODELS_DIR}/{model_name}",
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--max-num-seqs", str(config.max_num_seqs),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--port", "8000",
        "--disable-uvicorn-access-log",
        "--served-model-name", model_name,
    ]

    if config.max_model_len:
        cmd += ["--max-model-len", str(config.max_model_len)]

    if config.dtype != "auto":
        cmd += ["--dtype", config.dtype]

    # Extra args (ví dụ: --enforce-eager, --enable-prefix-caching)
    for key, value in config.extra_args.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            cmd.append(flag)
        elif value is not False and value is not None:
            cmd += [flag, str(value)]

    return cmd


def launch_container_sync(config: RunConfig, port: int) -> docker.models.containers.Container:
    """
    Khởi động Docker container (synchronous, chạy trong thread pool).
    Dùng docker-py để mount volumes, cấp GPU, map port.
    """
    client = get_docker_client()

    # GPU IDs dành cho container này (chỉ dùng tp_size GPUs)
    gpu_ids = config.gpu_ids[:config.tensor_parallel_size]
    device_ids = [str(g) for g in gpu_ids]

    # Tên container duy nhất để dễ nhận diện
    short_id = str(uuid.uuid4())[:8]
    model_short = config.model_name[:20].replace(".", "-").replace("_", "-")
    container_name = f"{CONTAINER_PREFIX}{model_short}-tp{config.tensor_parallel_size}-{short_id}"

    cmd = build_vllm_command(config)

    logger.info(f"Khởi động container: {container_name} trên GPUs {device_ids}, port {port}")
    logger.debug(f"vLLM args: {' '.join(cmd)}")

    container = client.containers.run(
        image=config.docker_image,
        command=cmd,
        detach=True,
        # --rm: tự xóa khi dừng, đảm bảo VRAM được thu hồi
        remove=True,
        name=container_name,
        # Map port 8000 trong container → port trên host
        ports={"8000/tcp": port},
        # Mount thư mục models từ host (read-only)
        volumes={
            HOST_MODELS_DIR: {
                "bind": CONTAINER_MODELS_DIR,
                "mode": "ro",
            }
        },
        # Dùng host IPC cho NCCL shared memory (ipc=host đã share /dev/shm của host, không cần shm_size)
        ipc_mode="host",
        # Cấp GPU cụ thể
        device_requests=[
            docker.types.DeviceRequest(
                device_ids=device_ids,
                capabilities=[["gpu"]],
            )
        ],
        # Env vars tối ưu (config.env_vars overrides defaults)
        environment={
            "OMP_NUM_THREADS": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            **{k: str(v) for k, v in config.env_vars.items()},
        },
    )

    return container


async def monitor_container(
    container,
    port: int,
    timeout_s: int = 900,
    ws_callback=None,
) -> ContainerOutcome:
    """
    Monitor container cho đến khi:
      1. Health check trả về 200 → READY
      2. Log có lỗi nghiêm trọng → FAILED
      3. Timeout → TIMEOUT

    ws_callback: async function(message: str) để stream progress lên WebSocket
    """
    start_time = time.time()
    log_buffer: list[str] = []
    error_class = None
    container_id = container.id[:12] if container.id else "unknown"

    # Chạy log streaming và health polling đồng thời
    async def stream_logs():
        nonlocal error_class
        loop = asyncio.get_event_loop()

        # Toàn bộ log iteration chạy trong thread pool để không block event loop.
        # Khi vLLM hết log (model loaded, im lặng), next(log_gen) sẽ block —
        # nếu chạy trong coroutine sẽ chặn poll_health() không chạy được.
        def _iter_logs_in_thread():
            nonlocal error_class
            try:
                for chunk in container.logs(stream=True, follow=True):
                    if isinstance(chunk, bytes):
                        line = chunk.decode("utf-8", errors="replace").rstrip()
                    else:
                        line = str(chunk).rstrip()

                    log_buffer.append(line)

                    ec = classify_log_line(line)
                    if ec and error_class is None:
                        error_class = ec

                    if ws_callback:
                        progress_msg = get_progress_message(line)
                        if progress_msg:
                            asyncio.run_coroutine_threadsafe(
                                ws_callback(f"[{container_id}] {progress_msg}"),
                                loop,
                            )
            except Exception as e:
                logger.debug(f"Log stream kết thúc: {e}")

        try:
            await loop.run_in_executor(None, _iter_logs_in_thread)
        except asyncio.CancelledError:
            pass

    async def poll_health():
        """Poll /health endpoint mỗi HEALTH_POLL_INTERVAL giây."""
        url = f"http://localhost:{port}/health"
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return True
                except Exception:
                    pass
                await asyncio.sleep(HEALTH_POLL_INTERVAL)

    # Khởi chạy log stream và health poll đồng thời
    log_task = asyncio.create_task(stream_logs())
    health_task = asyncio.create_task(poll_health())

    try:
        # Chờ: health OK hoặc timeout
        done, pending = await asyncio.wait(
            [health_task],
            timeout=timeout_s,
        )
    finally:
        # Hủy tất cả tasks còn đang chạy
        log_task.cancel()
        health_task.cancel()
        for t in [log_task, health_task]:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    elapsed = time.time() - start_time
    log_tail = "\n".join(log_buffer[-100:])  # 100 dòng cuối

    # Kiểm tra kết quả
    if health_task in done and not health_task.cancelled():
        try:
            result = health_task.result()
            if result:
                logger.info(f"Container {container_id} sẵn sàng sau {elapsed:.1f}s")
                return ContainerOutcome(
                    status="ready",
                    log_tail=log_tail,
                    startup_time_s=elapsed,
                    container_id=container_id,
                )
        except Exception:
            pass

    # Timeout hoặc lỗi
    if error_class:
        logger.warning(f"Container {container_id} gặp lỗi: {error_class} sau {elapsed:.1f}s")
        return ContainerOutcome(
            status="failed",
            error_class=error_class,
            log_tail=log_tail,
            startup_time_s=elapsed,
            container_id=container_id,
        )

    # Kiểm tra container có còn chạy không
    try:
        container.reload()
        status = container.status
    except Exception:
        status = "exited"

    if status == "exited":
        # Container tự thoát → OOM hoặc error
        detected_ec = error_class or ErrorClass.UNKNOWN
        logger.warning(f"Container {container_id} thoát sớm: {detected_ec}")
        return ContainerOutcome(
            status="failed",
            error_class=detected_ec,
            log_tail=log_tail,
            startup_time_s=elapsed,
            container_id=container_id,
        )

    logger.warning(f"Container {container_id} timeout sau {elapsed:.1f}s")
    return ContainerOutcome(
        status="timeout",
        error_class=ErrorClass.TIMEOUT,
        log_tail=log_tail,
        startup_time_s=elapsed,
        container_id=container_id,
    )


async def cleanup_container(container) -> None:
    """
    Dừng và xóa container triệt để.
    Đảm bảo VRAM được giải phóng hoàn toàn.
    """
    if container is None:
        return

    container_id = container.id[:12] if container.id else "unknown"
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(None, lambda: container.stop(timeout=15))
        logger.info(f"Container {container_id} đã dừng")
    except docker.errors.NotFound:
        pass  # Đã tự xóa (remove=True)
    except Exception as e:
        logger.warning(f"Không dừng được container {container_id}: {e}")
        # Force remove nếu stop thất bại
        try:
            await loop.run_in_executor(None, lambda: container.remove(force=True))
        except Exception as e2:
            logger.error(f"Không xóa được container {container_id}: {e2}")

    # Đợi ngắn để Linux thu hồi VRAM
    await asyncio.sleep(2.0)


async def cleanup_stale_containers() -> int:
    """
    Xóa tất cả containers autotuner còn sót từ lần chạy trước.
    Gọi khi FastAPI khởi động.
    Trả về số containers đã dọn.
    """
    client = get_docker_client()
    count = 0
    try:
        containers = client.containers.list(
            filters={"name": CONTAINER_PREFIX}
        )
        for c in containers:
            logger.warning(f"Dọn stale container: {c.name}")
            try:
                c.stop(timeout=10)
            except Exception:
                try:
                    c.remove(force=True)
                except Exception:
                    pass
            count += 1
    except Exception as e:
        logger.error(f"Lỗi cleanup stale containers: {e}")
    return count


def get_active_containers() -> list[str]:
    """Lấy danh sách tên containers autotuner đang chạy."""
    client = get_docker_client()
    try:
        containers = client.containers.list(filters={"name": CONTAINER_PREFIX})
        return [c.name for c in containers]
    except Exception:
        return []
