"""
FastAPI Backend — Orchestrator chính của vLLM Auto-Tuner.

Endpoints:
  GET  /api/models          - Danh sách models
  GET  /api/gpus            - GPU status
  POST /api/heuristic       - Tính VRAM budget
  POST /api/sweep           - Bắt đầu sweep tự động
  GET  /api/sweep/{id}      - Trạng thái sweep
  GET  /api/runs            - Lịch sử benchmark
  GET  /api/runs/{id}       - Chi tiết run
  DELETE /api/runs/{id}     - Xóa run
  GET  /api/export          - Download Excel
  WS   /ws                  - Real-time updates

Serve React SPA từ ../frontend/build/
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Optional

import aiosqlite
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import db
import docker_worker
from benchmark import run_benchmark
from exporter import export_to_xlsx
from heuristic import compute_heuristic, generate_sweep_configs
from models import (
    BenchmarkMetrics, BenchmarkResult, GPUInfo, HeuristicRequest,
    HeuristicResult, ModelInfo, RunConfig, RunStatus, SweepRequest,
    SweepStatus, WSMessage,
)
from scanner import scan_models, scan_single_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── WebSocket Manager ─────────────────────────────────────────────────────────

class WSManager:
    """Quản lý kết nối WebSocket và broadcast messages."""
    def __init__(self):
        self.connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        logger.info(f"WebSocket connected, total: {len(self.connections)}")

    def disconnect(self, ws: WebSocket):
        self.connections.remove(ws) if ws in self.connections else None
        logger.info(f"WebSocket disconnected, total: {len(self.connections)}")

    async def broadcast(self, event: str, data: dict):
        """Gửi message đến tất cả clients đang kết nối."""
        msg = json.dumps({"event": event, "data": data})
        dead = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = WSManager()

# ── Sweep state ───────────────────────────────────────────────────────────────

# Lưu trạng thái các sweep đang/đã chạy
_sweeps: dict[str, SweepStatus] = {}

# ── Cache model list ──────────────────────────────────────────────────────────

_models_cache: Optional[list[ModelInfo]] = None
_models_cache_time: float = 0
_MODELS_CACHE_TTL = 60  # giây


async def get_cached_models() -> list[ModelInfo]:
    global _models_cache, _models_cache_time
    if _models_cache is None or (time.time() - _models_cache_time) > _MODELS_CACHE_TTL:
        loop = asyncio.get_event_loop()
        _models_cache = await loop.run_in_executor(None, scan_models)
        _models_cache_time = time.time()
        # Lưu vào DB để tra cứu nhanh
        for m in _models_cache:
            await db.upsert_model(m)
    return _models_cache


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Khởi tạo khi app start, cleanup khi app stop."""
    await db.init_db()
    n = await docker_worker.cleanup_stale_containers()
    if n:
        logger.warning(f"Đã dọn {n} stale containers từ lần chạy trước")
    yield
    # Khi shutdown: dừng sweep đang chạy nếu có
    for sweep in _sweeps.values():
        sweep.status = "cancelled"


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="vLLM Auto-Tuner",
    description="Tự động tìm cấu hình vLLM tối ưu trên H100",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/models", response_model=list[ModelInfo])
async def list_models():
    """Quét và trả về danh sách models từ thư mục cài đặt."""
    return await get_cached_models()


@app.get("/api/models/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    """Lấy thông tin chi tiết một model."""
    models = await get_cached_models()
    for m in models:
        if m.name == model_name:
            return m
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' không tìm thấy")


@app.get("/api/gpus", response_model=list[GPUInfo])
async def list_gpus():
    """Lấy trạng thái GPU từ nvidia-smi."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, _query_gpus)
    return result


def _query_gpus() -> list[GPUInfo]:
    """Gọi nvidia-smi để lấy VRAM usage."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=10)
    except Exception as e:
        logger.error(f"nvidia-smi error: {e}")
        return []

    gpus = []
    active_names = docker_worker.get_active_containers()
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            idx = int(parts[0])
            gpus.append(GPUInfo(
                index=idx,
                name=parts[1],
                memory_total_gb=round(float(parts[2]) / 1024, 2),
                memory_used_gb=round(float(parts[3]) / 1024, 2),
                memory_free_gb=round(float(parts[4]) / 1024, 2),
                utilization_pct=float(parts[5]),
                in_use_by_autotuner=any(
                    str(idx) in name for name in active_names
                ),
            ))
        except (ValueError, IndexError):
            continue

    return gpus


@app.post("/api/heuristic", response_model=HeuristicResult)
async def heuristic_estimate(req: HeuristicRequest):
    """Tính VRAM budget và đề xuất max_num_seqs cho cấu hình cụ thể."""
    models = await get_cached_models()
    model = next((m for m in models if m.name == req.model_name), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' không tìm thấy")

    result = compute_heuristic(
        model=model,
        tp_size=req.tp_size,
        gpu_count=req.gpu_count,
        gpu_memory_utilization=req.gpu_memory_utilization,
        vram_per_gpu_gb=req.vram_per_gpu_gb,
    )
    return result


@app.post("/api/sweep")
async def start_sweep(req: SweepRequest):
    """
    Bắt đầu sweep tự động nhiều cấu hình.
    Chạy background task, trả về sweep_id ngay.
    """
    models = await get_cached_models()
    model = next((m for m in models if m.name == req.model_name), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{req.model_name}' không tìm thấy")

    sweep_id = str(uuid.uuid4())[:8]

    configs = generate_sweep_configs(
        model=model,
        gpu_ids=req.gpu_ids,
        tp_sizes=req.tp_sizes,
        max_num_seqs_values=req.max_num_seqs_values,
        gpu_memory_utils=req.gpu_memory_utils,
        docker_image=req.docker_image,
        benchmark_params=req.benchmark_params,
        dtype=req.dtype,
        max_model_len=req.max_model_len,
        extra_args=req.extra_args,
        env_vars=req.env_vars,
    )

    if not configs:
        raise HTTPException(
            status_code=400,
            detail="Không có cấu hình khả thi — kiểm tra lại GPU count và model size"
        )

    sweep_status = SweepStatus(
        sweep_id=sweep_id,
        status="running",
        total=len(configs),
        completed=0,
    )
    _sweeps[sweep_id] = sweep_status

    # Chạy sweep trong background
    asyncio.create_task(_run_sweep(sweep_id, model, configs, req))

    return {"sweep_id": sweep_id, "total_configs": len(configs)}


@app.get("/api/sweep/{sweep_id}")
async def get_sweep_status(sweep_id: str):
    """Lấy trạng thái và kết quả của một sweep."""
    sweep = _sweeps.get(sweep_id)
    if not sweep:
        raise HTTPException(status_code=404, detail=f"Sweep '{sweep_id}' không tìm thấy")
    return sweep


@app.get("/api/runs")
async def list_runs(
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """Lấy lịch sử benchmark runs."""
    return await db.get_runs(model_name=model_name, status=status, limit=limit, offset=offset)


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Lấy chi tiết đầy đủ của một run."""
    run = await db.get_run_detail(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' không tìm thấy")
    return run


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str):
    """Xóa một run khỏi database."""
    ok = await db.delete_run(run_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' không tìm thấy")
    return {"ok": True}


@app.get("/api/export")
async def export_runs(model_name: Optional[str] = None):
    """Xuất benchmark results ra Excel file."""
    runs = await db.get_all_runs_for_export(model_name=model_name)
    if not runs:
        raise HTTPException(status_code=404, detail="Không có dữ liệu để xuất")

    loop = asyncio.get_event_loop()
    filepath = await loop.run_in_executor(
        None, lambda: export_to_xlsx(runs, model_name=model_name)
    )

    return FileResponse(
        path=filepath,
        filename=os.path.basename(filepath),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── WebSocket ─────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint cho real-time updates từ sweep/benchmark."""
    await ws_manager.connect(ws)
    try:
        while True:
            # Giữ kết nối, nhận ping/pong
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)


# ── Background Sweep Logic ────────────────────────────────────────────────────

async def _run_sweep(
    sweep_id: str,
    model: ModelInfo,
    configs: list[RunConfig],
    req: SweepRequest,
) -> None:
    """
    Chạy tuần tự các cấu hình benchmark.
    Broadcast kết quả real-time qua WebSocket.
    """
    sweep = _sweeps[sweep_id]

    # WebSocket callback để stream log từ container
    async def ws_progress(msg: str):
        await ws_manager.broadcast("run_progress", {"message": msg, "sweep_id": sweep_id})

    for i, config in enumerate(configs):
        if sweep.status == "cancelled":
            break

        run_id = f"{sweep_id}-{i+1:02d}"
        result = BenchmarkResult(run_id=run_id, config=config)
        sweep.current_config = config

        # Tính heuristic cho config này
        h = compute_heuristic(
            model=model,
            tp_size=config.tensor_parallel_size,
            gpu_count=len(config.gpu_ids),
            gpu_memory_utilization=config.gpu_memory_utilization,
        )
        result.heuristic = h

        await ws_manager.broadcast("run_started", {
            "run_id": run_id,
            "sweep_id": sweep_id,
            "config": config.model_dump(),
            "heuristic": h.model_dump(),
            "index": i + 1,
            "total": len(configs),
        })

        # Lấy GPU locks cho tất cả GPUs config này cần
        gpu_ids_needed = config.gpu_ids[:config.tensor_parallel_size]
        locks = [docker_worker.get_gpu_lock(g) for g in gpu_ids_needed]

        # Acquire locks theo thứ tự tăng dần để tránh deadlock
        for lock in sorted(locks, key=id):
            await lock.acquire()

        port = await docker_worker.allocate_port()
        container = None

        try:
            if port is None:
                raise RuntimeError("Không còn port khả dụng trong dải 9200-9299")

            result.status = RunStatus.STARTING
            await db.update_run_status(run_id, RunStatus.STARTING)

            # Launch container trong thread pool
            loop = asyncio.get_event_loop()
            container = await loop.run_in_executor(
                None,
                lambda: docker_worker.launch_container_sync(config, port)
            )
            result.container_id = container.id[:12]

            # Monitor container đến khi ready hoặc lỗi
            outcome = await docker_worker.monitor_container(
                container=container,
                port=port,
                timeout_s=req.benchmark_params.timeout_per_run_s,
                ws_callback=ws_progress,
            )

            if outcome.status == "ready":
                result.status = RunStatus.BENCHMARKING
                await db.update_run_status(run_id, RunStatus.BENCHMARKING)
                await ws_manager.broadcast("run_progress", {
                    "message": f"[{run_id}] Container ready, bắt đầu benchmark...",
                    "sweep_id": sweep_id,
                })

                # Lấy tên model trong container
                import os
                container_model_name = os.path.basename(config.model_path)

                metrics = await run_benchmark(
                    endpoint=f"http://localhost:{port}",
                    model_name=container_model_name,
                    params=req.benchmark_params,
                    startup_time_s=outcome.startup_time_s,
                )
                result.metrics = metrics
                result.status = RunStatus.SUCCESS

            else:
                result.status = RunStatus.FAILED
                result.error_class = outcome.error_class
                result.error_log = outcome.log_tail

        except Exception as e:
            logger.exception(f"Lỗi run {run_id}: {e}")
            result.status = RunStatus.FAILED
            result.error_log = str(e)

        finally:
            # Cleanup container (đảm bảo VRAM được thu hồi)
            if container:
                await docker_worker.cleanup_container(container)
            if port:
                docker_worker.release_port(port)
            # Giải phóng GPU locks
            for lock in locks:
                lock.release()

        # Lưu kết quả
        from datetime import datetime
        result.completed_at = datetime.now().isoformat()
        await db.save_run(result)

        # Cập nhật sweep state
        sweep.completed += 1
        sweep.results.append(result)

        await ws_manager.broadcast("run_completed", {
            "run_id": run_id,
            "sweep_id": sweep_id,
            "status": result.status.value,
            "metrics": result.metrics.model_dump() if result.metrics else None,
            "error_class": result.error_class.value if result.error_class else None,
            "config": config.model_dump(),
        })

    # Sweep hoàn thành
    sweep.status = "completed"
    await ws_manager.broadcast("sweep_completed", {
        "sweep_id": sweep_id,
        "total": sweep.total,
        "completed": sweep.completed,
        "success_count": sum(1 for r in sweep.results if r.status == RunStatus.SUCCESS),
    })
    logger.info(f"Sweep {sweep_id} hoàn thành: {sweep.completed}/{sweep.total} runs")


# ── Static Files (React SPA) ──────────────────────────────────────────────────

_FRONTEND_BUILD = os.path.join(os.path.dirname(__file__), "..", "frontend", "build")

if os.path.isdir(_FRONTEND_BUILD):
    app.mount("/", StaticFiles(directory=_FRONTEND_BUILD, html=True), name="spa")
    logger.info(f"Serving React SPA từ {_FRONTEND_BUILD}")
else:
    @app.get("/")
    async def root():
        return {"message": "vLLM Auto-Tuner API. Frontend chưa được build."}


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=9100, workers=1, reload=False)
