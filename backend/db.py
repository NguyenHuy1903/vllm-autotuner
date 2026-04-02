"""
Database layer — SQLite async với aiosqlite.
Lưu trữ kết quả benchmark và thông tin models.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

import aiosqlite

from models import BenchmarkResult, ErrorClass, ModelInfo, RunStatus
from settings import settings

# Đường dẫn database
DB_PATH = settings.db_path

# DDL
CREATE_MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS models (
    name TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    architectures TEXT,
    model_type TEXT,
    is_moe INTEGER DEFAULT 0,
    total_params_b REAL DEFAULT 0,
    active_params_b REAL DEFAULT 0,
    num_experts INTEGER,
    num_experts_per_tok INTEGER,
    precision TEXT,
    bytes_per_param REAL,
    hidden_size INTEGER,
    num_hidden_layers INTEGER,
    num_attention_heads INTEGER,
    num_key_value_heads INTEGER,
    head_dim INTEGER,
    vocab_size INTEGER,
    intermediate_size INTEGER,
    moe_intermediate_size INTEGER,
    disk_size_gb REAL,
    is_multimodal INTEGER DEFAULT 0,
    torch_dtype TEXT,
    scanned_at TEXT
)
"""

CREATE_RUNS_TABLE = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    run_id TEXT PRIMARY KEY,
    model_name TEXT,
    gpu_ids TEXT,
    tensor_parallel_size INTEGER,
    max_num_seqs INTEGER,
    gpu_memory_utilization REAL,
    docker_image TEXT,
    dtype TEXT,
    max_model_len INTEGER,
    status TEXT DEFAULT 'pending',
    throughput_tok_s REAL,
    ttft_ms REAL,
    avg_latency_ms REAL,
    p50_latency_ms REAL,
    p99_latency_ms REAL,
    max_concurrent_tested INTEGER,
    startup_time_s REAL,
    error_class TEXT,
    error_log TEXT,
    heuristic_json TEXT,
    config_json TEXT,
    metrics_json TEXT,
    concurrency_details_json TEXT,
    created_at TEXT,
    completed_at TEXT
)
"""


async def init_db() -> None:
    """Khởi tạo database, tạo tables nếu chưa có."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(CREATE_MODELS_TABLE)
        await db.execute(CREATE_RUNS_TABLE)
        await db.commit()


async def upsert_model(model: ModelInfo) -> None:
    """Lưu hoặc cập nhật thông tin model."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO models VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            model.name, model.path,
            json.dumps(model.architectures),
            model.model_type,
            int(model.is_moe),
            model.total_params_b, model.active_params_b,
            model.num_experts, model.num_experts_per_tok,
            model.precision, model.bytes_per_param,
            model.hidden_size, model.num_hidden_layers,
            model.num_attention_heads, model.num_key_value_heads,
            model.head_dim, model.vocab_size,
            model.intermediate_size, model.moe_intermediate_size,
            model.disk_size_gb,
            int(model.is_multimodal),
            model.torch_dtype,
            datetime.now().isoformat(),
        ))
        await db.commit()


async def save_run(result: BenchmarkResult) -> None:
    """Lưu một benchmark run vào database."""
    metrics = result.metrics
    config = result.config

    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR REPLACE INTO benchmark_runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
        """, (
            result.run_id,
            config.model_name,
            json.dumps(config.gpu_ids),
            config.tensor_parallel_size,
            config.max_num_seqs,
            config.gpu_memory_utilization,
            config.docker_image,
            config.dtype,
            config.max_model_len,
            result.status.value,
            # Metrics (None nếu failed)
            metrics.throughput_tok_s if metrics else None,
            metrics.ttft_ms if metrics else None,
            metrics.avg_latency_ms if metrics else None,
            metrics.p50_latency_ms if metrics else None,
            metrics.p99_latency_ms if metrics else None,
            metrics.max_concurrent_tested if metrics else None,
            metrics.startup_time_s if metrics else None,
            result.error_class.value if result.error_class else None,
            result.error_log,
            result.heuristic.model_dump_json() if result.heuristic else None,
            config.model_dump_json(),
            metrics.model_dump_json() if metrics else None,
            json.dumps([d.model_dump() for d in metrics.concurrency_details]) if metrics else None,
            result.created_at,
            result.completed_at,
        ))
        await db.commit()


async def update_run_status(run_id: str, status: RunStatus) -> None:
    """Cập nhật status của một run."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "UPDATE benchmark_runs SET status = ? WHERE run_id = ?",
            (status.value, run_id)
        )
        await db.commit()


async def get_runs(
    model_name: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Lấy danh sách runs theo filter."""
    conditions = []
    params = []

    if model_name:
        conditions.append("model_name = ?")
        params.append(model_name)
    if status:
        conditions.append("status = ?")
        params.append(status)

    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    params.extend([limit, offset])

    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM benchmark_runs {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_run_detail(run_id: str) -> Optional[dict]:
    """Lấy chi tiết một run theo ID."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?", (run_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None


async def delete_run(run_id: str) -> bool:
    """Xóa một run."""
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "DELETE FROM benchmark_runs WHERE run_id = ?", (run_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


async def get_all_runs_for_export(model_name: Optional[str] = None) -> list[dict]:
    """Lấy tất cả runs (không phân trang) để export Excel."""
    params = []
    where = ""
    if model_name:
        where = "WHERE model_name = ?"
        params.append(model_name)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            f"SELECT * FROM benchmark_runs {where} ORDER BY created_at ASC",
            params
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
