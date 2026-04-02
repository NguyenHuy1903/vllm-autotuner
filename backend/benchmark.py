"""
Benchmark Runner — Đo throughput, TTFT và latency của vLLM endpoint.

Quy trình:
  1. Warmup: 3 requests tuần tự
  2. Throughput test: num_prompts requests với max concurrency
  3. Latency profile: tại từng mức concurrent_users
"""
from __future__ import annotations

import asyncio
import statistics
import time
from typing import Optional

import httpx

from models import BenchmarkMetrics, BenchmarkParams, ConcurrencyDetail

# Prompt cố định để benchmark (tránh cache bias khi so sánh configs)
BENCHMARK_PROMPT_BASE = (
    "Hãy mô tả chi tiết về kiến trúc của một mô hình ngôn ngữ lớn hiện đại, "
    "bao gồm cơ chế attention, feed-forward network và các kỹ thuật tối ưu hóa. "
    "Trình bày cụ thể về cách hoạt động của multi-head attention và vai trò của "
    "positional encoding trong transformer architecture. "
)

# Khoảng 4 chars/token cho tiếng Anh/Việt lẫn lộn
CHARS_PER_TOKEN = 4


def build_prompt(target_tokens: int) -> str:
    """Tạo prompt có kích thước xấp xỉ target_tokens tokens."""
    target_chars = target_tokens * CHARS_PER_TOKEN
    repeats = max(1, target_chars // len(BENCHMARK_PROMPT_BASE) + 1)
    return (BENCHMARK_PROMPT_BASE * repeats)[:target_chars]


async def _single_request_non_stream(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    sampling_params: dict,
) -> tuple[float, int]:
    """
    Gửi 1 request non-streaming.
    Trả về (latency_ms, tokens_generated).
    """
    t0 = time.perf_counter()
    resp = await client.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": False,
            **sampling_params,
        },
        timeout=120.0,
    )
    resp.raise_for_status()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    data = resp.json()
    tokens = data.get("usage", {}).get("completion_tokens", max_tokens)
    return elapsed_ms, tokens


async def _single_request_stream_ttft(
    client: httpx.AsyncClient,
    endpoint: str,
    model_name: str,
    prompt: str,
    max_tokens: int,
    sampling_params: dict,
) -> tuple[float, float, int]:
    """
    Gửi 1 request streaming, đo TTFT và total latency.
    Trả về (ttft_ms, total_latency_ms, tokens_generated).
    """
    t0 = time.perf_counter()
    ttft_ms = 0.0
    total_tokens = 0

    async with client.stream(
        "POST",
        f"{endpoint}/v1/completions",
        json={
            "model": model_name,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "stream": True,
            **sampling_params,
        },
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        first_chunk = True
        async for chunk in resp.aiter_lines():
            if chunk.startswith("data: ") and chunk != "data: [DONE]":
                if first_chunk:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    first_chunk = False
                total_tokens += 1  # ước lượng 1 token/chunk

    total_latency_ms = (time.perf_counter() - t0) * 1000
    return ttft_ms, total_latency_ms, total_tokens


async def run_benchmark(
    endpoint: str,
    model_name: str,
    params: BenchmarkParams,
    startup_time_s: float = 0.0,
) -> BenchmarkMetrics:
    """
    Chạy full benchmark suite và trả về BenchmarkMetrics.

    Args:
        endpoint: URL của vLLM server, ví dụ "http://localhost:9200"
        model_name: Tên model (khớp với --served-model-name trong container)
        params: Tham số benchmark
        startup_time_s: Thời gian startup container đã đo trước đó
    """
    prompt = build_prompt(params.prompt_len)

    sampling_params = {
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
        "n": params.n,
        "presence_penalty": params.presence_penalty,
        "frequency_penalty": params.frequency_penalty,
        "repetition_penalty": params.repetition_penalty,
    }

    async with httpx.AsyncClient(timeout=180.0) as client:
        # ── 1. Warmup ─────────────────────────────────────────────────────────
        for _ in range(3):
            try:
                await _single_request_non_stream(
                    client, endpoint, model_name, prompt, params.max_tokens, sampling_params
                )
            except Exception:
                pass  # Warmup failure không ảnh hưởng kết quả

        # ── 2. Throughput test (max concurrency) ──────────────────────────────
        max_concurrent = max(params.concurrent_users)
        sem = asyncio.Semaphore(max_concurrent)
        total_tokens = 0
        t_start = time.perf_counter()

        async def bounded_request():
            nonlocal total_tokens
            async with sem:
                try:
                    _, tokens = await _single_request_non_stream(
                        client, endpoint, model_name, prompt, params.max_tokens, sampling_params
                    )
                    total_tokens += tokens
                except Exception:
                    pass

        tasks = [bounded_request() for _ in range(params.num_prompts)]
        await asyncio.gather(*tasks)

        wall_time = time.perf_counter() - t_start
        throughput_tok_s = total_tokens / wall_time if wall_time > 0 else 0

        # ── 3. Latency profile theo từng mức concurrent_users ─────────────────
        concurrency_details: list[ConcurrencyDetail] = []
        all_latencies: list[float] = []
        all_ttfts: list[float] = []

        for c in sorted(params.concurrent_users):
            sem_c = asyncio.Semaphore(c)
            latencies: list[float] = []
            ttfts: list[float] = []
            tokens_c = 0
            t_c = time.perf_counter()

            async def bounded_stream_request():
                nonlocal tokens_c
                async with sem_c:
                    try:
                        ttft, total_lat, toks = await _single_request_stream_ttft(
                            client, endpoint, model_name, prompt, params.max_tokens, sampling_params
                        )
                        latencies.append(total_lat)
                        ttfts.append(ttft)
                        tokens_c += toks
                    except Exception:
                        pass

            # 10 requests per concurrency level
            n_samples = min(10, params.num_prompts)
            sample_tasks = [bounded_stream_request() for _ in range(n_samples)]
            await asyncio.gather(*sample_tasks)

            wall_c = time.perf_counter() - t_c

            if latencies:
                all_latencies.extend(latencies)
                all_ttfts.extend(ttfts)
                throughput_c = tokens_c / wall_c if wall_c > 0 else 0

                concurrency_details.append(ConcurrencyDetail(
                    concurrent_users=c,
                    throughput_tok_s=round(throughput_c, 2),
                    ttft_ms=round(statistics.median(ttfts), 2),
                    avg_latency_ms=round(statistics.mean(latencies), 2),
                    p99_latency_ms=round(_percentile(latencies, 99), 2),
                ))

    # ── 4. Tổng hợp metrics ───────────────────────────────────────────────────
    ttft_ms = round(statistics.median(all_ttfts), 2) if all_ttfts else 0.0
    avg_lat = round(statistics.mean(all_latencies), 2) if all_latencies else 0.0
    p50_lat = round(_percentile(all_latencies, 50), 2) if all_latencies else 0.0
    p99_lat = round(_percentile(all_latencies, 99), 2) if all_latencies else 0.0

    return BenchmarkMetrics(
        throughput_tok_s=round(throughput_tok_s, 2),
        ttft_ms=ttft_ms,
        avg_latency_ms=avg_lat,
        p50_latency_ms=p50_lat,
        p99_latency_ms=p99_lat,
        max_concurrent_tested=max(params.concurrent_users),
        startup_time_s=round(startup_time_s, 2),
        concurrency_details=concurrency_details,
    )


def _percentile(data: list[float], pct: int) -> float:
    """Tính percentile đơn giản (linear interpolation)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (pct / 100) * (len(sorted_data) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_data) - 1)
    frac = idx - lower
    return sorted_data[lower] * (1 - frac) + sorted_data[upper] * frac
