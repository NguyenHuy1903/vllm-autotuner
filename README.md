# vLLM Auto-Tuner & 3D Dashboard

Tự động tìm cấu hình vLLM tối ưu trên server 8× H100 80GB — không cần tinh chỉnh thủ công.

---

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────┐
│  HOST (Conda/Python)                                    │
│                                                         │
│  FastAPI :9100  ──REST/WS──►  React Dashboard          │
│      │                         (Plotly 3D scatter)     │
│      │                                                  │
│      ▼                                                  │
│  Orchestrator                                           │
│  ├── scanner.py    (quét /projects/.../models)          │
│  ├── heuristic.py  (tính VRAM budget)                   │
│  ├── docker_worker.py  (spin-up/kill container)         │
│  ├── benchmark.py  (đo throughput, TTFT, latency)       │
│  └── db.py         (SQLite lưu kết quả)                 │
│                                                         │
└───────────────────┬─────────────────────────────────────┘
                    │ docker run --gpus "device=X,Y"
                    ▼
┌─────────────────────────────────────────────────────────┐
│  WORKER (Docker vllm/vllm-openai)                       │
│                                                         │
│  vLLM OpenAI-compatible server :8000                    │
│  - Isolated GPU access                                  │
│  - VRAM 100% released khi container stop                │
└─────────────────────────────────────────────────────────┘
```

---

## Cấu trúc thư mục

```
vllm-autotuner/
├── backend/
│   ├── main.py           # FastAPI app: REST + WebSocket + static serve
│   ├── models.py         # Pydantic schemas
│   ├── scanner.py        # Quét /projects/MedTrivita/common/models/
│   ├── heuristic.py      # Math engine: VRAM, KV cache, sweep configs
│   ├── docker_worker.py  # Docker container lifecycle
│   ├── log_parser.py     # Regex phân loại lỗi từ Docker logs
│   ├── benchmark.py      # Đo throughput / TTFT / latency
│   ├── db.py             # SQLite (aiosqlite) persistence
│   └── exporter.py       # Xuất Excel (.xlsx)
├── frontend/
│   ├── package.json
│   ├── public/index.html
│   └── src/
│       ├── App.jsx                         # Layout chính
│       ├── api.js                          # REST + WebSocket client
│       └── components/
│           ├── ModelSelector.jsx           # Dropdown model + info badge
│           ├── GPUSelector.jsx             # Grid checkbox GPU + VRAM bar
│           ├── ConfigPanel.jsx             # Form cấu hình + VRAM estimator
│           ├── Dashboard3D.jsx             # Plotly scatter3d + heatmap 2D
│           └── DetailPopup.jsx             # Modal chi tiết khi click điểm
├── data/
│   └── results.db                          # SQLite (tự tạo khi chạy)
├── config.yaml                             # Cấu hình infra (paths/ports/image)
├── setup_and_run.sh                        # Script setup + build lần đầu
└── README.md
```

---

## Yêu cầu

| Thành phần | Phiên bản |
|-----------|----------|
| Python (conda env `data`) | 3.10.19 |
| Docker | ≥ 24 |
| Node.js | 18.x |
| GPU | NVIDIA H100 80GB (hoặc tương đương) |

**Python packages** (đã có sẵn trong env `data`):

```
fastapi, uvicorn, docker, openpyxl, httpx, websockets, aiosqlite
```

**Docker image**:

```
vllm/vllm-openai:v0.18.1   (hoặc nightly)
```

---

## Cài đặt & Chạy

### File cấu hình hạ tầng (YAML)

Toàn bộ đường dẫn/port/image đã được tách ra file:

```bash
config.yaml
```

Các key quan trọng:
- `paths.models_dir`
- `paths.db_path`
- `docker.default_image`
- `docker.port_start`, `docker.port_end`
- `api.host`, `api.port`

Bạn có thể override bằng biến môi trường `AUTOTUNER_*` nếu cần.

### Bước 1 — Setup lần đầu

```bash
cd /home/data_team/usr/huynq/MLops/vllm-autotuner
bash setup_and_run.sh
```

Script này sẽ:
1. Cài `aiosqlite` nếu chưa có
2. Cài `pyyaml` để đọc cấu hình YAML
3. `npm install` + `npm run build` React frontend
4. Khởi tạo SQLite database

### Bước 2 — Chạy Backend (dùng tmux để tránh rớt SSH)

```bash
tmux new-session -s autotuner

conda activate data
cd /home/data_team/usr/huynq/MLops/vllm-autotuner/backend
python main.py
```

> **Lưu ý**: Phải dùng `--workers 1` vì hệ thống dùng in-process state (port allocator, GPU locks, WebSocket manager).

Detach khỏi tmux: `Ctrl+B` rồi `D`  
Attach lại: `tmux attach -t autotuner`

### Bước 3 — SSH Port Forwarding (trên máy cá nhân)

```bash
ssh -L <api_port>:localhost:<api_port> <username>@<server-ip>
```

Mở trình duyệt: **http://localhost:<api_port>**

---

## Hướng dẫn sử dụng

### 1. Chọn Model

Dropdown tự động quét thư mục cấu hình tại `paths.models_dir` trong `config.yaml` và hiển thị:
- Tổng params (total / active cho MoE)
- Precision badge (FP8, BF16, GPTQ-W4...)
- Dung lượng disk
- Architecture

### 2. Chọn GPUs

Grid checkbox 8 GPU với VRAM usage real-time (refresh mỗi 10 giây).  
GPU đang bận sẽ hiển thị badge **BUSY**.

### 3. Dự tính VRAM (tùy chọn nhưng khuyến khích)

Nhấn **"Dự tính VRAM"** để xem breakdown trước khi chạy:

```
VRAM_usable = 80 GB × gpu_memory_utilization
VRAM_free   = VRAM_usable - VRAM_weights - 7 GB overhead
max_kv_tokens = VRAM_free / (2 × head_dim × num_kv_heads/TP × 2B × num_layers)
max_num_seqs  = max_kv_tokens / 2048 (avg seq len)
```

Ví dụ — Qwen3.5-122B-FP8, TP=4, 4 GPU, util=0.90:
```
Weights = 122B × 1 byte / 4 = 30.5 GB/GPU
Usable  = 80 × 0.90 = 72 GB
Free    = 72 - 30.5 - 7 = 34.5 GB  → max_num_seqs ≈ 256
```

### 4. Bắt đầu Sweep

Nhấn **"Bắt đầu Sweep"** — hệ thống tự động:
1. Sinh danh sách cấu hình (TP sizes × max_num_seqs × gpu_mem_util)
2. Lọc bỏ configs không khả thi theo heuristic
3. Tuần tự spin-up Docker container cho từng config
4. Đo benchmark (throughput, TTFT, P50/P99 latency)
5. Dừng container (VRAM freed) → chạy config tiếp theo
6. Stream kết quả real-time qua WebSocket

### 5. Phân tích 3D Chart

| Trục | Ý nghĩa |
|------|---------|
| X | `max_num_seqs` |
| Y | `tensor_parallel_size` |
| Z | Metric đã chọn (throughput / TTFT / latency) |

- **Xoay 3D**: kéo chuột trực tiếp trên chart
- **Projection XY/XZ/YZ**: click button để xoay về góc 2D + hiện heatmap
- **Click điểm**: mở popup chi tiết (config, VRAM breakdown, metrics, error log)
- **Đổi Z-axis**: dropdown chuyển giữa throughput / TTFT / P99

### 6. Xuất Excel

Button **"Xuất Excel"** trên header hoặc trong popup chi tiết.  
File gồm 3 sheets:

| Sheet | Nội dung |
|-------|---------|
| Summary | 1 row/run: metrics chính |
| Configs | Cấu hình đầy đủ từng run |
| Errors | Chỉ failed runs + error log |

---

## API Endpoints

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| GET | `/api/models` | Danh sách models (cache theo `runtime.models_cache_ttl_s`) |
| GET | `/api/models/{name}` | Chi tiết 1 model |
| GET | `/api/gpus` | GPU status từ nvidia-smi |
| POST | `/api/heuristic` | Tính VRAM breakdown |
| POST | `/api/sweep` | Bắt đầu sweep (background) |
| GET | `/api/sweep/{id}` | Trạng thái + kết quả sweep |
| GET | `/api/runs` | Lịch sử benchmark |
| GET | `/api/runs/{id}` | Chi tiết 1 run |
| DELETE | `/api/runs/{id}` | Xóa 1 run |
| GET | `/api/export` | Download Excel |
| WS | `/ws` | Real-time updates |

Swagger UI: **http://localhost:9100/docs**

---

## Phân loại lỗi tự động

| Pattern trong Docker log | Error Class |
|--------------------------|-------------|
| `CUDA out of memory` | `OOM` |
| `tensor parallel size.*greater than` | `INVALID_TP` |
| `Segmentation fault` / `SIGSEGV` | `SEGFAULT` |
| `RuntimeError:` / `torch.*Error` | `TORCH_ERROR` |
| `FileNotFoundError` / `model not found` | `MODEL_NOT_FOUND` |
| Timeout (900s default) | `TIMEOUT` |

---

## Cấu hình đã kiểm chứng (Production Reference)

| Model | TP | GPU Util | max_num_seqs | VRAM/GPU (weights) |
|-------|----|----------|-------------|---------------------|
| Qwen3.5-122B-A10B-FP8 | 4 | 0.90 | 256 | ~28 GB |
| Qwen3.5-397B-FP8 | 8 | 0.80 | 256 | ~46 GB |

> **397B note**: Phải dùng `gpu_memory_utilization=0.80` (không phải 0.90) do overhead 14GB từ 8 processes. Xem `Final_Opti.md` để biết thêm chi tiết.

---

## Troubleshooting

**Backend không start được:**
```bash
# Kiểm tra port 9100 có bị chiếm không
ss -tlnp | grep 9100
```

**Container không dừng sau crash:**
```bash
docker ps | grep autotuner
docker stop $(docker ps -q --filter name=autotuner)
```

**Frontend chưa build:**
```bash
cd frontend && npm install && npm run build
```

**Kiểm tra scanner hoạt động:**
```bash
cd backend
python3 scanner.py
```

**Kiểm tra heuristic với model cụ thể:**
```bash
cd backend
python3 heuristic.py
```
