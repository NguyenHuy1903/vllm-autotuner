# Advanced Options Panel — vLLM Auto-Tuner

## Context
ConfigPanel hiện tại quá naive: hardcode `concurrent_users=[1,4,8,16]`, `timeout=900`, không expose các vLLM flags quan trọng (prefix caching, chunked prefill, dtype, kv-cache-dtype...), và chỉ cho chọn 1 giá trị `gpu_memory_util` thay vì sweep nhiều giá trị. Cần thêm "⚙ Advanced" toggle để expose đủ các options theo team playbook và vLLM benchmark docs.

---

## Scope

### Backend — 2 files
**`backend/models.py`** — Thêm 3 fields vào `SweepRequest`:
```python
extra_args: dict = Field(default_factory=dict)   # vLLM flags cho tất cả configs
dtype: str = "auto"
max_model_len: Optional[int] = None
```
*(tp_sizes, max_num_seqs_values đã có sẵn — chỉ cần expose từ UI)*

**`backend/heuristic.py`** — Update `generate_sweep_configs` để nhận và propagate 3 fields mới vào từng `RunConfig`:
```python
def generate_sweep_configs(
    ...,
    extra_args: dict | None = None,
    dtype: str = "auto",
    max_model_len: int | None = None,
) -> list[RunConfig]:
    # Trong RunConfig(...) thêm:
    dtype=dtype,
    max_model_len=max_model_len,
    extra_args=extra_args or {},
```

**`backend/main.py`** — `start_sweep()` truyền 3 fields mới xuống `generate_sweep_configs()`.

---

### Frontend — 1 file
**`frontend/src/components/ConfigPanel.jsx`**

#### New state
```js
const [showAdvanced, setShowAdvanced] = useState(false);

// vLLM Server Flags
const [dtype, setDtype] = useState("auto");
const [maxModelLen, setMaxModelLen] = useState("");
const [kvCacheDtype, setKvCacheDtype] = useState("auto");
const [enablePrefixCaching, setEnablePrefixCaching] = useState(false);
const [enforceEager, setEnforceEager] = useState(false);
const [enableChunkedPrefill, setEnableChunkedPrefill] = useState(false);

// Sweep Control
const [advMemUtils, setAdvMemUtils] = useState([]);     // [] = dùng slider value
const [customTpSizes, setCustomTpSizes] = useState([]); // [] = auto
const [maxNumSeqsMode, setMaxNumSeqsMode] = useState("auto");
const [maxNumSeqsCustom, setMaxNumSeqsCustom] = useState("8,16,32,64");

// Benchmark Workload
const [concurrentUsers, setConcurrentUsers] = useState("1,4,8,16");
const [timeoutPerRun, setTimeoutPerRun] = useState(900);
```

#### UI layout (collapsed by default)
```
[⚙ Advanced Options ▼]

─── vLLM Server Flags ────────────────────────────
dtype:          <select> auto | bfloat16 | float16 | fp8
max-model-len:  <input number> (để trống = không set)
kv-cache-dtype: <select> auto | fp8

☐ --enable-prefix-caching   (tip: "Always enable for RAG/chat")
☐ --enforce-eager            (tip: "Tắt CUDA graphs — debug only")
☐ --enable-chunked-prefill   (tip: "Multi-tenant stability")

─── Sweep Parameters ─────────────────────────────
TP sizes (override auto):
  [☐ TP=1] [☐ TP=2] [☐ TP=4] [☐ TP=8]   ← filter to valid given GPU count
  (none checked = auto từ GPU count)

GPU mem utils (sweep nhiều giá trị):
  [☐ 0.80] [☐ 0.85] [☑ 0.90] [☐ 0.95]  ← pre-select value từ slider
  (fallback về slider value nếu không check gì)

max_num_seqs:
  <radio> auto  |  custom: <input> "8,16,32,64"

─── Benchmark Workload ───────────────────────────
Concurrent users: <input text> "1,4,8,16"
Timeout per run:  <input number> 900  (seconds)
```

#### Logic thay đổi trong `handleStartSweep`:
```js
// Build extra_args từ checkboxes
const extra_args = {};
if (enablePrefixCaching) extra_args["enable_prefix_caching"] = true;
if (enforceEager)        extra_args["enforce_eager"] = true;
if (enableChunkedPrefill) extra_args["enable_chunked_prefill"] = true;
if (kvCacheDtype !== "auto") extra_args["kv_cache_dtype"] = kvCacheDtype;

// Sweep params
const gpu_memory_utils = advMemUtils.length > 0 ? advMemUtils : [gpuMemUtil];
const tp_sizes = customTpSizes.length > 0 ? customTpSizes : null;  // null = auto
const max_num_seqs_values = maxNumSeqsMode === "custom"
  ? maxNumSeqsCustom.split(",").map(Number).filter(Boolean)
  : null;  // null = auto

await startSweep({
  model_name: selectedModel,
  gpu_ids: selectedGPUs,
  gpu_memory_utils,
  tp_sizes,
  max_num_seqs_values,
  docker_image: dockerImage,
  dtype: dtype !== "auto" ? dtype : "auto",
  max_model_len: maxModelLen ? parseInt(maxModelLen) : null,
  extra_args,
  benchmark_params: {
    num_prompts: numPrompts,
    prompt_len: promptLen,
    max_tokens: maxTokens,
    concurrent_users: concurrentUsers.split(",").map(Number).filter(Boolean),
    timeout_per_run_s: timeoutPerRun,
  },
});
```

---

## Critical Files
- `frontend/src/components/ConfigPanel.jsx` — primary change
- `backend/models.py` — add 3 fields to SweepRequest
- `backend/heuristic.py` — propagate new fields to RunConfig
- `backend/main.py` — pass new fields to generate_sweep_configs

## Reuse
- `RunConfig.extra_args` + `build_vllm_command` in `docker_worker.py` already handle arbitrary flags — không cần đổi docker_worker
- `SweepRequest.tp_sizes` và `max_num_seqs_values` đã có sẵn — chỉ cần expose từ UI
- `BenchmarkParams.concurrent_users` và `timeout_per_run_s` đã có sẵn trong model

## Verification
1. Restart backend, mở UI
2. Chọn model + GPU, mở Advanced panel — check hiển thị đúng
3. Tick `--enable-prefix-caching`, chọn dtype=`bfloat16`, check 3 mem utils
4. Bắt đầu Sweep → xem backend log: container command phải có `--enable-prefix-caching --dtype bfloat16`
5. Kiểm tra sweep chạy nhiều configs hơn khi chọn nhiều mem utils
6. Không mở Advanced → behavior giống như cũ (backward-compatible)
