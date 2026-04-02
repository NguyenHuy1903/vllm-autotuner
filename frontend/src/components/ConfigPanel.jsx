/**
 * ConfigPanel — Form cấu hình sweep: TP sizes, max_num_seqs, GPU util, docker image.
 * Tích hợp "Dự tính VRAM" để preview heuristic trước khi chạy.
 */
import React, { useState } from "react";
import { computeHeuristic, startSweep } from "../api";

const DOCKER_IMAGES = [
  "vllm/vllm-openai:v0.18.1",
  "vllm/vllm-openai:v0.18.0",
  "vllm/vllm-openai:nightly",
  "vllm/vllm-openai:latest",
];

const MEM_UTIL_OPTIONS = [0.80, 0.85, 0.90, 0.95];
const TP_OPTIONS = [1, 2, 4, 8];

function VRAMBreakdown({ heuristic }) {
  if (!heuristic) return null;
  const { vram_weights_per_gpu_gb: w, vram_overhead_gb: o, vram_free_per_gpu_gb: f, vram_usable_per_gpu_gb: u } = heuristic;
  const total = u || 72;
  const wPct = Math.min((w / total) * 100, 100);
  const oPct = Math.min((o / total) * 100, 100);
  const fPct = Math.max(0, (f / total) * 100);

  return (
    <div style={{ marginTop: 8, padding: "10px 12px", background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 4 }}>
      <div style={{ fontSize: 10, fontWeight: 600, color: "#6b7280", marginBottom: 8, textTransform: "uppercase", letterSpacing: "0.08em" }}>
        VRAM Budget / GPU
      </div>
      {/* Stacked bar */}
      <div style={{ display: "flex", height: 16, borderRadius: 3, overflow: "hidden", marginBottom: 8 }}>
        <div style={{ width: `${wPct}%`, background: "#3b82f6", title: "Weights" }} />
        <div style={{ width: `${oPct}%`, background: "#f59e0b", title: "Overhead" }} />
        <div style={{ width: `${fPct}%`, background: "#10b981", title: "KV Cache" }} />
      </div>
      <div style={{ display: "flex", gap: 12, fontSize: 11, fontFamily: "monospace" }}>
        <span><span style={{ color: "#3b82f6" }}>■</span> Weights: {w.toFixed(1)}GB</span>
        <span><span style={{ color: "#f59e0b" }}>■</span> Overhead: {o.toFixed(1)}GB</span>
        <span><span style={{ color: "#10b981" }}>■</span> KV Free: {f.toFixed(1)}GB</span>
      </div>
      <div style={{ marginTop: 8, fontSize: 11, fontFamily: "monospace", color: heuristic.feasible ? "#059669" : "#dc2626" }}>
        {heuristic.feasible
          ? `✓ Khả thi — đề xuất max_num_seqs = ${heuristic.suggested_max_num_seqs}`
          : `✗ Không đủ VRAM`}
      </div>
      {heuristic.warning && (
        <div style={{ fontSize: 11, color: "#d97706", marginTop: 4 }}>⚠ {heuristic.warning}</div>
      )}
    </div>
  );
}

export default function ConfigPanel({ selectedModel, selectedGPUs, onSweepStarted }) {
  const [dockerImage, setDockerImage] = useState(DOCKER_IMAGES[0]);
  const [gpuMemUtil, setGpuMemUtil] = useState(0.90);
  const [numPrompts, setNumPrompts] = useState(50);
  const [promptLen, setPromptLen] = useState(512);
  const [maxTokens, setMaxTokens] = useState(256);
  const [heuristic, setHeuristic] = useState(null);
  const [heuristicLoading, setHeuristicLoading] = useState(false);
  const [sweepLoading, setSweepLoading] = useState(false);
  const [error, setError] = useState(null);

  // Advanced panel
  const [showAdvanced, setShowAdvanced] = useState(false);
  // vLLM Server Flags
  const [dtype, setDtype] = useState("auto");
  const [maxModelLen, setMaxModelLen] = useState("");
  const [kvCacheDtype, setKvCacheDtype] = useState("auto");
  const [enablePrefixCaching, setEnablePrefixCaching] = useState(false);
  const [enforceEager, setEnforceEager] = useState(false);
  const [enableChunkedPrefill, setEnableChunkedPrefill] = useState(false);
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [languageModelOnly, setLanguageModelOnly] = useState(false);
  const [enableExpertParallel, setEnableExpertParallel] = useState(false);
  const [dataParallelSize, setDataParallelSize] = useState("");
  const [maxNumBatchedTokens, setMaxNumBatchedTokens] = useState("");
  const [performanceMode, setPerformanceMode] = useState("");
  const [reasoningParser, setReasoningParser] = useState("");
  // Environment Variables
  const [vllmUseDeepGemm, setVllmUseDeepGemm] = useState("");
  const [vllmUseFlashinferMoeFp8, setVllmUseFlashinferMoeFp8] = useState("");
  const [vllmAll2AllBackend, setVllmAll2AllBackend] = useState("");
  // Sweep Control
  const [advMemUtils, setAdvMemUtils] = useState([]);
  const [customTpSizes, setCustomTpSizes] = useState([]);
  const [maxNumSeqsMode, setMaxNumSeqsMode] = useState("auto");
  const [maxNumSeqsCustom, setMaxNumSeqsCustom] = useState("8,16,32,64");
  // Benchmark Workload
  const [concurrentUsers, setConcurrentUsers] = useState("1,4,8,16");
  const [timeoutPerRun, setTimeoutPerRun] = useState(900);
  // Sampling Params
  const [temperature, setTemperature] = useState(0.0);
  const [topP, setTopP] = useState(1.0);
  const [topK, setTopK] = useState(-1);
  const [nSeqs, setNSeqs] = useState(1);
  const [presencePenalty, setPresencePenalty] = useState(0.0);
  const [frequencyPenalty, setFrequencyPenalty] = useState(0.0);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.0);

  const canRun = selectedModel && selectedGPUs.length > 0;

  function toggleMemUtil(val) {
    setAdvMemUtils(prev =>
      prev.includes(val) ? prev.filter(v => v !== val) : [...prev, val]
    );
  }

  function toggleTpSize(val) {
    setCustomTpSizes(prev =>
      prev.includes(val) ? prev.filter(v => v !== val) : [...prev, val]
    );
  }

  async function handleEstimate() {
    if (!canRun) return;
    setHeuristicLoading(true);
    setError(null);
    try {
      // Tính cho TP = số GPU (worst case → kiểm tra feasibility)
      const tp = selectedGPUs.length;
      const result = await computeHeuristic({
        model_name: selectedModel,
        tp_size: tp,
        gpu_count: selectedGPUs.length,
        gpu_memory_utilization: gpuMemUtil,
        vram_per_gpu_gb: 80.0,
      });
      setHeuristic(result);
    } catch (e) {
      setError(e.message);
    } finally {
      setHeuristicLoading(false);
    }
  }

  async function handleStartSweep() {
    if (!canRun) return;
    setSweepLoading(true);
    setError(null);
    try {
      // Build extra_args from checkboxes and inputs
      const extra_args = {};
      if (enablePrefixCaching)   extra_args["enable_prefix_caching"] = true;
      if (enforceEager)          extra_args["enforce_eager"] = true;
      if (enableChunkedPrefill)  extra_args["enable_chunked_prefill"] = true;
      if (trustRemoteCode)       extra_args["trust_remote_code"] = true;
      if (languageModelOnly)     extra_args["language_model_only"] = true;
      if (enableExpertParallel)  extra_args["enable_expert_parallel"] = true;
      if (kvCacheDtype !== "auto") extra_args["kv_cache_dtype"] = kvCacheDtype;
      if (dataParallelSize)      extra_args["data_parallel_size"] = parseInt(dataParallelSize);
      if (maxNumBatchedTokens)   extra_args["max_num_batched_tokens"] = parseInt(maxNumBatchedTokens);
      if (performanceMode)       extra_args["performance_mode"] = performanceMode;
      if (reasoningParser)       extra_args["reasoning_parser"] = reasoningParser;

      // Build env_vars
      const env_vars = {};
      if (vllmUseDeepGemm !== "")       env_vars["VLLM_USE_DEEP_GEMM"] = vllmUseDeepGemm;
      if (vllmUseFlashinferMoeFp8 !== "") env_vars["VLLM_USE_FLASHINFER_MOE_FP8"] = vllmUseFlashinferMoeFp8;
      if (vllmAll2AllBackend)            env_vars["VLLM_ALL2ALL_BACKEND"] = vllmAll2AllBackend;

      const gpu_memory_utils = advMemUtils.length > 0 ? advMemUtils : [gpuMemUtil];
      const tp_sizes = customTpSizes.length > 0 ? customTpSizes : null;
      const max_num_seqs_values = maxNumSeqsMode === "custom"
        ? maxNumSeqsCustom.split(",").map(Number).filter(Boolean)
        : null;

      const result = await startSweep({
        model_name: selectedModel,
        gpu_ids: selectedGPUs,
        gpu_memory_utils,
        tp_sizes,
        max_num_seqs_values,
        docker_image: dockerImage,
        dtype,
        max_model_len: maxModelLen ? parseInt(maxModelLen) : null,
        extra_args,
        env_vars,
        benchmark_params: {
          num_prompts: numPrompts,
          prompt_len: promptLen,
          max_tokens: maxTokens,
          concurrent_users: concurrentUsers.split(",").map(Number).filter(Boolean),
          timeout_per_run_s: timeoutPerRun,
          temperature,
          top_p: topP,
          top_k: topK,
          n: nSeqs,
          presence_penalty: presencePenalty,
          frequency_penalty: frequencyPenalty,
          repetition_penalty: repetitionPenalty,
        },
      });
      onSweepStarted && onSweepStarted(result.sweep_id, result.total_configs);
    } catch (e) {
      setError(e.message);
    } finally {
      setSweepLoading(false);
    }
  }

  const labelStyle = { fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em", display: "block", marginBottom: 3, marginTop: 10 };
  const inputStyle = { width: "100%", padding: "6px 8px", border: "1px solid #d1d5db", borderRadius: 4, fontSize: 12, fontFamily: "monospace", boxSizing: "border-box" };
  const sectionStyle = { fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.1em", marginTop: 14, marginBottom: 6, borderBottom: "1px solid #e5e7eb", paddingBottom: 3 };
  const checkboxRowStyle = { display: "flex", alignItems: "center", gap: 6, marginTop: 6, fontSize: 12 };
  const chipStyle = (active) => ({
    padding: "3px 10px", borderRadius: 12, fontSize: 11, fontFamily: "monospace", cursor: "pointer",
    border: `1px solid ${active ? "#2563eb" : "#d1d5db"}`,
    background: active ? "#dbeafe" : "#f9fafb",
    color: active ? "#1d4ed8" : "#374151",
    userSelect: "none",
  });

  return (
    <div>
      <label style={labelStyle}>Docker Image</label>
      <select value={dockerImage} onChange={e => setDockerImage(e.target.value)} style={inputStyle}>
        {DOCKER_IMAGES.map(img => <option key={img} value={img}>{img}</option>)}
      </select>

      <label style={labelStyle}>GPU Memory Utilization: {(gpuMemUtil * 100).toFixed(0)}%</label>
      <input
        type="range" min={0.70} max={0.95} step={0.05}
        value={gpuMemUtil}
        onChange={e => setGpuMemUtil(parseFloat(e.target.value))}
        style={{ width: "100%", cursor: "pointer" }}
      />
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#9ca3af", fontFamily: "monospace" }}>
        <span>70%</span><span>80%</span><span>90%</span><span>95%</span>
      </div>

      <label style={labelStyle}>Số prompts benchmark</label>
      <input type="number" min={10} max={200} value={numPrompts}
        onChange={e => setNumPrompts(parseInt(e.target.value))} style={inputStyle} />

      <label style={labelStyle}>Prompt length (tokens ≈)</label>
      <input type="number" min={128} max={4096} value={promptLen}
        onChange={e => setPromptLen(parseInt(e.target.value))} style={inputStyle} />

      <label style={labelStyle}>Max tokens sinh ra</label>
      <input type="number" min={64} max={2048} value={maxTokens}
        onChange={e => setMaxTokens(parseInt(e.target.value))} style={inputStyle} />

      {/* Heuristic preview */}
      <button
        onClick={handleEstimate}
        disabled={!canRun || heuristicLoading}
        style={{
          marginTop: 14, width: "100%", padding: "7px 0",
          background: canRun ? "#f3f4f6" : "#e5e7eb",
          border: "1px solid #d1d5db", borderRadius: 4,
          fontSize: 12, cursor: canRun ? "pointer" : "not-allowed",
          fontWeight: 600, color: canRun ? "#374151" : "#9ca3af",
        }}
      >
        {heuristicLoading ? "Đang tính..." : "🔢 Dự tính VRAM"}
      </button>

      <VRAMBreakdown heuristic={heuristic} />

      {/* Advanced Options Toggle */}
      <button
        onClick={() => setShowAdvanced(v => !v)}
        style={{
          marginTop: 12, width: "100%", padding: "6px 0",
          background: showAdvanced ? "#eff6ff" : "#f9fafb",
          border: `1px solid ${showAdvanced ? "#bfdbfe" : "#e5e7eb"}`,
          borderRadius: 4, fontSize: 12, cursor: "pointer",
          fontWeight: 600, color: showAdvanced ? "#1d4ed8" : "#6b7280",
          textAlign: "left", paddingLeft: 10,
        }}
      >
        ⚙ Advanced Options {showAdvanced ? "▲" : "▼"}
      </button>

      {showAdvanced && (
        <div style={{ border: "1px solid #e5e7eb", borderTop: "none", borderRadius: "0 0 4px 4px", padding: "10px 12px", background: "#fafafa" }}>

          {/* vLLM Server Flags */}
          <div style={sectionStyle}>vLLM Server Flags</div>

          <label style={labelStyle}>dtype</label>
          <select value={dtype} onChange={e => setDtype(e.target.value)} style={inputStyle}>
            {["auto", "bfloat16", "float16", "fp8"].map(v => <option key={v} value={v}>{v}</option>)}
          </select>

          <label style={labelStyle}>max-model-len (để trống = không set)</label>
          <input type="number" min={512} value={maxModelLen}
            onChange={e => setMaxModelLen(e.target.value)} placeholder="e.g. 8192"
            style={inputStyle} />

          <label style={labelStyle}>kv-cache-dtype</label>
          <select value={kvCacheDtype} onChange={e => setKvCacheDtype(e.target.value)} style={inputStyle}>
            {["auto", "fp8"].map(v => <option key={v} value={v}>{v}</option>)}
          </select>

          <div style={checkboxRowStyle}>
            <input type="checkbox" id="prefixCache" checked={enablePrefixCaching}
              onChange={e => setEnablePrefixCaching(e.target.checked)} />
            <label htmlFor="prefixCache" style={{ fontSize: 12, cursor: "pointer" }}>
              --enable-prefix-caching
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(Always enable for RAG/chat)</span>
            </label>
          </div>
          <div style={checkboxRowStyle}>
            <input type="checkbox" id="enforceEager" checked={enforceEager}
              onChange={e => setEnforceEager(e.target.checked)} />
            <label htmlFor="enforceEager" style={{ fontSize: 12, cursor: "pointer" }}>
              --enforce-eager
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(Tắt CUDA graphs — debug only)</span>
            </label>
          </div>
          <div style={checkboxRowStyle}>
            <input type="checkbox" id="chunkedPrefill" checked={enableChunkedPrefill}
              onChange={e => setEnableChunkedPrefill(e.target.checked)} />
            <label htmlFor="chunkedPrefill" style={{ fontSize: 12, cursor: "pointer" }}>
              --enable-chunked-prefill
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(Multi-tenant stability)</span>
            </label>
          </div>
          <div style={checkboxRowStyle}>
            <input type="checkbox" id="trustRemoteCode" checked={trustRemoteCode}
              onChange={e => setTrustRemoteCode(e.target.checked)} />
            <label htmlFor="trustRemoteCode" style={{ fontSize: 12, cursor: "pointer" }}>
              --trust-remote-code
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(Required for custom models)</span>
            </label>
          </div>
          <div style={checkboxRowStyle}>
            <input type="checkbox" id="languageModelOnly" checked={languageModelOnly}
              onChange={e => setLanguageModelOnly(e.target.checked)} />
            <label htmlFor="languageModelOnly" style={{ fontSize: 12, cursor: "pointer" }}>
              --language-model-only
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(Skip chat template, pure LM)</span>
            </label>
          </div>
          <div style={checkboxRowStyle}>
            <input type="checkbox" id="expertParallel" checked={enableExpertParallel}
              onChange={e => setEnableExpertParallel(e.target.checked)} />
            <label htmlFor="expertParallel" style={{ fontSize: 12, cursor: "pointer" }}>
              --enable-expert-parallel
              <span style={{ color: "#9ca3af", marginLeft: 6 }}>(MoE expert parallelism)</span>
            </label>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
            <div>
              <label style={labelStyle}>--data-parallel-size</label>
              <input type="number" min={1} max={8} value={dataParallelSize}
                onChange={e => setDataParallelSize(e.target.value)} placeholder="e.g. 8"
                style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>--max-num-batched-tokens</label>
              <input type="number" min={256} value={maxNumBatchedTokens}
                onChange={e => setMaxNumBatchedTokens(e.target.value)} placeholder="e.g. 4352"
                style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>--performance-mode</label>
              <select value={performanceMode} onChange={e => setPerformanceMode(e.target.value)} style={inputStyle}>
                <option value="">— none —</option>
                <option value="throughput">throughput</option>
                <option value="latency">latency</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>--reasoning-parser</label>
              <input type="text" value={reasoningParser}
                onChange={e => setReasoningParser(e.target.value)} placeholder="e.g. qwen3, deepseek_r1"
                style={inputStyle} />
            </div>
          </div>

          {/* Environment Variables */}
          <div style={sectionStyle}>Environment Variables</div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
            <div>
              <label style={labelStyle}>VLLM_USE_DEEP_GEMM</label>
              <select value={vllmUseDeepGemm} onChange={e => setVllmUseDeepGemm(e.target.value)} style={inputStyle}>
                <option value="">— default —</option>
                <option value="1">1 (enable)</option>
                <option value="0">0 (disable)</option>
              </select>
            </div>
            <div>
              <label style={labelStyle}>VLLM_USE_FLASHINFER_MOE_FP8</label>
              <select value={vllmUseFlashinferMoeFp8} onChange={e => setVllmUseFlashinferMoeFp8(e.target.value)} style={inputStyle}>
                <option value="">— default —</option>
                <option value="1">1 (enable)</option>
                <option value="0">0 (disable)</option>
              </select>
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={labelStyle}>VLLM_ALL2ALL_BACKEND</label>
              <input type="text" value={vllmAll2AllBackend}
                onChange={e => setVllmAll2AllBackend(e.target.value)}
                placeholder="e.g. deepep_high_throughput"
                style={inputStyle} />
            </div>
          </div>

          {/* Sweep Parameters */}
          <div style={sectionStyle}>Sweep Parameters</div>

          <label style={{ ...labelStyle, marginTop: 6 }}>TP sizes (none = auto từ GPU count)</label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 4 }}>
            {TP_OPTIONS.map(tp => (
              <span key={tp} style={chipStyle(customTpSizes.includes(tp))}
                onClick={() => toggleTpSize(tp)}>
                TP={tp}
              </span>
            ))}
          </div>

          <label style={{ ...labelStyle, marginTop: 10 }}>GPU mem utils (none = dùng slider)</label>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 4 }}>
            {MEM_UTIL_OPTIONS.map(v => (
              <span key={v} style={chipStyle(advMemUtils.includes(v))}
                onClick={() => toggleMemUtil(v)}>
                {(v * 100).toFixed(0)}%
              </span>
            ))}
          </div>

          <label style={{ ...labelStyle, marginTop: 10 }}>max_num_seqs</label>
          <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 4 }}>
            <label style={{ fontSize: 12, cursor: "pointer" }}>
              <input type="radio" name="maxNumSeqsMode" value="auto"
                checked={maxNumSeqsMode === "auto"}
                onChange={() => setMaxNumSeqsMode("auto")} style={{ marginRight: 4 }} />
              auto
            </label>
            <label style={{ fontSize: 12, cursor: "pointer" }}>
              <input type="radio" name="maxNumSeqsMode" value="custom"
                checked={maxNumSeqsMode === "custom"}
                onChange={() => setMaxNumSeqsMode("custom")} style={{ marginRight: 4 }} />
              custom:
            </label>
            <input type="text" value={maxNumSeqsCustom}
              onChange={e => setMaxNumSeqsCustom(e.target.value)}
              disabled={maxNumSeqsMode !== "custom"}
              style={{ ...inputStyle, width: 140, opacity: maxNumSeqsMode !== "custom" ? 0.4 : 1 }}
              placeholder="8,16,32,64" />
          </div>

          {/* Benchmark Workload */}
          <div style={sectionStyle}>Benchmark Workload</div>

          <label style={labelStyle}>Concurrent users (comma-separated)</label>
          <input type="text" value={concurrentUsers}
            onChange={e => setConcurrentUsers(e.target.value)}
            style={inputStyle} placeholder="1,4,8,16" />

          <label style={labelStyle}>Timeout per run (seconds)</label>
          <input type="number" min={60} max={3600} value={timeoutPerRun}
            onChange={e => setTimeoutPerRun(parseInt(e.target.value))} style={inputStyle} />

          {/* Sampling Params */}
          <div style={sectionStyle}>Sampling Params</div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0 10px" }}>
            <div>
              <label style={labelStyle}>temperature</label>
              <input type="number" min={0} max={2} step={0.1} value={temperature}
                onChange={e => setTemperature(parseFloat(e.target.value))} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>top_p</label>
              <input type="number" min={0} max={1} step={0.05} value={topP}
                onChange={e => setTopP(parseFloat(e.target.value))} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>top_k <span style={{ color: "#9ca3af" }}>(-1=off)</span></label>
              <input type="number" min={-1} step={1} value={topK}
                onChange={e => setTopK(parseInt(e.target.value))} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>n <span style={{ color: "#9ca3af" }}>(sequences)</span></label>
              <input type="number" min={1} max={8} step={1} value={nSeqs}
                onChange={e => setNSeqs(parseInt(e.target.value))} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>presence_penalty</label>
              <input type="number" min={-2} max={2} step={0.1} value={presencePenalty}
                onChange={e => setPresencePenalty(parseFloat(e.target.value))} style={inputStyle} />
            </div>
            <div>
              <label style={labelStyle}>frequency_penalty</label>
              <input type="number" min={-2} max={2} step={0.1} value={frequencyPenalty}
                onChange={e => setFrequencyPenalty(parseFloat(e.target.value))} style={inputStyle} />
            </div>
            <div style={{ gridColumn: "1 / -1" }}>
              <label style={labelStyle}>repetition_penalty</label>
              <input type="number" min={0.5} max={2} step={0.05} value={repetitionPenalty}
                onChange={e => setRepetitionPenalty(parseFloat(e.target.value))} style={inputStyle} />
            </div>
          </div>
        </div>
      )}

      {error && (
        <div style={{ marginTop: 8, padding: "8px 10px", background: "#fee2e2", border: "1px solid #fca5a5", borderRadius: 4, fontSize: 12, color: "#dc2626" }}>
          {error}
        </div>
      )}

      {/* Start Sweep */}
      <button
        onClick={handleStartSweep}
        disabled={!canRun || sweepLoading}
        style={{
          marginTop: 10, width: "100%", padding: "9px 0",
          background: canRun && !sweepLoading ? "#2563eb" : "#93c5fd",
          border: "none", borderRadius: 4,
          fontSize: 13, fontWeight: 700, color: "#fff",
          cursor: canRun && !sweepLoading ? "pointer" : "not-allowed",
        }}
      >
        {sweepLoading ? "Đang khởi động..." : "▶ Bắt đầu Sweep"}
      </button>
    </div>
  );
}
