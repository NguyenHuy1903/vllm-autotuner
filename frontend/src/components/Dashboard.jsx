/**
 * Dashboard.jsx — Main dashboard component (extracted from App.jsx)
 */
import React, { useCallback, useEffect, useRef, useState } from "react";
import { connectWebSocket, exportExcel, fetchGPUs, fetchModels, fetchRunDetail, fetchRuns } from "../api";
import ConfigPanel from "./ConfigPanel";
import Dashboard3D from "./Dashboard3D";
import DetailPopup from "./DetailPopup";
import GPUSelector from "./GPUSelector";
import ModelSelector from "./ModelSelector";

export default function Dashboard() {
  // ── State ───────────────────────────────────────────────────────────────────
  const [models, setModels] = useState([]);
  const [gpus, setGPUs] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedGPUs, setSelectedGPUs] = useState([]);
  const [runs, setRuns] = useState([]);
  const [sweepInfo, setSweepInfo] = useState(null);  // { sweep_id, total, completed, logs }
  const [detailRun, setDetailRun] = useState(null);  // Run hiển thị trong popup
  const [loadingModels, setLoadingModels] = useState(false);
  const [logs, setLogs] = useState([]);  // Real-time log messages
  const logsEndRef = useRef(null);
  const [sidebarWidth, setSidebarWidth] = useState(300);
  const isDragging = useRef(false);
  const dragStartX = useRef(0);
  const dragStartWidth = useRef(0);

  function onResizeMouseDown(e) {
    isDragging.current = true;
    dragStartX.current = e.clientX;
    dragStartWidth.current = sidebarWidth;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  }

  useEffect(() => {
    function onMouseMove(e) {
      if (!isDragging.current) return;
      const delta = e.clientX - dragStartX.current;
      const next = Math.max(220, Math.min(600, dragStartWidth.current + delta));
      setSidebarWidth(next);
    }
    function onMouseUp() {
      if (!isDragging.current) return;
      isDragging.current = false;
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    }
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);
    return () => {
      window.removeEventListener("mousemove", onMouseMove);
      window.removeEventListener("mouseup", onMouseUp);
    };
  }, []);

  // ── Load initial data ───────────────────────────────────────────────────────
  useEffect(() => {
    setLoadingModels(true);
    fetchModels().then(setModels).catch(console.error).finally(() => setLoadingModels(false));
    refreshGPUs();
    fetchRuns({ limit: 200 }).then(setRuns).catch(console.error);
  }, []);

  // Auto-refresh GPUs mỗi 10 giây
  useEffect(() => {
    const interval = setInterval(refreshGPUs, 10000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  function refreshGPUs() {
    fetchGPUs().then(setGPUs).catch(console.error);
  }

  // ── WebSocket ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const ws = connectWebSocket((event, data) => {
      switch (event) {
        case "run_started":
          addLog(`[${data.run_id}] Bắt đầu: TP=${data.config?.tensor_parallel_size} Seqs=${data.config?.max_num_seqs}`);
          break;

        case "run_progress":
          addLog(data.message);
          break;

        case "run_completed":
          addLog(
            data.status === "success"
              ? `[${data.run_id}] ✓ Success — ${data.metrics?.throughput_tok_s?.toFixed(0)} tok/s, TTFT ${data.metrics?.ttft_ms?.toFixed(0)}ms`
              : `[${data.run_id}] ✗ ${data.status} — ${data.error_class || ""}`
          );
          // Thêm run mới vào danh sách
          setRuns(prev => {
            const exists = prev.find(r => r.run_id === data.run_id);
            if (exists) return prev.map(r => r.run_id === data.run_id ? { ...r, ...flattenRun(data) } : r);
            return [flattenRun(data), ...prev];
          });
          setSweepInfo(prev => prev ? { ...prev, completed: (prev.completed || 0) + 1 } : prev);
          refreshGPUs();
          break;

        case "sweep_completed":
          addLog(`Sweep ${data.sweep_id} hoàn thành: ${data.completed}/${data.total} runs (${data.success_count} thành công)`);
          setSweepInfo(prev => prev ? { ...prev, status: "completed" } : prev);
          // Refresh full run list
          fetchRuns({ limit: 200 }).then(setRuns).catch(console.error);
          break;

        default:
          break;
      }
    });

    return () => ws.close();
  }, []);

  function addLog(msg) {
    const time = new Date().toTimeString().slice(0, 8);
    setLogs(prev => [...prev.slice(-200), `${time} ${msg}`]);
  }

  // Run từ WS event chưa có full fields → tạo flat object tương thích Dashboard3D
  function flattenRun(data) {
    return {
      run_id: data.run_id,
      model_name: data.config?.model_name || "",
      tensor_parallel_size: data.config?.tensor_parallel_size,
      max_num_seqs: data.config?.max_num_seqs,
      gpu_memory_utilization: data.config?.gpu_memory_utilization,
      gpu_ids: JSON.stringify(data.config?.gpu_ids || []),
      docker_image: data.config?.docker_image || "",
      status: data.status,
      throughput_tok_s: data.metrics?.throughput_tok_s,
      ttft_ms: data.metrics?.ttft_ms,
      avg_latency_ms: data.metrics?.avg_latency_ms,
      p99_latency_ms: data.metrics?.p99_latency_ms,
      error_class: data.error_class,
      created_at: new Date().toISOString(),
    };
  }

  // ── Handlers ────────────────────────────────────────────────────────────────
  function handleToggleGPU(gpuId) {
    setSelectedGPUs(prev =>
      prev.includes(gpuId) ? prev.filter(g => g !== gpuId) : [...prev, gpuId]
    );
  }

  function handleSweepStarted(sweepId, total) {
    setSweepInfo({ sweep_id: sweepId, total, completed: 0, status: "running" });
    addLog(`Sweep ${sweepId} bắt đầu — ${total} cấu hình cần test`);
  }

  async function handleClickRun(runId) {
    try {
      const detail = await fetchRunDetail(runId);
      setDetailRun(detail);
    } catch (e) {
      console.error("Không lấy được run detail:", e);
    }
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  // Filter runs theo model đang chọn
  const filteredRuns = selectedModel
    ? runs.filter(r => r.model_name === selectedModel)
    : runs;

  return (
    <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", background: "#f9fafb" }}>

      {/* ── Body ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>

        {/* Left sidebar */}
        <div style={{
          width: sidebarWidth, flexShrink: 0, background: "#fff",
          borderRight: "1px solid #e5e7eb", overflowY: "auto",
          padding: "16px 14px", position: "relative",
        }}>
          {loadingModels ? (
            <div style={{ color: "#9ca3af", fontSize: 13 }}>Đang quét models...</div>
          ) : (
            <ModelSelector
              models={models}
              selectedModel={selectedModel}
              onSelect={setSelectedModel}
            />
          )}

          <GPUSelector
            gpus={gpus}
            selectedGPUs={selectedGPUs}
            onToggle={handleToggleGPU}
          />

          <div style={{ height: 1, background: "#f3f4f6", margin: "12px 0" }} />

          <ConfigPanel
            selectedModel={selectedModel}
            selectedGPUs={selectedGPUs}
            onSweepStarted={handleSweepStarted}
          />
        </div>

        {/* Resize handle */}
        <div
          onMouseDown={onResizeMouseDown}
          style={{
            width: 5, flexShrink: 0, cursor: "col-resize",
            background: "transparent",
            borderRight: "2px solid transparent",
            transition: "border-color 0.15s",
          }}
          onMouseEnter={e => e.currentTarget.style.borderRightColor = "#93c5fd"}
          onMouseLeave={e => e.currentTarget.style.borderRightColor = "transparent"}
        />

        {/* Main area */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", padding: "16px 20px" }}>

          {/* Sweep progress bar */}
          {sweepInfo && (
            <div style={{
              marginBottom: 12, padding: "8px 14px",
              background: sweepInfo.status === "completed" ? "#f0fdf4" : "#eff6ff",
              border: `1px solid ${sweepInfo.status === "completed" ? "#bbf7d0" : "#bfdbfe"}`,
              borderRadius: 4, fontSize: 12,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontWeight: 600, color: "#1f2937" }}>
                  Sweep {sweepInfo.sweep_id}
                  {sweepInfo.status === "completed" ? " ✓ Hoàn thành" : " · Đang chạy..."}
                </span>
                <span style={{ fontFamily: "monospace", color: "#6b7280" }}>
                  {sweepInfo.completed}/{sweepInfo.total}
                </span>
              </div>
              <div style={{ height: 6, background: "#e5e7eb", borderRadius: 3, overflow: "hidden" }}>
                <div style={{
                  height: "100%",
                  width: `${sweepInfo.total > 0 ? (sweepInfo.completed / sweepInfo.total) * 100 : 0}%`,
                  background: sweepInfo.status === "completed" ? "#16a34a" : "#2563eb",
                  borderRadius: 3, transition: "width 0.3s",
                }} />
              </div>
            </div>
          )}

          {/* 3D Dashboard */}
          <div style={{ flex: 1, overflow: "hidden" }}>
            <Dashboard3D
              runs={filteredRuns}
              onClickRun={handleClickRun}
            />
          </div>
        </div>
      </div>

      {/* ── Bottom: Real-time logs ── */}
      <div style={{
        height: 130, background: "#1f2937", borderTop: "1px solid #374151",
        padding: "8px 16px", overflowY: "auto", flexShrink: 0,
      }}>
        <div style={{ fontSize: 10, fontWeight: 700, color: "#6b7280", marginBottom: 4, letterSpacing: "0.1em" }}>
          LOGS
        </div>
        {logs.length === 0 && (
          <div style={{ color: "#4b5563", fontSize: 12, fontFamily: "monospace" }}>
            Chưa có logs. Bắt đầu sweep để xem real-time output...
          </div>
        )}
        {logs.map((log, i) => (
          <div key={i} style={{
            fontSize: 11, fontFamily: "monospace", lineHeight: 1.7,
            color: log.includes("✓") ? "#4ade80" : log.includes("✗") ? "#f87171" : "#d1d5db",
          }}>
            {log}
          </div>
        ))}
        <div ref={logsEndRef} />
      </div>

      {/* Detail Popup */}
      {detailRun && (
        <DetailPopup run={detailRun} onClose={() => setDetailRun(null)} />
      )}
    </div>
  );
}
