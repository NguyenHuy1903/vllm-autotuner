/**
 * DetailPopup — Modal hiển thị chi tiết khi click vào một điểm trên 3D chart.
 * Hiển thị: config, VRAM breakdown, metrics, error log.
 */
import React from "react";
import { exportExcel } from "../api";

const STATUS_COLORS = {
  success: { bg: "#dcfce7", color: "#16a34a", label: "SUCCESS" },
  failed: { bg: "#fee2e2", color: "#dc2626", label: "FAILED" },
  timeout: { bg: "#fef3c7", color: "#d97706", label: "TIMEOUT" },
  pending: { bg: "#f3f4f6", color: "#6b7280", label: "PENDING" },
};

function MetricRow({ label, value, unit = "" }) {
  if (value == null || value === 0) return null;
  return (
    <div style={{ display: "flex", justifyContent: "space-between", padding: "4px 0", borderBottom: "1px solid #f3f4f6" }}>
      <span style={{ color: "#6b7280", fontSize: 12 }}>{label}</span>
      <span style={{ fontFamily: "monospace", fontWeight: 600, fontSize: 12 }}>
        {typeof value === "number" ? value.toFixed(2) : value}{unit}
      </span>
    </div>
  );
}

function CodeBlock({ children }) {
  return (
    <pre style={{
      background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 4,
      padding: "10px 12px", fontSize: 11, fontFamily: "monospace",
      lineHeight: 1.6, overflowX: "auto", maxHeight: 200, overflowY: "auto",
      whiteSpace: "pre-wrap", wordBreak: "break-word",
    }}>
      {children}
    </pre>
  );
}

export default function DetailPopup({ run, onClose }) {
  if (!run) return null;

  const statusInfo = STATUS_COLORS[run.status] || STATUS_COLORS.pending;
  let config = {};
  let metrics = null;
  let heuristic = null;

  try { config = JSON.parse(run.config_json || "{}"); } catch (e) {}
  try { metrics = JSON.parse(run.metrics_json || "null"); } catch (e) {}
  try { heuristic = JSON.parse(run.heuristic_json || "null"); } catch (e) {}

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        background: "rgba(0,0,0,0.5)",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div style={{
        background: "#fff", borderRadius: 8, padding: 24,
        width: "min(640px, 90vw)", maxHeight: "85vh",
        overflowY: "auto", boxShadow: "0 20px 60px rgba(0,0,0,0.2)",
      }}>
        {/* Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 16 }}>
          <div>
            <div style={{ fontFamily: "monospace", fontSize: 13, fontWeight: 700, color: "#111827" }}>
              Run #{run.run_id}
            </div>
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 2 }}>
              {run.model_name} · {run.created_at?.slice(0, 19).replace("T", " ")}
            </div>
          </div>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <span style={{
              padding: "3px 10px", borderRadius: 12, fontSize: 11, fontWeight: 700,
              background: statusInfo.bg, color: statusInfo.color,
            }}>
              {statusInfo.label}
            </span>
            <button onClick={onClose} style={{
              background: "none", border: "none", fontSize: 18,
              cursor: "pointer", color: "#6b7280", lineHeight: 1,
            }}>×</button>
          </div>
        </div>

        {/* Config */}
        <div style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: "#374151", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Cấu hình
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
            <MetricRow label="Tensor Parallel" value={run.tensor_parallel_size} />
            <MetricRow label="Max Num Seqs" value={run.max_num_seqs} />
            <MetricRow label="GPU Mem Util" value={run.gpu_memory_utilization} />
            <MetricRow label="GPU IDs" value={run.gpu_ids} />
            <MetricRow label="Docker Image" value={run.docker_image} />
            <MetricRow label="Startup (s)" value={run.startup_time_s} unit="s" />
          </div>
        </div>

        {/* VRAM Heuristic */}
        {heuristic && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#374151", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              VRAM Budget (Dự tính)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
              <MetricRow label="Weights/GPU" value={heuristic.vram_weights_per_gpu_gb} unit=" GB" />
              <MetricRow label="Free/GPU" value={heuristic.vram_free_per_gpu_gb} unit=" GB" />
              <MetricRow label="Max KV tokens" value={heuristic.max_kv_tokens} />
              <MetricRow label="Đề xuất max_num_seqs" value={heuristic.suggested_max_num_seqs} />
            </div>
          </div>
        )}

        {/* Metrics (nếu thành công) */}
        {metrics && run.status === "success" && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: "#374151", marginBottom: 6, textTransform: "uppercase", letterSpacing: "0.08em" }}>
              Kết quả Benchmark
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "4px 16px" }}>
              <MetricRow label="Throughput" value={metrics.throughput_tok_s} unit=" tok/s" />
              <MetricRow label="TTFT" value={metrics.ttft_ms} unit=" ms" />
              <MetricRow label="Avg Latency" value={metrics.avg_latency_ms} unit=" ms" />
              <MetricRow label="P99 Latency" value={metrics.p99_latency_ms} unit=" ms" />
              <MetricRow label="Max Concurrent" value={metrics.max_concurrent_tested} />
              <MetricRow label="Startup" value={metrics.startup_time_s} unit="s" />
            </div>

            {/* Concurrency details */}
            {metrics.concurrency_details?.length > 0 && (
              <div style={{ marginTop: 8 }}>
                <div style={{ fontSize: 10, color: "#6b7280", marginBottom: 4 }}>Latency theo concurrent users:</div>
                <table style={{ width: "100%", fontSize: 11, fontFamily: "monospace", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ background: "#f9fafb" }}>
                      {["Concurrency", "Throughput", "TTFT", "Avg Lat", "P99"].map(h => (
                        <th key={h} style={{ padding: "4px 8px", textAlign: "right", color: "#6b7280", borderBottom: "1px solid #e5e7eb", fontSize: 10 }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.concurrency_details.map(d => (
                      <tr key={d.concurrent_users}>
                        <td style={{ padding: "4px 8px", textAlign: "right" }}>{d.concurrent_users}</td>
                        <td style={{ padding: "4px 8px", textAlign: "right" }}>{d.throughput_tok_s?.toFixed(0)}</td>
                        <td style={{ padding: "4px 8px", textAlign: "right" }}>{d.ttft_ms?.toFixed(1)}</td>
                        <td style={{ padding: "4px 8px", textAlign: "right" }}>{d.avg_latency_ms?.toFixed(1)}</td>
                        <td style={{ padding: "4px 8px", textAlign: "right" }}>{d.p99_latency_ms?.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Error log (nếu thất bại) */}
        {run.error_class && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: "#dc2626", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                Lỗi: {run.error_class}
              </div>
            </div>
            {run.error_log && <CodeBlock>{run.error_log}</CodeBlock>}
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", gap: 8, justifyContent: "flex-end", marginTop: 4 }}>
          <button
            onClick={() => exportExcel(run.model_name)}
            style={{
              padding: "7px 14px", background: "#f3f4f6", border: "1px solid #d1d5db",
              borderRadius: 4, fontSize: 12, cursor: "pointer", fontWeight: 500,
            }}
          >
            📊 Xuất Excel
          </button>
          <button
            onClick={onClose}
            style={{
              padding: "7px 14px", background: "#2563eb", border: "none",
              borderRadius: 4, fontSize: 12, cursor: "pointer", fontWeight: 500, color: "#fff",
            }}
          >
            Đóng
          </button>
        </div>
      </div>
    </div>
  );
}
