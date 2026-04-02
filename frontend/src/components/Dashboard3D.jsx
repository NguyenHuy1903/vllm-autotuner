/**
 * Dashboard3D — Plotly scatter3d với:
 *   - 3 trục: X=max_num_seqs, Y=tensor_parallel_size, Z=metric (throughput/TTFT/latency)
 *   - Click điểm → DetailPopup
 *   - Projection buttons XY/YZ/XZ → xoay camera về góc 2D + render heatmap
 *   - Z-axis selector để đổi metric
 */
import React, { useState, useCallback, useMemo } from "react";
import Plot from "react-plotly.js";

const Z_METRICS = [
  { key: "throughput_tok_s", label: "Throughput (tok/s)", higher_is_better: true },
  { key: "ttft_ms", label: "TTFT (ms)", higher_is_better: false },
  { key: "avg_latency_ms", label: "Avg Latency (ms)", higher_is_better: false },
  { key: "p99_latency_ms", label: "P99 Latency (ms)", higher_is_better: false },
];

// Camera positions để chiếu 2D
const PROJECTION_CAMERAS = {
  "3D": { eye: { x: 1.5, y: 1.5, z: 1.2 }, up: { x: 0, y: 0, z: 1 } },
  "XY": { eye: { x: 0, y: 0, z: 2.5 }, up: { x: 0, y: 1, z: 0 } },  // nhìn từ trên xuống
  "XZ": { eye: { x: 0, y: 2.5, z: 0 }, up: { x: 0, y: 0, z: 1 } },  // nhìn từ phía Y
  "YZ": { eye: { x: 2.5, y: 0, z: 0 }, up: { x: 0, y: 0, z: 1 } },  // nhìn từ phía X
};

export default function Dashboard3D({ runs, onClickRun }) {
  const [zMetric, setZMetric] = useState(Z_METRICS[0]);
  const [projection, setProjection] = useState("3D");

  const selectedZKey = zMetric.key;

  // Tách runs thành success và failed
  const { successRuns, failedRuns } = useMemo(() => {
    const successRuns = runs.filter(r => r.status === "success" && r[selectedZKey] != null && r[selectedZKey] > 0);
    const failedRuns = runs.filter(r => r.status !== "success");
    return { successRuns, failedRuns };
  }, [runs, selectedZKey]);

  // Tạo hover text
  const makeHoverText = useCallback((run) => {
    const z = run[selectedZKey];
    return [
      `<b>${run.model_name}</b>`,
      `TP=${run.tensor_parallel_size} · Seqs=${run.max_num_seqs}`,
      `GPU Util=${(run.gpu_memory_utilization * 100).toFixed(0)}%`,
      z != null ? `${zMetric.label}: ${z.toFixed(2)}` : "",
      `Status: ${run.status}`,
    ].filter(Boolean).join("<br>");
  }, [selectedZKey, zMetric.label]);

  // Trace thành công — màu theo throughput
  const traceSuccess = {
    type: "scatter3d",
    mode: "markers",
    name: "Success",
    x: successRuns.map(r => r.max_num_seqs),
    y: successRuns.map(r => r.tensor_parallel_size),
    z: successRuns.map(r => r[selectedZKey] || 0),
    text: successRuns.map(makeHoverText),
    hoverinfo: "text",
    customdata: successRuns.map(r => r.run_id),
    marker: {
      size: 7,
      color: successRuns.map(r => r[selectedZKey] || 0),
      colorscale: zMetric.higher_is_better ? "Viridis" : "RdYlGn_r",
      colorbar: {
        title: { text: zMetric.label, side: "right" },
        thickness: 12,
        len: 0.6,
      },
      opacity: 0.85,
      line: { color: "#fff", width: 0.5 },
    },
  };

  // Trace thất bại — đỏ, z=0
  const traceFailed = {
    type: "scatter3d",
    mode: "markers",
    name: "Failed",
    x: failedRuns.map(r => r.max_num_seqs || 0),
    y: failedRuns.map(r => r.tensor_parallel_size || 0),
    z: failedRuns.map(() => 0),
    text: failedRuns.map(r => `<b>FAILED</b><br>${r.model_name}<br>TP=${r.tensor_parallel_size} Seqs=${r.max_num_seqs}<br>${r.error_class || ""}`),
    hoverinfo: "text",
    customdata: failedRuns.map(r => r.run_id),
    marker: {
      size: 6,
      color: "#ef4444",
      symbol: "x",
      opacity: 0.7,
    },
  };

  const layout = {
    scene: {
      xaxis: { title: { text: "max_num_seqs", font: { size: 11 } }, gridcolor: "#e5e7eb" },
      yaxis: { title: { text: "TP Size", font: { size: 11 } }, gridcolor: "#e5e7eb" },
      zaxis: { title: { text: zMetric.label, font: { size: 11 } }, gridcolor: "#e5e7eb" },
      camera: PROJECTION_CAMERAS[projection],
      bgcolor: "#fafafa",
    },
    margin: { l: 0, r: 10, b: 0, t: 30 },
    legend: {
      x: 0.01, y: 0.99,
      bgcolor: "rgba(255,255,255,0.8)",
      bordercolor: "#e5e7eb",
      borderwidth: 1,
      font: { size: 11 },
    },
    paper_bgcolor: "#fff",
    hoverlabel: {
      bgcolor: "#1f2937",
      font: { color: "#fff", size: 12, family: "monospace" },
      bordercolor: "#374151",
    },
  };

  // Click handler
  function handleClick(data) {
    if (!data?.points?.length) return;
    const pt = data.points[0];
    const runId = pt.customdata;
    if (runId) {
      onClickRun && onClickRun(runId);
    }
  }

  // Heatmap 2D khi ở projection mode
  const showHeatmap = projection !== "3D" && successRuns.length >= 4;
  const heatmapData = useMemo(() => {
    if (!showHeatmap) return null;
    // Tạo pivot table: rows = Y axis, cols = X axis, values = Z
    const xVals = [...new Set(successRuns.map(r => r.max_num_seqs))].sort((a, b) => a - b);
    const yVals = [...new Set(successRuns.map(r => r.tensor_parallel_size))].sort((a, b) => a - b);
    const zMatrix = yVals.map(y =>
      xVals.map(x => {
        const run = successRuns.find(r => r.max_num_seqs === x && r.tensor_parallel_size === y);
        return run ? run[selectedZKey] : null;
      })
    );
    return { xVals, yVals, zMatrix };
  }, [showHeatmap, successRuns, selectedZKey]);

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      {/* Controls */}
      <div style={{ display: "flex", gap: 8, alignItems: "center", padding: "8px 0", flexWrap: "wrap" }}>
        {/* Z-axis selector */}
        <div>
          <span style={{ fontSize: 11, color: "#6b7280", marginRight: 6 }}>Trục Z:</span>
          <select
            value={zMetric.key}
            onChange={e => setZMetric(Z_METRICS.find(m => m.key === e.target.value))}
            style={{ fontSize: 12, padding: "3px 6px", border: "1px solid #d1d5db", borderRadius: 4, fontFamily: "monospace" }}
          >
            {Z_METRICS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
          </select>
        </div>

        {/* Projection buttons */}
        <div style={{ display: "flex", gap: 4 }}>
          {["3D", "XY", "XZ", "YZ"].map(proj => (
            <button
              key={proj}
              onClick={() => setProjection(proj)}
              style={{
                padding: "4px 10px",
                fontSize: 11, fontFamily: "monospace", fontWeight: 600,
                border: `1px solid ${projection === proj ? "#2563eb" : "#d1d5db"}`,
                borderRadius: 3,
                background: projection === proj ? "#eff6ff" : "#fff",
                color: projection === proj ? "#2563eb" : "#6b7280",
                cursor: "pointer",
              }}
            >
              {proj}
            </button>
          ))}
        </div>

        {/* Stats */}
        <div style={{ fontSize: 11, color: "#6b7280", marginLeft: "auto" }}>
          {successRuns.length} thành công · {failedRuns.length} thất bại
          {successRuns.length > 0 && (
            <span style={{ color: "#16a34a", marginLeft: 8, fontFamily: "monospace" }}>
              Max: {Math.max(...successRuns.map(r => r[selectedZKey] || 0)).toFixed(1)}
            </span>
          )}
        </div>
      </div>

      {/* Empty state */}
      {runs.length === 0 && (
        <div style={{
          flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
          color: "#9ca3af", fontSize: 13, border: "1px dashed #e5e7eb", borderRadius: 8,
        }}>
          Chưa có dữ liệu — bắt đầu sweep để xem 3D chart
        </div>
      )}

      {/* 3D Scatter Plot */}
      {runs.length > 0 && (
        <div style={{ flex: 1 }}>
          <Plot
            data={[traceSuccess, traceFailed]}
            layout={layout}
            onClick={handleClick}
            config={{
              displayModeBar: true,
              modeBarButtonsToRemove: ["sendDataToCloud"],
              responsive: true,
            }}
            style={{ width: "100%", height: "100%", minHeight: 420 }}
            useResizeHandler
          />
        </div>
      )}

      {/* Heatmap 2D khi ở projection mode */}
      {showHeatmap && heatmapData && (
        <div style={{ marginTop: 16 }}>
          <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 4 }}>
            Heatmap 2D ({projection === "XY" ? "max_num_seqs vs TP" : projection === "XZ" ? "max_num_seqs vs " + zMetric.label : "TP vs " + zMetric.label})
          </div>
          <Plot
            data={[{
              type: "heatmap",
              x: heatmapData.xVals,
              y: heatmapData.yVals,
              z: heatmapData.zMatrix,
              colorscale: zMetric.higher_is_better ? "Viridis" : "RdYlGn_r",
              xgap: 2, ygap: 2,
              colorbar: { thickness: 12, len: 0.8 },
              hovertemplate: `Seqs=%{x}<br>TP=%{y}<br>${zMetric.label}=%{z:.2f}<extra></extra>`,
            }]}
            layout={{
              xaxis: { title: "max_num_seqs", tickfont: { size: 11 } },
              yaxis: { title: "TP Size", tickfont: { size: 11 } },
              margin: { l: 50, r: 80, t: 10, b: 40 },
              paper_bgcolor: "#fff",
              plot_bgcolor: "#fafafa",
            }}
            config={{ responsive: true, displayModeBar: false }}
            style={{ width: "100%", height: 220 }}
            useResizeHandler
          />
        </div>
      )}
    </div>
  );
}
