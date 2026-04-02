/**
 * GPUSelector — Grid checkbox chọn GPU, hiển thị VRAM usage real-time.
 */
import React from "react";

function GPUCard({ gpu, selected, onToggle }) {
  const usedPct = gpu.memory_total_gb > 0
    ? (gpu.memory_used_gb / gpu.memory_total_gb) * 100
    : 0;
  const free = gpu.memory_free_gb.toFixed(1);

  // Màu bar theo mức độ sử dụng
  const barColor = usedPct > 80 ? "#dc2626" : usedPct > 60 ? "#d97706" : "#16a34a";

  return (
    <div
      onClick={onToggle}
      style={{
        padding: "8px 10px",
        border: `2px solid ${selected ? "#2563eb" : "#e5e7eb"}`,
        borderRadius: 4,
        cursor: "pointer",
        background: selected ? "#eff6ff" : "#fff",
        transition: "border-color 0.15s, background 0.15s",
        userSelect: "none",
        opacity: gpu.in_use_by_autotuner ? 0.6 : 1,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <span style={{ fontFamily: "monospace", fontWeight: 700, fontSize: 12, color: selected ? "#2563eb" : "#374151" }}>
          GPU {gpu.index}
        </span>
        {gpu.in_use_by_autotuner && (
          <span style={{ fontSize: 9, color: "#d97706", fontWeight: 600 }}>BUSY</span>
        )}
      </div>
      {/* VRAM bar */}
      <div style={{ height: 4, background: "#f3f4f6", borderRadius: 2, overflow: "hidden", marginBottom: 3 }}>
        <div style={{ height: "100%", width: `${usedPct}%`, background: barColor, borderRadius: 2, transition: "width 0.5s" }} />
      </div>
      <div style={{ fontSize: 10, color: "#6b7280", fontFamily: "monospace" }}>
        {free} GB free / {gpu.memory_total_gb.toFixed(0)} GB
      </div>
    </div>
  );
}

export default function GPUSelector({ gpus, selectedGPUs, onToggle }) {
  const totalFreeGb = selectedGPUs
    .map(id => gpus.find(g => g.index === id))
    .filter(Boolean)
    .reduce((sum, g) => sum + g.memory_free_gb, 0);

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", marginBottom: 4 }}>
        GPUs ({selectedGPUs.length} chọn · {totalFreeGb.toFixed(0)} GB free)
      </label>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 6 }}>
        {gpus.map(gpu => (
          <GPUCard
            key={gpu.index}
            gpu={gpu}
            selected={selectedGPUs.includes(gpu.index)}
            onToggle={() => onToggle(gpu.index)}
          />
        ))}
      </div>

      {selectedGPUs.length === 0 && (
        <div style={{ fontSize: 11, color: "#ef4444", marginTop: 4 }}>
          Chọn ít nhất 1 GPU
        </div>
      )}
    </div>
  );
}
