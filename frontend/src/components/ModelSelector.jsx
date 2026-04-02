/**
 * ModelSelector — Dropdown chọn model, hiển thị thông tin params + precision.
 */
import React from "react";

const PRECISION_COLORS = {
  fp8:        "#16a34a",
  bfloat16:   "#2563eb",
  float16:    "#2563eb",
  "gptq-w4":  "#d97706",
  "gptq-w8":  "#7c3aed",
  "bnb-nf4":  "#d97706",
  "bnb-fp4":  "#d97706",
  "hqq-4bit": "#d97706",
  nvfp4:      "#d97706",
};

function PrecisionBadge({ precision }) {
  const color = PRECISION_COLORS[precision] || "#6b7280";
  return (
    <span style={{
      display: "inline-block",
      padding: "1px 7px",
      borderRadius: 3,
      fontSize: 10,
      fontFamily: "monospace",
      fontWeight: 600,
      textTransform: "uppercase",
      letterSpacing: "0.06em",
      background: color + "22",
      color,
      border: `1px solid ${color}44`,
      marginLeft: 6,
    }}>
      {precision}
    </span>
  );
}

export default function ModelSelector({ models, selectedModel, onSelect }) {
  const model = models.find(m => m.name === selectedModel);

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontSize: 11, fontWeight: 600, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.1em", display: "block", marginBottom: 4 }}>
        Model
      </label>

      <select
        value={selectedModel || ""}
        onChange={e => onSelect(e.target.value)}
        style={{
          width: "100%",
          padding: "7px 10px",
          border: "1px solid #d1d5db",
          borderRadius: 4,
          fontSize: 12,
          background: "#fff",
          fontFamily: "monospace",
          cursor: "pointer",
        }}
      >
        <option value="">-- Chọn model --</option>
        {models.map(m => (
          <option key={m.name} value={m.name}>
            {m.name}
          </option>
        ))}
      </select>

      {/* Chi tiết model đã chọn */}
      {model && (
        <div style={{
          marginTop: 8,
          padding: "10px 12px",
          background: "#f9fafb",
          border: "1px solid #e5e7eb",
          borderRadius: 4,
          fontSize: 12,
          lineHeight: 1.8,
        }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "4px 16px" }}>
            <span>
              <span style={{ color: "#9ca3af" }}>Params: </span>
              <strong style={{ fontFamily: "monospace" }}>
                {model.total_params_b.toFixed(1)}B total
              </strong>
              {model.is_moe && (
                <span style={{ color: "#6b7280", fontFamily: "monospace", fontSize: 11 }}>
                  {" / "}{model.active_params_b.toFixed(1)}B active
                </span>
              )}
            </span>
            <span>
              <span style={{ color: "#9ca3af" }}>Precision: </span>
              <PrecisionBadge precision={model.precision} />
            </span>
            <span>
              <span style={{ color: "#9ca3af" }}>Disk: </span>
              <span style={{ fontFamily: "monospace" }}>{model.disk_size_gb.toFixed(1)} GB</span>
            </span>
            {model.is_moe && (
              <span>
                <span style={{ color: "#9ca3af" }}>MoE: </span>
                <span style={{ fontFamily: "monospace", color: "#7c3aed" }}>
                  {model.num_experts}E / {model.num_experts_per_tok}A
                </span>
              </span>
            )}
            {model.is_multimodal && (
              <span style={{ color: "#0e7490", fontFamily: "monospace", fontSize: 11 }}>
                [Multimodal]
              </span>
            )}
          </div>
          <div style={{ color: "#9ca3af", fontSize: 11, marginTop: 4, fontFamily: "monospace" }}>
            {model.architectures[0] || model.model_type}
            {" · "}
            {model.num_hidden_layers}L × {model.hidden_size}H × {model.head_dim}D
          </div>
        </div>
      )}
    </div>
  );
}
