/**
 * API client cho vLLM Auto-Tuner backend.
 * REST endpoints + WebSocket connection.
 */

const API_BASE = "";  // Same origin (proxied in dev, trực tiếp trong prod)

async function apiFetch(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

// ── Models ────────────────────────────────────────────────────────────────────

export const fetchModels = () =>
  apiFetch("/api/models");

export const fetchModel = (name) =>
  apiFetch(`/api/models/${encodeURIComponent(name)}`);

// ── GPUs ──────────────────────────────────────────────────────────────────────

export const fetchGPUs = () =>
  apiFetch("/api/gpus");

// ── Heuristic ─────────────────────────────────────────────────────────────────

export const computeHeuristic = (body) =>
  apiFetch("/api/heuristic", {
    method: "POST",
    body: JSON.stringify(body),
  });

// ── Sweep ─────────────────────────────────────────────────────────────────────

export const startSweep = (body) =>
  apiFetch("/api/sweep", {
    method: "POST",
    body: JSON.stringify(body),
  });

export const getSweepStatus = (sweepId) =>
  apiFetch(`/api/sweep/${sweepId}`);

// ── Runs ──────────────────────────────────────────────────────────────────────

export const fetchRuns = (params = {}) => {
  const qs = new URLSearchParams(
    Object.fromEntries(Object.entries(params).filter(([, v]) => v != null))
  ).toString();
  return apiFetch(`/api/runs${qs ? "?" + qs : ""}`);
};

export const fetchRunDetail = (runId) =>
  apiFetch(`/api/runs/${runId}`);

export const deleteRun = (runId) =>
  apiFetch(`/api/runs/${runId}`, { method: "DELETE" });

// ── Export ────────────────────────────────────────────────────────────────────

export const exportExcel = (modelName) => {
  const qs = modelName ? `?model_name=${encodeURIComponent(modelName)}` : "";
  window.open(`${API_BASE}/api/export${qs}`, "_blank");
};

// ── WebSocket ─────────────────────────────────────────────────────────────────

/**
 * Kết nối WebSocket, nhận real-time updates từ sweep.
 * @param {function} onMessage - callback(event, data)
 * @returns {WebSocket} instance để đóng khi cần
 */
export function connectWebSocket(onMessage) {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Trong dev: connect đến proxy port, trong prod: same host
  const host = window.location.hostname === "localhost"
    ? "localhost:9100"
    : window.location.host;
  const ws = new WebSocket(`${protocol}//${host}/ws`);

  ws.onopen = () => console.log("WebSocket connected");
  ws.onclose = () => console.log("WebSocket disconnected");
  ws.onerror = (e) => console.error("WebSocket error:", e);

  ws.onmessage = (e) => {
    try {
      const { event, data } = JSON.parse(e.data);
      onMessage(event, data);
    } catch (err) {
      console.warn("WS parse error:", err);
    }
  };

  // Keepalive ping mỗi 30s
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send("ping");
    }
  }, 30000);

  ws.addEventListener("close", () => clearInterval(pingInterval));

  return ws;
}
