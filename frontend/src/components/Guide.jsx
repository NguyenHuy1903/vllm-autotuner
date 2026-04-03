/**
 * Guide.jsx — Guide page explaining vLLM Auto-Tuner parameters
 */
import React from "react";

export default function Guide() {
  const styles = {
    container: {
      flex: 1,
      overflowY: "auto",
      padding: "40px",
      maxWidth: "1200px",
      margin: "0 auto",
      fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    },
    title: {
      fontSize: 32,
      fontWeight: 700,
      color: "#1f2937",
      marginBottom: 8,
    },
    subtitle: {
      fontSize: 16,
      color: "#6b7280",
      marginBottom: 32,
    },
    section: {
      marginBottom: 32,
      padding: "24px",
      background: "#f9fafb",
      borderLeft: "4px solid #16a34a",
      borderRadius: 4,
    },
    sectionTitle: {
      fontSize: 20,
      fontWeight: 700,
      color: "#16a34a",
      marginBottom: 16,
      display: "flex",
      alignItems: "center",
      gap: 8,
    },
    paramBox: {
      marginBottom: 16,
      padding: "16px",
      background: "#fff",
      borderRadius: 4,
      borderLeft: "3px solid #2563eb",
    },
    paramName: {
      fontSize: 14,
      fontWeight: 700,
      color: "#1f2937",
      fontFamily: "monospace",
      marginBottom: 4,
    },
    paramDescription: {
      fontSize: 13,
      color: "#6b7280",
      lineHeight: 1.6,
      marginBottom: 8,
    },
    paramRange: {
      fontSize: 12,
      background: "#eff6ff",
      padding: "8px 12px",
      borderRadius: 3,
      fontFamily: "monospace",
      color: "#1e40af",
    },
    tips: {
      background: "#fef3c7",
      borderLeft: "4px solid #f59e0b",
      padding: "16px",
      borderRadius: 4,
      marginTop: 16,
    },
    tipTitle: {
      fontWeight: 700,
      color: "#92400e",
      marginBottom: 8,
    },
    tipText: {
      fontSize: 13,
      color: "#78350f",
      lineHeight: 1.6,
    },
  };

  return (
    <div style={styles.container}>
      <h1 style={styles.title}>📖 vLLM Auto-Tuner Guide</h1>
      <p style={styles.subtitle}>
        Tìm hiểu chi tiết về các tham số chính để tối ưu hóa hiệu suất vLLM
      </p>

      {/* Core Configuration Parameters */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          ⚙️ Core Configuration Parameters
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>tensor_parallel_size (TP)</div>
          <div style={styles.paramDescription}>
            Số lượng GPU được sử dụng để phân tán các tensor trong mô hình. Tăng TP sẽ chia nhỏ mô hình lớn trên nhiều GPU.
          </div>
          <div style={styles.paramRange}>
            Giá trị: 1, 2, 4, 8 (tùy thuộc số GPU khả dụng)
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • TP=1: Không phân tán, mô hình phải vừa trên 1 GPU
              <br />
              • TP=4,8: Tốt cho mô hình rất lớn (70B+), nhưng đòi hỏi độ trễ giao tiếp cao
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>max_num_seqs</div>
          <div style={styles.paramDescription}>
            Số lượng request (sequence) tối đa có thể xử lý song song trong một batch.
          </div>
          <div style={styles.paramRange}>
            Giá trị: 16, 32, 64, 128, 256 (phụ thuộc GPU memory)
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • Cao hơn = throughput cao + OOM risk cao
              <br />
              • Thấp hơn = throughput thấp + ổn định hơn
              <br />
              • Cần cân bằng giữa throughtput và reliability
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>gpu_memory_utilization</div>
          <div style={styles.paramDescription}>
            Tỷ lệ phần trăm GPU memory dùng để cache KV (Key-Value) và intermediate tensors.
          </div>
          <div style={styles.paramRange}>
            Giá trị: 0.6, 0.7, 0.8, 0.9 (không khuyến khích vượt 0.95)
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • 0.9 = gần hết memory, risk OOM cao
              <br />
              • 0.7-0.8 = đối với H100 thường cho kết quả tốt nhất
              <br />
              • Tăng = throughput tốt, nhưng latency có thể tăng
            </div>
          </div>
        </div>
      </div>

      {/* Quantization & Optimization */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          🚀 Quantization & Optimization
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>quantization</div>
          <div style={styles.paramDescription}>
            Công nghệ nén mô hình để giảm memory footprint và tăng tốc độ compute.
          </div>
          <div style={styles.paramRange}>
            Giá trị: "fp8", "int4", "int8", None (no quantization)
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • fp8: Giảm ~50% memory, tốc độ cao, chất lượng tốt
              <br />
              • int4: Giảm ~75% memory, nhưng chất lượng có thể thấp
              <br />
              • None: Full precision (fp16/bf16), chất lượng tốt nhất
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>dtype</div>
          <div style={styles.paramDescription}>
            Kiểu dữ liệu cho các phép tính (computation dtype).
          </div>
          <div style={styles.paramRange}>
            Giá trị: "float16", "bfloat16", "float32"
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • bfloat16: Tốt cho H100, cân bằng tốc độ & precision
              <br />
              • float16: Tốt cho older GPUs (A100), nhưng có numeric issues
              <br />
              • float32: Chất lượng cao + compute chậm
            </div>
          </div>
        </div>
      </div>

      {/* Advanced Parameters */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          🔧 Advanced Parameters
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>enforce_eager</div>
          <div style={styles.paramDescription}>
            Buộc sử dụng eager execution thay vì graph compilation (CUDA graph).
          </div>
          <div style={styles.paramRange}>
            Giá trị: true/false
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • false (default): Dùng CUDA graph → tốc độ cao hơn ~20%
              <br />
              • true: Dùng eager mode → flexible nhưng chậm hơn
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>max_seq_len_to_capture</div>
          <div style={styles.paramDescription}>
            Chiều dài sequence tối đa được capture vào CUDA graph.
          </div>
          <div style={styles.paramRange}>
            Giá trị: 4096, 8192, 32768 (phụ thuộc model)
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • Lớn hơn = cover nhiều cases, nhưng memory overhead cao
              <br />
              • Nhỏ hơn = ít memory, nhưng một số sequences sẽ không được optimize
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>chunked_prefill</div>
          <div style={styles.paramDescription}>
            Chia nhỏ prefill phase thành các chunks để tránh OOM trên request lớn.
          </div>
          <div style={styles.paramRange}>
            Giá trị: true/false
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Lưu ý:</div>
            <div style={styles.tipText}>
              • true: An toàn hơn cho sequences dài, nhưng latency cao
              <br />
              • false: Nhanh hơn cho input bình thường
            </div>
          </div>
        </div>
      </div>

      {/* Metrics Explanation */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          📊 Performance Metrics
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>throughput_tok_s</div>
          <div style={styles.paramDescription}>
            Số token được generate mỗi giây. Chỉ số này thể hiện khả năng xử lý workload lớn.
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Giải thích:</div>
            <div style={styles.tipText}>
              • Cao hơn = tốt hơn → mục tiêu tối ưu chính
              <br />
              • Phụ thuộc: batch size, max_num_seqs, TP, quantization
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>ttft_ms (Time To First Token)</div>
          <div style={styles.paramDescription}>
            Thời gian từ khi nhận request đến khi generate token đầu tiên. Đại diện latency user-facing.
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Giải thích:</div>
            <div style={styles.tipText}>
              • Thấp hơn = tốt hơn → mục tiêu thứ hai
              <br />
              • Bị ảnh hưởng bởi input length, batch size, prefill optimization
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>avg_latency_ms</div>
          <div style={styles.paramDescription}>
            Trung bình độ trễ per-token trong decode phase.
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Giải thích:</div>
            <div style={styles.tipText}>
              • Thấp hơn = tốt hơn
              <br />
              • Phụ thuộc: attention implementation, TP, quantization
            </div>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>p99_latency_ms</div>
          <div style={styles.paramDescription}>
            Độ trễ tại percentile 99 (tail latency). Phản ánh trải nghiệm người dùng xấu nhất.
          </div>
          <div style={styles.tips}>
            <div style={styles.tipTitle}>💡 Giải thích:</div>
            <div style={styles.tipText}>
              • Dùng để đánh giá worst-case performance
              <br />
              • Bị ảnh hưởng bởi memory pressure, contention, batch variability
            </div>
          </div>
        </div>
      </div>

      {/* Best Practices */}
      <div style={styles.section}>
        <div style={styles.sectionTitle}>
          ✨ Best Practices
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>1. Xác định mục tiêu</div>
          <div style={styles.paramDescription}>
            Bạn muốn tối ưu:
            <ul style={{ marginTop: 8, marginBottom: 0, paddingLeft: 20 }}>
              <li>Throughput cao? (batch inference, offline)\n</li>
              <li>Latency thấp? (interactive, real-time)</li>
              <li>Cân bằng? (production)</li>
            </ul>
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>2. Bắt đầu từ safe defaults</div>
          <div style={styles.paramDescription}>
            • tensor_parallel_size = 1-2
            <br />
            • max_num_seqs = 32-64
            <br />
            • gpu_memory_utilization = 0.7
            <br />
            • dtype = bfloat16 (cho H100)
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>3. Sweep theo chiều</div>
          <div style={styles.paramDescription}>
            Không thay tất cả tham số cùng lúc. Sweep từng tham số 1:
            <br />
            • Step 1: Tìm best tensor_parallel_size
            <br />
            • Step 2: Tìm best max_num_seqs với TP cố định
            <br />
            • Step 3: Tăng gpu_memory_utilization
          </div>
        </div>

        <div style={styles.paramBox}>
          <div style={styles.paramName}>4. Monitor OOM & latency</div>
          <div style={styles.paramDescription}>
            • Nếu OOM xảy ra: giảm max_num_seqs hoặc gpu_memory_utilization
            <br />
            • Nếu latency quá cao: giảm batch size hoặc enable quantization
          </div>
        </div>
      </div>

      {/* Resources */}
      <div style={{ ...styles.section, borderLeftColor: "#f59e0b", background: "#fffbeb" }}>
        <div style={{ ...styles.sectionTitle, color: "#f59e0b" }}>
          📚 Resources
        </div>
        <ul style={{ marginLeft: 20, marginBottom: 0, color: "#6b7280", lineHeight: 2 }}>
          <li>
            <a
              href="/vllm_sampling_params.html"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#2563eb", textDecoration: "none" }}
            >
              vLLM Sampling Parameters Documentation
            </a>
          </li>
          <li>
            <a
              href="https://docs.vllm.ai"
              target="_blank"
              rel="noopener noreferrer"
              style={{ color: "#2563eb", textDecoration: "none" }}
            >
              Official vLLM Documentation
            </a>
          </li>
        </ul>
      </div>
    </div>
  );
}
