# vLLM Auto-Tuner Packaging & Deployment

Tài liệu này hướng dẫn 2 phương pháp đóng gói và triển khai dự án `vllm-autotuner` cho môi trường nội bộ.

---

## Method 1: Bare Metal Appliance (Systemd + Bash Scripts) - RECOMENDED
Phương pháp này phù hợp cho các GPU node nội bộ đã có sẵn Conda và Docker. Systemd sẽ quản lý daemon, đảm bảo tự động chạy khi khởi động lại máy và dễ dàng xem log.

**Cấu trúc thư mục:**
```text
vllm-autotuner/
├── config.yaml
├── setup_and_run.sh
├── scripts/
│   ├── package.sh           # Script tạo file tarball để rdist/scp
│   ├── install_service.sh   # Script cài đặt systemd service (cần sudo)
│   ├── start.sh             # Script systemd gọi để start process
│   └── stop.sh              # Script systemd gọi để stop/cleanup
```

**Workflow:**
1. **Đóng gói (Build/Pack):** Chạy `bash scripts/package.sh` để tạo ra `vllm-autotuner-release.tar.gz`.
2. **Deploy:** Copy `vllm-autotuner-release.tar.gz` sang server đích và giải nén.
3. **Cài đặt:** Chạy `sudo bash scripts/install_service.sh` để đăng ký service với systemd.
4. **Quản lý:**
   - Khởi động: `sudo systemctl start vllm-autotuner`
   - Dừng: `sudo systemctl stop vllm-autotuner`
   - Xem log: `journalctl -u vllm-autotuner -f`
   - Khởi động cùng OS: `sudo systemctl enable vllm-autotuner`

*(Các file script cho Method 1 được đặt trong thư mục `scripts/`)*

---

## Method 2: Containerized (Docker Compose)
Phương pháp này cô lập toàn bộ backend và frontend vào chung 1 container hoặc 2 container song song, giao tiếp với Docker socket của host để spawn vLLM worker container. Phù hợp nếu server đích chưa cấu hình Conda / Python environment chuẩn.

**Cấu trúc thư mục thêm vào:**
```text
vllm-autotuner/
├── Dockerfile.app           # Đóng gói Node.js build (frontend) và FastAPI (backend)
├── docker-compose.yml       # Orchestrator
```

**Lưu ý khi dùng Method 2:**
1. Cần mount `/var/run/docker.sock` từ host vào container của `app` để backend có thể bật/tắt vLLM benchmark instance.
2. Setup phức tạp hơn về networking (container-in-container hoặc sibling-containers).
3. Do yêu cầu tương tác cực sát với GPU, vLLM instance vẫn được spawn dưới dạng Docker từ Python script, nên `docker-compose` chỉ dùng để bọc orchestrator.

**Sử dụng:**
```bash
docker-compose up -d --build
```
Log xem bằng: `docker-compose logs -f`