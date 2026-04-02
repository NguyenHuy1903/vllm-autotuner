#!/usr/bin/env bash

SERVICE_NAME="vllm-autotuner.service"
TARGET="/etc/systemd/system/${SERVICE_NAME}"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

if [[ $EUID -ne 0 ]]; then
   echo "Lỗi: Script này cần chạy bằng sudo: sudo bash scripts/install_service.sh"
   exit 1
fi

echo "==> Đang cài đặt $SERVICE_NAME..."
cp "${DIR}/scripts/${SERVICE_NAME}" ${TARGET}

# Cập nhật đường dẫn WorkingDirectory theo thư mục hiện tại
sed -i "s|/opt/vllm-autotuner|${DIR}|g" ${TARGET}
# Cập nhật user theo owner của thư mục hiện tại
USER_NAME=$(ls -ld ${DIR} | awk '{print $3}')
GROUP_NAME=$(ls -ld ${DIR} | awk '{print $4}')
sed -i "s|User=data_team|User=${USER_NAME}|g" ${TARGET}
sed -i "s|Group=data_team|Group=${GROUP_NAME}|g" ${TARGET}

echo "==> Reloading systemd daemon..."
systemctl daemon-reload
systemctl enable ${SERVICE_NAME}

echo "==> Cài đặt thành công! Thư mục ứng dụng: ${DIR}"
echo "    Bắt đầu service:   sudo systemctl start vllm-autotuner"
echo "    Dừng service:      sudo systemctl stop vllm-autotuner"
echo "    Kiểm tra state:    sudo systemctl status vllm-autotuner"
echo "    Xem log:           journalctl -u vllm-autotuner -f"
