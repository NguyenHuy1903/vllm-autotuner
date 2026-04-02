#!/usr/bin/env bash

echo "==> Dừng vLLM Auto-Tuner Systemd daemon..."

# Xoá các container worker còn chạy do stop app python
echo ">> Tiêu huỷ các vLLM containers đang còn treo lại nếu có..."
dangling_containers=$(docker ps -a --format "{{.ID}}\t{{.Names}}" | grep 'autotuner-' | awk '{print $1}')
if [ -n "$dangling_containers" ]; then
    echo ">> Xóa: $dangling_containers"
    docker rm -f $dangling_containers
else
    echo ">> Không có container vLLM worker nào treo lại."
fi

echo "==> Cleanup hoàn tất!"
exit 0
