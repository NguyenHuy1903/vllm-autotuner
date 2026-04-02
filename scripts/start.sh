#!/usr/bin/env bash

# Lấy thư mục gốc
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "${DIR}"

echo "==> Khởi chạy vLLM Auto-Tuner Systemd daemon..."

# Activate conda. Phải hardcode path này hoặc lấy từ người dùng, hiện tại lấy theo môi trường data_team
if [ -f "/home/data_team/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/data_team/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
    echo ">> [CẢNH BÁO] Không tìm thấy file conda.sh để activate conda data!"
fi

conda activate data
export PYTHONPATH="${DIR}"

# Chạy app
exec python backend/main.py
