#!/usr/bin/env bash
# ============================================================
# vLLM Auto-Tuner — Setup & Run Script
# Chạy script này 1 lần để cài deps và build frontend.
# Sau đó dùng tmux để chạy backend thường xuyên.
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/backend"
FRONTEND_DIR="$SCRIPT_DIR/frontend"

echo "=== vLLM Auto-Tuner Setup ==="

# ── 1. Kích hoạt conda env ────────────────────────────────────────────────────
source /home/data_team/miniconda3/etc/profile.d/conda.sh
conda activate data
echo "[1/4] Conda env 'data' đã kích hoạt"

# ── 2. Cài Python dependencies còn thiếu ──────────────────────────────────────
echo "[2/4] Kiểm tra Python dependencies..."
pip install -q aiosqlite 2>/dev/null || true
echo "  ✓ aiosqlite"

# ── 3. Build React Frontend ───────────────────────────────────────────────────
echo "[3/4] Build React frontend..."
cd "$FRONTEND_DIR"

if [ ! -d "node_modules" ]; then
    echo "  npm install (lần đầu, có thể mất vài phút)..."
    npm install --silent
fi

echo "  npm run build..."
npm run build --silent
echo "  ✓ Frontend built tại $FRONTEND_DIR/build"

# ── 4. Khởi tạo database ─────────────────────────────────────────────────────
echo "[4/4] Khởi tạo SQLite database..."
mkdir -p "$SCRIPT_DIR/data"
cd "$BACKEND_DIR"
python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')
import db
asyncio.run(db.init_db())
print('  ✓ Database khởi tạo tại ../data/results.db')
"

echo ""
echo "=== Setup hoàn tất! ==="
echo ""
echo "Để chạy backend, mở tmux session:"
echo ""
echo "  tmux new-session -s autotuner"
echo "  conda activate data"
echo "  cd $BACKEND_DIR"
echo "  uvicorn main:app --host 0.0.0.0 --port 9100 --workers 1"
echo ""
echo "SSH Port Forwarding (trên máy cá nhân của bạn):"
echo ""
echo "  ssh -L 9100:localhost:9100 data_team@172.22.132.68"
echo "  # Sau đó mở: http://localhost:9100"
echo ""
