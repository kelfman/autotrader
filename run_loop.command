#!/bin/bash
# Double-click this file in Finder to run the trading research loop.
# It will open a Terminal window and run 10 iterations automatically.

# Find this script's directory (works wherever the file lives)
DIR="$(cd "$(dirname "$0")" && pwd)"

cd "$DIR"
source .venv/bin/activate

echo "======================================"
echo " Autotrader Research Loop"
echo " $(date)"
echo "======================================"
echo ""

python research_loop.py --iterations 10 --symbol "BTC/USDT" --model "claude-sonnet-4-6"

echo ""
echo "======================================"
echo " Done. Press any key to close."
echo "======================================"
read -n 1
