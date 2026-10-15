#!/usr/bin/env bash
# gen_x10.sh — 최종 LoRA로 test 첫 프레임(inc_p1_t10, lap_p1_t10)에서 동작별 영상 10개씩(seed 0~9) 생성.
# 사용: conda activate 후  GPU=1 TAG=r16_s1980 OUT=eval/r16_x10 bash scripts/gen_x10.sh <LoRA 폴더>
#   (r16·r32 각각 이 방식으로 20개씩 만들어 비교했다: 432×768, 30 step, guidance 7.0)
set -uo pipefail
cd "$PROJECT_ROOT"
L="$1"; GPU="${GPU:-1}"; TAG="${TAG:-$(basename "$L")}"; OUT="${OUT:-eval/x10_$TAG}"
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES="$GPU"
echo "=== $(date -u +%T) 시작: $L → $OUT"
python server_kit/infer_lora.py --data_dir "$PROJECT_ROOT/data/dataset/test" --height 432 --width 768 \
  --out_dir "$OUT" --lora_dir "$L" --tag "$TAG" --seeds 0 1 2 3 4 5 6 7 8 9 || echo "[실패]"
echo "=== $(date -u +%T) 생성 끝"
