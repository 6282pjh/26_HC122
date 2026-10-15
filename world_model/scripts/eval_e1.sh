#!/usr/bin/env bash
# eval_e1.sh — 매뉴얼 6장 평가를 한 번에 (base / 504 / 1008 / 1980 / swap), 432×768.
# 사용: conda activate 후  GPU=1 R=r16 bash scripts/eval_e1.sh <lora_checkpoints/..._e1_r16 경로>   (BASE=0이면 base 생략)
set -uo pipefail
cd "$PROJECT_ROOT"
L="$1"; R="${R:-r16}"; GPU="${GPU:-1}"; BASE="${BASE:-1}"
T="$PROJECT_ROOT/data/dataset/test"; HW="--height 432 --width 768"
export CUDA_VISIBLE_DEVICES="$GPU"   # 스크립트 안에서는 cuda:0으로 보임
# 연구실 실행 당시 infer_lora.py에는 CUDA 초기화 누락 버그가 있어 torch.cuda.init()을 먼저 부르는 래퍼로 실행했다.
# 지금 저장소의 infer_lora.py는 고쳐져 있어 래퍼 없이도 돌지만, 실행 기록과 같게 래퍼를 남겨 둔다(있어도 해 없음).
INFER='import sys, runpy, torch; torch.cuda.init(); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name="__main__")'
run() { echo "=== $(date -u +%T) $*"; python -c "$INFER" server_kit/infer_lora.py --data_dir "$T" $HW --seeds 0 1 "$@" || echo "[실패] $*"; }
[ "$BASE" = 1 ] && run --out_dir eval/base_432 --tag base
run --out_dir "eval/e1_$R" --lora_dir "$L/checkpoint-27" --tag "${R}_s504"
run --out_dir "eval/e1_$R" --lora_dir "$L/checkpoint-55" --tag "${R}_s1008"
run --out_dir "eval/e1_$R" --lora_dir "$L"               --tag "${R}_s1980"
run --out_dir "eval/e1_$R" --lora_dir "$L"               --tag "${R}_s1980" --swap_prompt
echo "=== $(date -u +%T) 평가 끝"
