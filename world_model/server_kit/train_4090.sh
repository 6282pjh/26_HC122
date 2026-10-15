#!/usr/bin/env bash
# train_4090.sh — RTX 4090(24GB)용 LoRA 학습 실행 (캐시 → 학습)
#
# 사용 예 (항상 tmux 안에서, conda activate cosmos_surgical 후):
#   메모리 측정(G0):  RUN=g0_528 NCLIPS=4 EPOCHS=1 bash server_kit/train_4090.sh
#   E1 파일럿:        RUN=e1_r16 EPOCHS=110 CKPT_EPOCHS=28 bash server_kit/train_4090.sh
#   GPU1에서 동시에:  GPU=1 RUN=e1_r32 RANK=32 EPOCHS=110 CKPT_EPOCHS=28 bash server_kit/train_4090.sh
#   두 GPU로 한 실험(DDP, 미검증): GPU=0,1 RUN=e1_ddp EPOCHS=110 bash server_kit/train_4090.sh
#
# 변수(기본값):
#   RUN=필수 DATA=$PROJECT_ROOT/data/dataset/train H=528 W=960 RANK=16 LR=1e-4
#   EPOCHS=1 CKPT_EPOCHS=EPOCHS NCLIPS=0(전체) GPU=0 I2W='{"1": 1.0}' SEED=0
#   WARMUP=자동(총 step 5%, 최소 10) F_MAX=1.0 F_MIN=0.2 LOG_EVERY=10
#
# 동작:
#   1) $PROJECT_ROOT/data/cache/<DATA폴더명>_<H>x<W>/ 에 캐시가 없으면 cache_latents.py 먼저 실행
#   2) train_lora_4090.py 실행 (GPU에는 DiT만), 출력: $PROJECT_ROOT/lora_checkpoints/<날짜>_<RUN>/
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project.env"

: "${RUN:?RUN=실험이름 을 지정하세요}"
DATA="${DATA:-$PROJECT_ROOT/data/dataset/train}"
H="${H:-528}"; W="${W:-960}"; RANK="${RANK:-16}"; LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-1}"; CKPT_EPOCHS="${CKPT_EPOCHS:-$EPOCHS}"; NCLIPS="${NCLIPS:-0}"
I2W_DEFAULT='{"1": 1.0}'
GPU="${GPU:-0}"; I2W="${I2W:-$I2W_DEFAULT}"; SEED="${SEED:-0}"
F_MAX="${F_MAX:-1.0}"; F_MIN="${F_MIN:-0.2}"; LOG_EVERY="${LOG_EVERY:-10}"
(( H % 16 == 0 && W % 16 == 0 )) || { echo "[X] H/W는 16의 배수여야 함"; exit 1; }
PY="${CONDA_PREFIX:-}/bin/python"; ACC="${CONDA_PREFIX:-}/bin/accelerate"
[ -x "$ACC" ] || { echo "[X] conda activate $CONDA_ENV_PREFIX 먼저 (accelerate 절대경로 없음)"; exit 1; }
# [HDD판] 공유 서버: 지정 GPU를 이미 누가 쓰고 있으면 경고 (Isaac Lab 학습 등과 겹치면 OOM)
if command -v nvidia-smi >/dev/null; then
  for g in $(echo "$GPU" | tr ',' ' '); do
    used=$(nvidia-smi -i "$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -dc '0-9')
    if [ "${used:-0}" -gt 2000 ]; then
      echo "[! ] GPU$g 에서 이미 ${used}MiB 사용 중 — 다른 작업과 겹치면 OOM. 10초 후 계속 (Ctrl+C로 취소)"
      nvidia-smi -i "$g" --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/     /'
      sleep 10
    fi
  done
fi
NPROC=$(echo "$GPU" | tr ',' '\n' | wc -l)
FIRST_GPU=$(echo "$GPU" | cut -d, -f1)

# 1) 캐시
CACHE="$PROJECT_ROOT/data/cache/$(basename "$DATA")_${H}x${W}"
mkdir -p "$PROJECT_ROOT/data/cache"
CACHE_DEVICE="${CACHE_DEVICE:-cuda:0}"   # 테스트용: CACHE_DEVICE=cpu
CACHE_EXTRA="${CACHE_EXTRA:-}"          # 예: CACHE_EXTRA=--text_on_cpu (캐시 단계 OOM 시)
NV=$(ls "$DATA/videos"/*.mp4 | wc -l)
if [ ! -f "$CACHE/index.json" ] || [ "$("$PY" -c "import json;print(len(json.load(open('$CACHE/index.json'))['clips']))")" != "$NV" ]; then
  echo "[캐시] $CACHE 생성 (클립 $NV개, GPU $FIRST_GPU)"
  CUDA_VISIBLE_DEVICES="$FIRST_GPU" "$PY" "$HERE/cache_latents.py" --data_dir "$DATA" --cache_dir "$CACHE" \
     --height "$H" --width "$W" --device "$CACHE_DEVICE" --model_id "$MODEL_ID" --revision "$MODEL_REVISION" $CACHE_EXTRA 2>&1 | tee "$CACHE.log"
fi

# 서브셋(G0 측정용): 앞 NCLIPS개만 쓰는 index.json을 별도 폴더에
if [ "$NCLIPS" -gt 0 ]; then
  SUB="${CACHE}_sub${NCLIPS}_${RUN}"; rm -rf "$SUB"; mkdir -p "$SUB"
  "$PY" - "$CACHE" "$SUB" "$NCLIPS" <<'EOF'
import json, os, sys
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
idx = json.load(open(os.path.join(src, "index.json")))
idx["clips"] = idx["clips"][:n]
for f in [c["latent"] for c in idx["clips"]] + list(idx["prompt_embeds"].values()):
    os.symlink(os.path.join(src, f), os.path.join(dst, f))
json.dump(idx, open(os.path.join(dst, "index.json"), "w"), ensure_ascii=False, indent=1)
EOF
  CACHE="$SUB"
fi
N=$("$PY" -c "import json;print(len(json.load(open('$CACHE/index.json'))['clips']))")

# 총 step (1 epoch = 클립 수 / GPU 수, batch 1) — 스케줄러 길이를 여기에 맞춤
STEPS_PER_EPOCH=$(( (N + NPROC - 1) / NPROC ))
TOTAL=$(( STEPS_PER_EPOCH * EPOCHS ))
# accelerate는 다중 GPU에서 스케줄러를 GPU 수만큼 더 진행시키므로 스케줄러 길이는 클립수×epoch 기준
SCHED_TOTAL=$(( N * EPOCHS ))
WARMUP="${WARMUP:-$(( SCHED_TOTAL / 20 > 10 ? SCHED_TOTAL / 20 : 10 ))}"
(( WARMUP < SCHED_TOTAL )) || WARMUP=$(( SCHED_TOTAL > 1 ? SCHED_TOTAL - 1 : 0 ))

OUT="$PROJECT_ROOT/lora_checkpoints/$(date +%Y%m%d)_${RUN}"
mkdir -p "$OUT"
cat > "$OUT/run_config.txt" <<EOF
run=$RUN date=$(date -Is) host=$(hostname) gpu=$GPU nproc=$NPROC
data=$DATA cache=$CACHE clips=$N epochs=$EPOCHS optimizer_steps=$TOTAL sched_steps=$SCHED_TOTAL ckpt_epochs=$CKPT_EPOCHS
res=${H}x${W} frames=93 rank=$RANK alpha=$RANK lr=$LR warmup=$WARMUP f_max=$F_MAX f_min=$F_MIN
conditional_frames_probs=$I2W seed=$SEED script=train_lora_4090.py
diffusers=$(git -C "$DIFFUSERS_DIR" rev-parse --short HEAD 2>/dev/null || echo ?)
EOF
cat "$OUT/run_config.txt"

export CUDA_VISIBLE_DEVICES="$GPU" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 메모리 단편화 완화
MULTI=""; [ "$NPROC" -gt 1 ] && MULTI="--multi_gpu"
nohup "$ACC" launch $MULTI --num_processes "$NPROC" --mixed_precision bf16 "$HERE/train_lora_4090.py" \
  --pretrained_model_name_or_path "$MODEL_ID" --revision "$MODEL_REVISION" \
  --train_data_dir "$DATA" --latent_cache_dir "$CACHE" --output_dir "$OUT" \
  --height "$H" --width "$W" --num_frames 93 \
  --train_batch_size 1 --dataloader_num_workers 0 \
  --num_train_epochs "$EPOCHS" --checkpointing_epochs "$CKPT_EPOCHS" \
  --learning_rate "$LR" --scheduler_warm_up_steps "$WARMUP" --num_training_steps "$SCHED_TOTAL" \
  --scheduler_f_max "$F_MAX" --scheduler_f_min "$F_MIN" \
  --lora_rank "$RANK" --lora_alpha "$RANK" \
  --conditional_frames_probs "$I2W" --log_every "$LOG_EVERY" \
  --gradient_checkpointing --allow_tf32 --seed "$SEED" --report_to tensorboard \
  > "$OUT/train.log" 2>&1 &
echo $! > "$OUT/pid"
echo "[시작] PID=$(cat "$OUT/pid")  로그: tail -f $OUT/train.log   지표: cat $OUT/metrics.csv"
