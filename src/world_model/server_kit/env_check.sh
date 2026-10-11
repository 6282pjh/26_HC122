#!/usr/bin/env bash
# env_check.sh — 새 서버가 학습 가능한 상태인지 한 번에 점검(아무것도 설치/수정하지 않음)
# 사용: bash server_kit/env_check.sh            (env 활성화 전/후 둘 다 가능)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project.env"

ok()   { echo "  [OK] $*"; }
warn() { echo "  [! ] $*"; }
bad()  { echo "  [X ] $*"; }

echo "== 1. GPU / 드라이버"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | sed 's/^/  /'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | awk '{s+=$1} END {print "  사용 중 메모리 합계: " s " MiB (다른 사람 작업 확인)"}'
else
  bad "nvidia-smi 없음"
fi

echo "== 2. 디스크 (필요 추정 40~60GB: 가중치 ~21.1GB + 데이터 + 체크포인트)"
mkdir -p "$PROJECT_ROOT" 2>/dev/null
df -h "$PROJECT_ROOT" | sed 's/^/  /'
avail_gb=$(df -BG --output=avail "$PROJECT_ROOT" | tail -1 | tr -dc '0-9')
[ "${avail_gb:-0}" -ge 60 ] && ok "여유 ${avail_gb}GB" || warn "여유 ${avail_gb}GB (<60GB)"

echo "== 2b. RAM (캐시 단계에서 파이프라인 전체 ~21GB를 CPU에 올림, 추론 offload도 비슷)"
free -g | sed 's/^/  /'
avail_ram=$(free -g | awk '/^Mem:/{print $7}')
[ "${avail_ram:-0}" -ge 32 ] && ok "available ${avail_ram}GB" || warn "available ${avail_ram}GB (<32GB) — 학습 2개 + 추론 동시 실행 금지"

echo "== 3. 도구"
for c in conda tmux ffmpeg git; do
  command -v $c >/dev/null && ok "$c: $(command -v $c)" || warn "$c 없음"
done

echo "== 4. Python 패키지 (현재 활성 env: ${CONDA_DEFAULT_ENV:-없음})"
python - <<'EOF'
import importlib
want = {"torch": "2.6.0+cu124", "torchvision": "0.21.0+cu124", "diffusers": None,
        "transformers": "4.52.4", "peft": "0.19.1", "accelerate": "1.14.0", "huggingface_hub": "0.36.2",
        "datasets": None, "imageio_ffmpeg": None, "tensorboard": None}
for name, ver in want.items():
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        flag = "OK" if (ver is None or v == ver) else "! "
        print(f"  [{flag}] {name} {v}" + ("" if flag == "OK" else f"  (1차 환경: {ver})"))
    except Exception as e:
        print(f"  [X ] {name} 없음 ({type(e).__name__})")
try:
    import torch
    print(f"  torch.cuda.is_available = {torch.cuda.is_available()}, CUDA {torch.version.cuda}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU{i}: {p.name}, {p.total_memory/2**30:.1f}GB, sm_{p.major}{p.minor}, bf16={torch.cuda.is_bf16_supported()}")
        if p.major >= 12:
            print("  [! ] Blackwell(sm_120) — torch 2.6+cu124는 미지원. CUDA 12.8+/PyTorch 2.7+ 필요(부록 C)")
except Exception:
    pass
try:
    from diffusers import Cosmos2_5_PredictBasePipeline  # noqa
    print("  [OK] Cosmos2_5_PredictBasePipeline import 가능")
except Exception as e:
    print(f"  [X ] Cosmos2_5_PredictBasePipeline import 실패: {e}")
EOF

echo "== 5. diffusers 포크 커밋 / 653행 패치"
if [ -d "$DIFFUSERS_DIR/.git" ]; then
  cur=$(git -C "$DIFFUSERS_DIR" rev-parse HEAD)
  [ "$cur" = "$DIFFUSERS_COMMIT" ] && ok "커밋 $cur" || warn "커밋 $cur (1차: $DIFFUSERS_COMMIT)"
  python "$HERE/check_patch.py" "$DIFFUSERS_DIR"
else
  warn "$DIFFUSERS_DIR 없음 → setup_env.sh 먼저"
fi

echo "== 6. HuggingFace / HDD 경로"
echo "  HF_HOME=$HF_HOME   TMPDIR=${TMPDIR:-}"
case "$HF_HOME" in "$PROJECT_ROOT"/*) ok "HF_HOME이 프로젝트 경로에 있음";; *) warn "HF_HOME이 HDD가 아님 → env를 activate했는지 확인";; esac
if [ -f "$HF_HOME/token" ]; then ok "토큰 파일 있음 ($HF_HOME/token)"
elif [ -f "$HOME/.cache/huggingface/token" ]; then warn "토큰이 홈(~/.cache/huggingface)에만 있음 → activate 후 huggingface-cli login 다시"
else warn "토큰 없음 → huggingface-cli login"; fi
if [ -n "${CONDA_PREFIX:-}" ]; then
  [ -f "$CONDA_PREFIX/etc/conda/activate.d/hanium_env.sh" ] && ok "activate.d 환경변수 등록됨" || warn "activate.d 미등록 (setup_env.sh 6단계)"
fi
if command -v huggingface-cli >/dev/null; then
  huggingface-cli whoami 2>&1 | head -2 | sed 's/^/  /'
else
  warn "huggingface-cli 없음"
fi
du -sh "$HF_HOME" 2>/dev/null | sed 's/^/  캐시 용량: /'
