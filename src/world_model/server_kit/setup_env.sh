#!/usr/bin/env bash
# setup_env.sh — 새 서버에 1차(DIS04) 학습 환경을 재현한다. (HDD판)
# 기준: 2026-08-18 DIS04 pip freeze(haneium_backup_20260818/cosmos_pip.txt)
# 사용: bash server_kit/setup_env.sh        (conda가 PATH에 있어야 함)
#
# 대상: A6000/4090 등 Ampere·Ada GPU. Blackwell(RTX PRO 6000)은 torch 2.6+cu124 미지원 → 쓰면 안 됨(부록 C).
#
# [HDD판에서 바뀐 점]
#  1) env를 -p $CONDA_ENV_PREFIX (HDD)에 생성, conda-forge 채널만 사용
#     (최근 conda는 defaults 채널에 ToS 동의를 요구해서 비대화식 create가 멈출 수 있음)
#  2) 모든 고정 버전을 constraints 파일 하나로 묶어 처음부터 적용 → 어떤 pip install도 핀을 못 바꿈
#     (원본은 다 설치한 뒤 torch만 assert → transformers 등이 바뀐 건 못 잡음)
#  3) `datasets` 추가 — train_lora_4090.py 23행 `import datasets`. 원본 목록에 빠져 있어서
#     cache_latents.py(학습 스크립트를 import함)부터 ModuleNotFoundError로 멈춤
#  4) cosmos-guardrail 설치 제거 — 0.3.1은 transformers>=5.0을 요구해서, 원본 순서대로 깔면
#     transformers 4.52.4가 조용히 5.x로 올라감(2026-10-04 uv 의존성 해석으로 확인).
#     우리 학습/추론 코드는 전부 MockSafetyChecker를 넘기므로 필요 없음
#     (포크 pipeline_cosmos2_5_predict.py L223~224: safety_checker가 None일 때만 CosmosSafetyChecker() 생성)
#  5) activate.d에 HF_HOME 등 환경변수 등록 → conda activate만 하면 경로가 항상 같음
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project.env"

mkdir -p "$PROJECT_ROOT"/{code,data,metas,scripts,lora_checkpoints,hf_cache,outputs,eval} \
         "$PIP_CACHE_DIR" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR" "$(dirname "$CONDA_ENV_PREFIX")"
echo "[1/6] 폴더 구조: $PROJECT_ROOT  (env: $CONDA_ENV_PREFIX)"

eval "$(conda shell.bash hook)"
export CONDA_PKGS_DIRS="$PROJECT_ROOT/.cache/conda_pkgs"   # conda 패키지 캐시도 HDD로
mkdir -p "$CONDA_PKGS_DIRS"
if [ -x "$CONDA_ENV_PREFIX/bin/python" ]; then
  echo "[2/6] conda env 이미 있음 → 재사용 ($CONDA_ENV_PREFIX)"
else
  echo "[2/6] conda env 생성 (python $PY_VERSION, conda-forge)"
  conda create -y -p "$CONDA_ENV_PREFIX" -c conda-forge --override-channels "python=$PY_VERSION"
fi
conda activate "$CONDA_ENV_PREFIX"
PIP="python -m pip"

echo "[3/6] PyTorch 2.6.0 + cu124"
CONSTRAINTS="$PROJECT_ROOT/.cache/constraints_hanium.txt"
cat > "$CONSTRAINTS" <<'EOC'
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
transformers==4.52.4
tokenizers==0.21.4
peft==0.19.1
accelerate==1.14.0
huggingface_hub==0.36.2
hf-xet==1.5.1
safetensors==0.8.0
sentencepiece==0.2.1
imageio==2.37.3
imageio-ffmpeg==0.6.0
opencv-python==4.13.0.92
numpy==2.2.6
einops==0.8.2
pandas==2.3.3
scikit-image==0.25.2
datasets==3.6.0
EOC
$PIP install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "[4/6] diffusers 포크 소스 설치 ($DIFFUSERS_COMMIT)"
if [ ! -d "$DIFFUSERS_DIR/.git" ]; then
  git clone -b "$DIFFUSERS_BRANCH" "$DIFFUSERS_REPO" "$DIFFUSERS_DIR"
fi
git -C "$DIFFUSERS_DIR" fetch -q origin "$DIFFUSERS_BRANCH" || true
git -C "$DIFFUSERS_DIR" checkout -q "$DIFFUSERS_COMMIT"
$PIP install -c "$CONSTRAINTS" -e "$DIFFUSERS_DIR"
python "$HERE/check_patch.py" "$DIFFUSERS_DIR" --apply || echo "  (train_lora_4090.py에는 패치가 이미 들어 있어서 4090 경로엔 영향 없음)"

echo "[5/6] 나머지 패키지(1차 환경 버전 고정, constraints로 잠금)"
$PIP install -c "$CONSTRAINTS" \
  transformers==4.52.4 tokenizers==0.21.4 peft==0.19.1 accelerate==1.14.0 \
  huggingface_hub==0.36.2 hf-xet==1.5.1 safetensors==0.8.0 sentencepiece==0.2.1 \
  imageio==2.37.3 imageio-ffmpeg==0.6.0 opencv-python==4.13.0.92 numpy==2.2.6 \
  einops==0.8.2 tqdm pandas==2.3.3 scikit-image==0.25.2 tensorboard \
  datasets==3.6.0
python - <<'EOP'
import torch, transformers, huggingface_hub, datasets
print("torch", torch.__version__, "| transformers", transformers.__version__, "| hub", huggingface_hub.__version__, "| datasets", datasets.__version__)
assert torch.__version__.startswith("2.6.0"), "torch 버전이 바뀜!"
assert transformers.__version__ == "4.52.4", "transformers 버전이 바뀜!"
from diffusers import Cosmos2_5_PredictBasePipeline  # noqa
print("Cosmos2_5_PredictBasePipeline import OK")
EOP

echo "[6/6] activate.d에 환경변수 등록"
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
# [2026-10-05 연구실 서버 반영] ~/.bashrc가 HF_HOME 등을 이미 export하고 있으면
#   deactivate 때 unset만 해서는 원래 값이 사라진다 → activate 때 원래 값을 저장하고 deactivate 때 복원.
ACT="$CONDA_PREFIX/etc/conda/activate.d/hanium_env.sh"
cat > "$ACT" <<'EOF'
# setup_env.sh가 생성 — conda activate 시 자동 적용
for _v in TMPDIR HF_HOME PIP_CACHE_DIR TORCH_HOME TRITON_CACHE_DIR; do
  eval "[ -n \"\${$_v+x}\" ] && export _HANIUM_OLD_$_v=\"\${$_v}\""
done
unset _v
EOF
cat >> "$ACT" <<EOF
export PROJECT_ROOT="$PROJECT_ROOT"
export HF_HOME="$HF_HOME"
export PIP_CACHE_DIR="$PIP_CACHE_DIR"
export TORCH_HOME="$TORCH_HOME"
export TRITON_CACHE_DIR="$TRITON_CACHE_DIR"
export TMPDIR="$TMPDIR"
export TOKENIZERS_PARALLELISM=false
EOF
cat > "$CONDA_PREFIX/etc/conda/deactivate.d/hanium_env.sh" <<'EOF'
unset PROJECT_ROOT TOKENIZERS_PARALLELISM
for _v in TMPDIR HF_HOME PIP_CACHE_DIR TORCH_HOME TRITON_CACHE_DIR; do
  eval "if [ -n \"\${_HANIUM_OLD_$_v+x}\" ]; then export $_v=\"\${_HANIUM_OLD_$_v}\"; else unset $_v; fi; unset _HANIUM_OLD_$_v"
done
unset _v
EOF

echo
echo "완료. 다음 순서(★ 로그인은 반드시 activate 후 → 토큰이 HDD의 \$HF_HOME/token에 저장됨):"
echo "  conda deactivate; conda activate $CONDA_ENV_PREFIX"
echo "  bash $HERE/env_check.sh"
echo "  huggingface-cli login"
echo "  python $HERE/download_model.py"
