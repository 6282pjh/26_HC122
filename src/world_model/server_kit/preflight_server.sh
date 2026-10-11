#!/usr/bin/env bash
# preflight_server.sh — 설치 전에 "이 서버에서 돌릴 각이 나오는지" 한 번에 확인. (아무것도 설치/수정하지 않음)
# 사용: bash server_kit/preflight_server.sh 2>&1 | tee preflight.log   → 출력 전체를 공유
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/project.env"
ok()   { echo "  [OK] $*"; }
warn() { echo "  [! ] $*"; }
bad()  { echo "  [X ] $*"; FAIL=1; }
FAIL=0

echo "== 0. 서버"
echo "  host=$(hostname)  user=$(whoami)  $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2)"
echo "  PROJECT_ROOT=$PROJECT_ROOT"
echo "  CONDA_ENV_PREFIX=$CONDA_ENV_PREFIX"

echo "== 1. HDD 경로 / 용량 (필요: 가중치 21GB + env ~10GB + pip 캐시 ~4GB + 캐시·체크포인트 → 60GB 이상)"
if [ -d "$HDD_BASE" ]; then
  ok "$HDD_BASE 존재"
  if touch "$HDD_BASE/.hanium_write_test" 2>/dev/null; then rm -f "$HDD_BASE/.hanium_write_test"; ok "쓰기 가능"
  else bad "$HDD_BASE 쓰기 불가 (권한)"; fi
  df -h "$HDD_BASE" | sed 's/^/  /'
  a=$(df -BG --output=avail "$HDD_BASE" | tail -1 | tr -dc '0-9')
  [ "${a:-0}" -ge 60 ] && ok "HDD 여유 ${a}GB" || bad "HDD 여유 ${a}GB (<60GB)"
else
  bad "$HDD_BASE 없음 → project.env의 HDD_BASE 수정"
fi
[ -e "$PROJECT_ROOT" ] && warn "$PROJECT_ROOT 이미 있음 (덮어쓰지는 않음, 내용 확인)" || ok "$PROJECT_ROOT 새로 만들 예정"
[ -e "$CONDA_ENV_PREFIX" ] && warn "$CONDA_ENV_PREFIX 이미 있음 → setup_env.sh가 재사용" || ok "env 새로 만들 예정"
echo "  홈 디렉터리 여유: $(df -h --output=avail "$HOME" | tail -1 | tr -d ' ')  (홈엔 거의 안 씀)"

echo "== 2. RAM / CPU"
free -g | sed 's/^/  /'
r=$(free -g | awk '/^Mem:/{print $2}')
[ "${r:-0}" -ge 48 ] && ok "RAM ${r}GB" || warn "RAM ${r}GB — 캐시 단계(파이프라인 전체 CPU 로드 ~21GB) 때 다른 작업 줄이기"
echo "  CPU: $(nproc) cores"

echo "== 3. GPU / 드라이버 (torch 2.6.0+cu124 → 드라이버 ≥ 550 권장)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,driver_version --format=csv,noheader | sed 's/^/  /'
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  [ "${drv:-0}" -ge 550 ] && ok "driver $drv" || warn "driver $drv (<550) — cu124 wheel이 안 뜰 수 있음"
  if nvidia-smi --query-gpu=name --format=csv,noheader | grep -qi "blackwell\|RTX PRO 6000\|5090"; then
    bad "Blackwell GPU 감지 — torch 2.6+cu124 미지원. 이 키트는 4090/A6000 서버 전용"
  fi
  echo "  GPU 사용 중인 프로세스:"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null | sed 's/^/    /' || true
  for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    echo "    pid $pid → user $(ps -o user= -p "$pid" 2>/dev/null)"
  done
else
  bad "nvidia-smi 없음"
fi

echo "== 4. 도구"
if command -v conda >/dev/null; then ok "conda $(conda --version 2>&1 | awk '{print $2}') @ $(command -v conda)"
else bad "conda가 PATH에 없음 → eval \"\$(<conda경로>/bin/conda shell.bash hook)\" 후 다시"; fi
for c in git tmux curl unzip; do
  command -v $c >/dev/null && ok "$c" || warn "$c 없음"
done
command -v ffmpeg >/dev/null && ok "ffmpeg" || echo "  [--] ffmpeg 없음 (괜찮음: imageio-ffmpeg가 자체 바이너리 사용)"

echo "== 5. 네트워크 (설치·다운로드에 필요한 곳)"
for u in https://huggingface.co https://github.com https://download.pytorch.org/whl/cu124/ https://pypi.org/simple/ https://conda.anaconda.org/conda-forge/; do
  code=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$u" 2>/dev/null); code=${code:-000}
  case "$code" in 2*|3*) ok "$u ($code)";; *) bad "$u 접속 실패 ($code) — 프록시/방화벽 확인";; esac
done

echo
[ "$FAIL" -eq 0 ] && echo "결과: 치명적 문제 없음 → bash server_kit/setup_env.sh 진행 가능" \
                  || echo "결과: [X] 항목 해결 후 진행"
