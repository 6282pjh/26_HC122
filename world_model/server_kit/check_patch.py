# -*- coding: utf-8 -*-
"""
check_patch.py — LoRA 학습 스크립트의 conditional_frames_probs 키 패치(653행 int(k)) 확인/적용.

배경(8/19):
  --conditional_frames_probs '{"1": 1.0}' 를 주면 argparse(type=json.loads)가
  키를 문자열 "1"로 만든다. 653행
      frames_options = list(args.conditional_frames_probs.keys())
  가 ["1"]이 되어 create_condition_mask(num_cond_latent_frames=...)에 문자열이 들어가는 버그.
  (기본값 {1: 0.5, 2: 0.5}은 파이썬 dict라 int 키 → 문제 없음. CLI로 줄 때만 터짐)

패치(1줄):
      frames_options = [int(k) for k in args.conditional_frames_probs.keys()]

포크 terarachang/diffusers @13cfbb0 의 examples/cosmos/train_cosmos_predict25_lora.py
653행이 위 원본 줄과 같은 것을 2026-10-04에 GitHub에서 직접 확인함.

사용:
  python check_patch.py <diffusers 포크 경로>          # 확인만
  python check_patch.py <diffusers 포크 경로> --apply  # 원본 줄이 정확히 있을 때만 교체(.orig 백업)
"""
import shutil
import sys
from pathlib import Path

ORIG = "frames_options = list(args.conditional_frames_probs.keys())"
NEW = "frames_options = [int(k) for k in args.conditional_frames_probs.keys()]"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
apply = "--apply" in sys.argv
root = Path(args[0] if args else ".")
script = root / "examples" / "cosmos" / "train_cosmos_predict25_lora.py"
if not script.exists():
    found = sorted(root.rglob("train_cosmos_predict25_lora.py"))
    if not found:
        print("  [X ] train_cosmos_predict25_lora.py를 찾지 못함 — 포크 브랜치/커밋 확인")
        sys.exit(1)
    script = found[0]

text = script.read_text(encoding="utf-8")
lines = text.splitlines()
print(f"  스크립트: {script} ({len(lines)}줄)")
for i, l in enumerate(lines):
    if "conditional_frames_probs" in l:
        print(f"    L{i + 1}: {l.strip()}")

if NEW in text:
    print("  [OK] int(k) 패치 적용돼 있음")
    sys.exit(0)
if ORIG not in text:
    print("  [! ] 원본 줄도 패치 줄도 없음 — 커밋이 다름. 위 줄을 보고 직접 판단")
    sys.exit(1)
if not apply:
    print("  [! ] 패치 안 됨 → python check_patch.py <경로> --apply")
    sys.exit(1)

bak = script.with_suffix(".py.orig")
if not bak.exists():
    shutil.copy2(script, bak)
script.write_text(text.replace(ORIG, NEW, 1), encoding="utf-8")
print(f"  [OK] 패치 적용 (백업: {bak.name})")
