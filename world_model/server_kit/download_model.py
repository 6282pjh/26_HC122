# -*- coding: utf-8 -*-
"""
download_model.py — Cosmos-Predict2.5-2B (diffusers/base/post-trained) 가중치를 HF_HOME에 받는다.

전제:
  - HuggingFace에서 nvidia/Cosmos-Predict2.5-2B 라이선스 동의 + `huggingface-cli login` 완료
  - project.env를 source 해서 HF_HOME이 $PROJECT_ROOT/hf_cache로 잡혀 있을 것
    (source server_kit/project.env && python server_kit/download_model.py)

받은 뒤에는 학습/추론 모두 from_pretrained("nvidia/Cosmos-Predict2.5-2B",
revision="diffusers/base/post-trained")만 쓰면 이 캐시를 그대로 사용한다.
"""
import os
import sys
from pathlib import Path

model_id = os.environ.get("MODEL_ID", "nvidia/Cosmos-Predict2.5-2B")
revision = os.environ.get("MODEL_REVISION", "diffusers/base/post-trained")
hf_home = os.environ.get("HF_HOME")
if not hf_home:
    print("[중단] HF_HOME 미설정 → 먼저 `source server_kit/project.env`")
    sys.exit(1)

from huggingface_hub import HfApi, snapshot_download  # noqa: E402

try:
    who = HfApi().whoami()
    print(f"[HF] 로그인: {who.get('name')}")
except Exception as e:
    print(f"[중단] HF 로그인 안 됨: {e}\n  → huggingface-cli login")
    sys.exit(1)

print(f"[다운로드] {model_id} @ {revision} → HF_HOME={hf_home}")
path = snapshot_download(repo_id=model_id, revision=revision)
total = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
# snapshot 폴더는 blobs로의 심볼릭 링크라 stat()이 실제 파일 크기를 따라감
print(f"[완료] {path}")
print(f"  용량 합계 ≈ {total / 1e9:.1f} GB  (10/4 조사 기준 약 21.1GB 예상)")
