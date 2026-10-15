# web_server — 음성 명령 · 월드 모델 예측 · 승인 · 로봇 전달

월드 모델 기반 수술 어시스턴트 로봇(낭만닥터)의 서버 쪽 코드입니다.
집도의의 음성 명령을 받아 **Cosmos-Predict2.5-2B + LoRA로 동작을 미리 영상으로 생성**하고, 집도의가 웹 콘솔에서 승인하면 Jetson이 실제 로봇팔을 움직입니다.

## 흐름

```
iPad 웹 콘솔 ──녹음──▶ inputs/incoming/{job}.wav ──▶ remote_realtime_listener.py
   ▲                                                   │ ① Whisper STT
   │                                                   │ ② 명령 분류 (키워드 → 필요 시 LLM 보정)
   │                                                   │ ③ Cosmos + LoRA 예측 영상 (Gemini E 촬영 이미지 입력)
   │◀──────── outputs/{job}.json / _result.mp4 ◀───────┘
   │
 집도의 승인 ──▶ robot_bridge/command.json ──▶ Jetson (/robot/command 폴링 → /robot/ack)
```

## 구성

| 경로 | 역할 |
|---|---|
| `remote_realtime_listener.py` | wav 감시 → STT → 명령 분류 → Cosmos LoRA 예측 영상 생성 → review 전환. "수술 완료"는 영상 없이 로봇으로 직행 |
| `captions.json` | 동작별 Cosmos 프롬프트. **LoRA 학습 캡션과 글자까지 같아야 함** |
| `web_console/app.py` | Flask 승인 콘솔(포트 5050). 녹음 업로드, job 상태, 2단계 승인, Jetson용 `/robot/command`·`/robot/ack` |
| `web_console/flask_capture_routes.py`, `capture_bridge/` | 카메라 클라이언트(Gemini E)와 촬영 START/STOP·이미지 업로드 |
| `web_console/templates/`, `static/` | iPad 화면 (녹음 → 생성 중 → 검토·승인 → 로봇 상태) |
| `assets/reference_frame.png` | 촬영 이미지가 없을 때 쓰는 기준 프레임 (2차 학습 test 클립 첫 프레임) |
| `tools/make_embeds.py` | 캡션 2개와 기본 negative prompt의 텍스트 임베딩을 미리 계산 → `assets/prompt_embeds/` (git 제외) |
| `tests/smoke_test.py` | 웹 콘솔 + 워커를 실제 모델로 한 번에 돌려 보는 테스트 (빈 GPU에서만 실행되도록 시작 시 GPU 점유를 검사) |

## 설계 포인트

- **명령 분류**: 키워드(절개/개복/수술+완료)로 먼저 판단해 지연 없이 처리하고, 키워드가 맞지 않을 때만 Groq LLM이 STT 오인식(예: "개복"→"복구")을 보정합니다. LLM은 절개/개복만 고를 수 있고, 승인 없이 실행되는 "수술 완료"는 키워드로만 인정합니다.
- **프롬프트 고정**: LoRA는 학습 캡션 문장에 맞춰 학습되므로, 생성 프롬프트는 LLM이 새로 쓰지 않고 `captions.json`의 학습 캡션을 그대로 씁니다.
- **입력 프레임**: 녹음 시작과 동시에 Gemini E가 찍은 현재 수술 환경 이미지를 학습 클립과 같은 방식(가운데 960:528 크롭 → 960×528)으로 만든 뒤, 학습 스크립트와 같은 PIL bilinear로 768×432까지 줄여서 넘깁니다. 파이프라인에 큰 이미지를 그대로 넘기면 내부에서 nearest 보간으로 줄여 학습 전처리와 달라지기 때문입니다.
- **생성 스펙**: 432×768, 93프레임, 16fps, 30 step, guidance 7.0, seed 42 — 2차 LoRA 학습(E1) 해상도와 같습니다. RTX 4090 실측 영상당 194~226초.
- **텍스트 인코더 생략**: 캡션이 2개로 고정이라 텍스트 임베딩을 미리 계산해 두고, 워커는 텍스트 인코더(약 16GB)를 로드하지 않습니다. 임베딩을 만든 캡션과 `captions.json`이 다르면(sha1 비교) 자동으로 텍스트 인코더 경로로 돌아갑니다.
- **GPU 배치**: GPU가 2장이면 Whisper는 다른 GPU에 둡니다. 1장이면 임베딩을 쓸 때는 같은 GPU, 텍스트 인코더 경로일 때는 CPU에 둡니다.
- **GPU 메모리**: 40GB 미만 GPU(예: RTX 4090)는 CPU offload, 그 이상(A6000)은 전부 GPU 상주로 자동 선택.

## 실행

```bash
conda activate cosmos_surgical
pip install -r requirements.txt          # torch·diffusers 포크는 requirements.txt 주석 참고
conda install -c conda-forge ffmpeg      # ffmpeg 명령이 없으면 STT에서 실패

# (권장, 처음 한 번) 텍스트 임베딩 미리 계산 — 빈 GPU에서 약 3분, nvidia-smi 피크 약 17.5GB
CUDA_VISIBLE_DEVICES=0 python tools/make_embeds.py

# 터미널 1 — 웹 콘솔
cd web_console && python app.py

# 터미널 2 — 워커
export COSMOS_LORA_DIR=/path/to/lora_checkpoints/20261005_e1_r16   # world_model 학습 결과, 없으면 base 모델
export GROQ_API_KEY=...                                         # 없으면 키워드 분류만
python remote_realtime_listener.py
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `HC122_WORKDIR` | 이 폴더 | 작업 폴더 (웹 콘솔과 워커가 같은 값) |
| `COSMOS_LORA_DIR` | (없음) | `pytorch_lora_weights.safetensors`가 있는 폴더 |
| `COSMOS_EMBEDS_DIR`, `COSMOS_USE_EMBEDS` | `assets/prompt_embeds`, auto | 미리 계산한 임베딩 위치 / `off`면 텍스트 인코더 경로 |
| `WHISPER_DOWNLOAD_ROOT` | (whisper 기본 `~/.cache/whisper`) | Whisper 모델 저장 위치 (공유 서버는 프로젝트 경로 권장) |
| `COSMOS_OFFLOAD` | `auto` | `on` / `off` / `auto` |
| `COSMOS_STEPS`, `COSMOS_SEED`, `COSMOS_GUIDANCE` | 30, 42, 7.0 | 생성 설정 |
| `COSMOS_HEIGHT`, `COSMOS_WIDTH` | 432, 768 | **LoRA 학습 해상도와 같아야 함** |
| `COSMOS_GPU` | 0 | Cosmos를 올릴 GPU |
| `WHISPER_DEVICE` | 자동 | GPU 2장이면 다른 GPU, 1장이면 임베딩 사용 시 같은 GPU·아니면 cpu |
| `COSMOS_EXPECTED_SEC` | 210 | 화면 "남은 시간" 첫 예상값(이후 실측으로 보정) |
| `WHISPER_MODEL`, `WHISPER_BEAM_SIZE` | medium, 5 | STT 설정 |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | 보조 분류 LLM |

ffmpeg가 PATH에 있어야 합니다(녹음 wav 변환, mp4 faststart).

### 실측 (연구실 RTX 4090 24GB, r16 LoRA, 432×768·93프레임·30 step, 2026-10-11)

| 항목 | 텍스트 인코더 경로 | 임베딩 경로 |
|---|---|---|
| GPU 피크 (nvidia-smi) | 21,928 MiB (텍스트 인코딩 순간) | **9,492 MiB** |
| 프로세스 RAM 피크 | 약 22.5GB (Cosmos만) | **8.8GB** |
| 첫 생성 / 이후 생성 | 384초 / 212초 | 195초 / - |
| 생성 결과 | 기준 | 기준과 **픽셀 동일** |

- Whisper medium: GPU 4.9GB, 문장당 0.4~0.5초 / CPU에서는 문장당 약 10초.
- 임베딩 경로 + Whisper를 같은 GPU에 올리면 계산상 약 14.4GB.
- 모델은 한 번만 로드해 상주시킵니다. 공유 서버에서 RAM이 부족해 swap이 일어나면 첫 실행이 크게 느려질 수 있습니다.
- Cosmos를 올릴 GPU는 `CUDA_VISIBLE_DEVICES`로 고르고 `COSMOS_GPU=0`으로 두는 방식이 검증된 방식입니다.

## 테스트

```bash
# tests/wavs/ 에 녹음 3개를 넣는다: inc_절개선따라절개해줘.wav, lap_개복해줘.wav, done_수술끝났어.wav
CUDA_VISIBLE_DEVICES=<빈 GPU> COSMOS_LORA_DIR=<LoRA 폴더> COSMOS_SEED=0 \
  python tests/smoke_test.py --test_frames <dataset/test/first_frames> --out <결과 폴더>
```

연구실 RTX 4090에서 3건(절개 · 개복 · 수술 완료) 모두 통과했습니다 (2026-10-11). 녹음 파일은 저장소에 포함하지 않습니다.
