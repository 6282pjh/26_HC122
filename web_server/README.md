# web_server — 음성 명령 · 월드 모델 예측 · 승인 · 로봇 전달

수술 어시스턴트 로봇(낭만닥터)의 중심 서버입니다. 집도의의 음성 명령을 인식해 동작을 분류하고, **LoRA로 학습한 Cosmos-Predict2.5-2B가 그 동작의 결과를 미리 영상으로 생성**합니다. 집도의가 iPad 웹 콘솔에서 영상을 보고 승인하면, 그 명령만 Jetson으로 전달되어 실제 로봇팔이 움직입니다.

<img src="../images/feature_console.png" width="640" alt="iPad 승인 콘솔">

## 동작 흐름

```
iPad 웹 콘솔 ──녹음──▶ inputs/incoming/{job}.wav ──▶ remote_realtime_listener.py
   ▲                                                   │ ① Whisper STT
   │                                                   │ ② 명령 분류 (키워드 → 필요 시 LLM 보정)
   │                                                   │ ③ Cosmos + LoRA 예측 영상 (Gemini E 촬영 이미지 입력)
   │◀──────── outputs/{job}.json / _result.mp4 ◀───────┘
   │
 집도의 승인 ──▶ robot_bridge/command.json ──▶ Jetson (/robot/command 폴링 → /robot/ack)
```

웹 콘솔(`web_console/app.py`)과 워커(`remote_realtime_listener.py`)는 파일로만 주고받습니다. 웹 콘솔은 녹음을 저장하고 화면을 보여주는 일만 하고, 무거운 모델(Whisper, Cosmos)은 워커가 한 번 로드해 상주시킵니다.

## 핵심 설계

**1. 키워드 우선 명령 분류**
수술 명령은 같은 말에 항상 같은 결과가 나와야 합니다. 그래서 Whisper 디코딩을 결정론적으로 고정하고(`temperature=0`, `beam_size=5`), 분류는 키워드(절개 / 개복 / 수술+완료)로 먼저 판단합니다. 키워드가 맞지 않을 때만 Groq LLM이 STT 오인식(예: "개복"→"복구")을 보정합니다. LLM은 절개·개복만 고를 수 있고, 승인 화면 없이 바로 실행되는 "수술 완료"는 키워드로만 인정합니다.

**2. 학습 때와 같은 조건으로 생성**
LoRA는 학습 캡션 문장에 맞춰 학습되므로, 생성 프롬프트는 LLM이 새로 쓰지 않고 `captions.json`의 학습 캡션을 글자 그대로 씁니다. 입력 장면도 녹음 시작과 동시에 Gemini E가 찍은 사진을 학습 클립과 같은 방식(가운데 960:528 크롭 → 960×528 → PIL bilinear 768×432)으로 만들어 넘깁니다. 파이프라인에 큰 이미지를 그대로 넘기면 내부에서 nearest 보간으로 줄어 학습 전처리와 달라지기 때문입니다.

**3. 텍스트 인코더 없이 생성**
캡션이 동작별 2문장으로 고정이므로, 텍스트 임베딩을 `tools/make_embeds.py`로 한 번만 계산해 두고 워커는 텍스트 인코더(약 16GB)를 로드하지 않습니다. 생성 결과는 픽셀 단위로 같고 GPU 피크는 절반 이하로 줄었습니다. 임베딩을 만든 캡션과 `captions.json`이 다르면(sha1 비교) 자동으로 텍스트 인코더 경로로 전환합니다.

**4. GPU에 맞춘 자동 배치**
40GB 미만 GPU(RTX 4090)는 CPU offload, 그 이상(A6000)은 전부 GPU에 상주시킵니다. GPU가 2장이면 Whisper를 다른 GPU에 둡니다.

**5. 2단계 승인과 로봇 명령 동기화**
- 첫 탭은 "장전"만 하고 **3초 안에 한 번 더** 눌러야 승인됩니다. 장갑 착용이나 오접촉에 의한 실행을 막기 위한 장치입니다.
- 승인된 명령은 `command.json`에 게시만 하고, Jetson이 주기적으로 가져갑니다. 승인할 때마다 고유 `job_id`를 새로 만들고, 임시 파일에 쓴 뒤 `os.replace`로 원자적으로 교체합니다.
- Jetson의 ACK가 현재 명령의 `job_id`와 다르면 `409`로 거부해, 늦게 도착한 이전 응답이 최신 상태를 덮어쓰지 못하게 합니다.
- 카메라 촬영 요청(`capture_bridge/`)은 첫 프레임 업로드 처리가 그 사이에 들어온 STOP 요청을 덮어쓰지 않도록, START 처리 중일 때만 상태를 갱신합니다.

## 실측 결과

RTX 4090 24GB, r16 LoRA, 432×768 · 93프레임 · 30 step, 2026-10-11

| 항목 | 텍스트 인코더 경로 | 임베딩 경로 |
|---|---|---|
| GPU 피크 (nvidia-smi) | 21,928 MiB | **9,492 MiB** |
| 프로세스 RAM 피크 | 약 22.5GB | **8.8GB** |
| 첫 생성 / 이후 생성 | 384초 / 212초 | 195초 / - |
| 생성 결과 | 기준 | 기준과 **픽셀 동일** |

- 예측 영상 생성: 영상당 194~226초 (432×768, 93프레임, 16fps, guidance 7.0)
- Whisper medium STT: GPU 4.9GB, 문장당 0.4~0.5초
- 웹 콘솔 + 워커 통합 테스트(`tests/smoke_test.py`): 절개 · 개복 · 수술 완료 3건 모두 통과

## 파일 구성

| 경로 | 역할 |
|---|---|
| `remote_realtime_listener.py` | 워커: wav 감시 → STT → 명령 분류 → Cosmos LoRA 예측 영상 생성 → 검토 단계 전환. "수술 완료"는 영상 없이 로봇으로 전달 |
| `captions.json` | 동작별 Cosmos 프롬프트 (LoRA 학습 캡션과 동일) |
| `web_console/app.py` | Flask 승인 콘솔(포트 5050): 녹음 업로드, job 상태, 2단계 승인, Jetson용 `/robot/command`·`/robot/ack` |
| `web_console/flask_capture_routes.py`, `capture_bridge/` | 카메라 클라이언트(Gemini E)와 촬영 START/STOP·이미지 업로드 |
| `web_console/templates/`, `static/` | iPad 화면 (녹음 → 생성 중 → 검토·승인 → 로봇 상태) |
| `assets/reference_frame.png` | 촬영 이미지가 없을 때 쓰는 기준 프레임 |
| `tools/make_embeds.py` | 캡션 텍스트 임베딩 사전 계산 |
| `tests/smoke_test.py` | 웹 콘솔 + 워커를 실제 모델로 돌려 보는 통합 테스트 |

<details>
<summary><b>▶ 실행 방법</b></summary>

```bash
conda activate cosmos_surgical
pip install -r requirements.txt          # torch·diffusers 포크는 requirements.txt 주석 참고
conda install -c conda-forge ffmpeg      # 녹음 wav 변환, mp4 faststart에 필요

# (처음 한 번) 텍스트 임베딩 미리 계산 — nvidia-smi 피크 약 17.5GB
CUDA_VISIBLE_DEVICES=0 python tools/make_embeds.py

# 터미널 1 — 웹 콘솔
cd web_console && python app.py

# 터미널 2 — 워커
export COSMOS_LORA_DIR=/path/to/lora_checkpoints/20261005_e1_r16   # world_model 학습 결과
export GROQ_API_KEY=...                                         # 없으면 키워드 분류만
python remote_realtime_listener.py
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `HC122_WORKDIR` | 이 폴더 | 작업 폴더 (웹 콘솔과 워커가 같은 값) |
| `COSMOS_LORA_DIR` | (없음) | `pytorch_lora_weights.safetensors`가 있는 폴더 |
| `COSMOS_EMBEDS_DIR`, `COSMOS_USE_EMBEDS` | `assets/prompt_embeds`, auto | 미리 계산한 임베딩 위치 / `off`면 텍스트 인코더 경로 |
| `WHISPER_DOWNLOAD_ROOT` | `~/.cache/whisper` | Whisper 모델 저장 위치 |
| `COSMOS_OFFLOAD` | `auto` | `on` / `off` / `auto` |
| `COSMOS_STEPS`, `COSMOS_SEED`, `COSMOS_GUIDANCE` | 30, 42, 7.0 | 생성 설정 |
| `COSMOS_HEIGHT`, `COSMOS_WIDTH` | 432, 768 | LoRA 학습 해상도와 같아야 함 |
| `COSMOS_GPU` | 0 | Cosmos를 올릴 GPU (`CUDA_VISIBLE_DEVICES`와 함께 사용) |
| `WHISPER_DEVICE` | 자동 | GPU 2장이면 다른 GPU, 1장이면 임베딩 사용 시 같은 GPU·아니면 cpu |
| `COSMOS_EXPECTED_SEC` | 210 | 화면 "남은 시간" 첫 예상값(이후 실측으로 보정) |
| `WHISPER_MODEL`, `WHISPER_BEAM_SIZE` | medium, 5 | STT 설정 |
| `GROQ_MODEL` | `openai/gpt-oss-20b` | 보조 분류 LLM |

**통합 테스트**

```bash
# tests/wavs/ 에 녹음 3개: inc_절개선따라절개해줘.wav, lap_개복해줘.wav, done_수술끝났어.wav
CUDA_VISIBLE_DEVICES=<빈 GPU> COSMOS_LORA_DIR=<LoRA 폴더> COSMOS_SEED=0 \
  python tests/smoke_test.py --test_frames <dataset/test/first_frames> --out <결과 폴더>
```

</details>
