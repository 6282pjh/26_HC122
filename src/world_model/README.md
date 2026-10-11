# world_model — Cosmos-Predict2.5-2B LoRA 학습 · 추론

수술 세트장에서 직접 촬영한 로봇 동작 영상으로 NVIDIA Cosmos-Predict2.5-2B(Image2World)에 LoRA를 학습해, **현재 장면 1장 + 동작 캡션 → 93프레임(약 5.8초) 미래 영상**을 생성합니다. 학습한 LoRA는 [`../control_server`](../control_server)의 워커가 불러 씁니다.

## 흐름

```
[노트북] tools/          촬영(Gemini E) → 동작 구간 검출 → 93프레임 클립 + 캡션 + 첫 프레임 → 규격 검사
            │  dataset/train (18클립), dataset/test (2클립)
            ▼
[GPU 서버] server_kit/   cache_latents.py  텍스트 인코더·VAE를 한 번만 돌려 latent·캡션 임베딩을 디스크에 저장
                         train_lora_4090.py GPU에 DiT만 올려 LoRA 학습 (train_4090.sh가 둘을 순서대로 실행)
                         infer_lora.py      test 첫 프레임 + 캡션 → 평가 영상
            ▼
           scripts/      평가 일괄 실행(eval_e1.sh), 시드 10개 생성(gen_x10.sh), 요약 시트(contact_sheet.py)
```

## 폴더

| 경로 | 내용 |
|---|---|
| `captions.json` | 동작별 캡션 2문장. 학습·추론에서 **글자까지 같은 문장**을 써야 함 |
| `tools/` | 노트북(Windows)용 데이터 도구: `record_take.py`(ffmpeg 녹화), `inspect_take.py`(움직임 구간 검출), `make_clips.py`(클립 제작), `check_clips.py`(규격 검사), `selftest.py`(합성 영상으로 전체 검증) |
| `server_kit/` | GPU 서버용: 환경 설치(`setup_env.sh`, `constraints_hanium.txt`), 점검(`env_check.sh`, `preflight_server.sh`), 모델 다운로드, 캐시·학습·추론 |
| `scripts/` | 평가·비교용 실행 스크립트 |
| `results/` | 실제 학습 때 런처가 남긴 설정 기록 (`run_config_e1_r16.txt`, `run_config_e1_r32.txt`) |

## 데이터 규격

- 촬영: Orbbec Gemini E 1920×1080 30fps → 동작 구간만 잘라 **960×528, 16fps, 정확히 93프레임** 클립
- 동작 구간이 5.75초보다 길면 추출 fps를 `92 ÷ 구간 길이`로 낮춰 93프레임에 맞춤 (절개 약 4.4배, 개복 약 2.3배 빠르게)
- `dataset/<train|test>/videos/*.mp4` + `metas/*.txt`(캡션) + `first_frames/*.png`, 파일명 앞 3글자(`inc`/`lap`)가 동작
- 2차 데이터: 1개 위치(P1)에서 10테이크 → train 18클립(절개 9 + 개복 9), test 2클립(t10)
- 데이터·영상·가중치는 저장소에 포함하지 않습니다.

## 24GB GPU에서 학습하는 방법

원본 학습 스크립트(포크 `examples/cosmos/train_cosmos_predict25_lora.py`)는 매 step마다 텍스트 인코더(Qwen2.5-VL-7B, 약 16GB), VAE, DiT를 모두 GPU에 올립니다. 1차(A6000 48GB, 528×960)에서 피크가 41.5GB였습니다.

캡션이 동작별로 고정된 2문장이고 VAE 인코딩이 결정적(평균값 사용)이라, 두 결과를 **한 번만 계산해 저장**하고 학습 중에는 DiT만 GPU에 올립니다. 작은 테스트 모델로 원본 스크립트와 나란히 돌렸을 때 step loss 5개가 일치했고 최종 LoRA 가중치 차이는 0.0이었습니다. 원본에서 바꾼 곳은 `train_lora_4090.py`에 `# [4090]` 주석으로 표시했습니다.

## 실행 (GPU 서버)

```bash
bash server_kit/preflight_server.sh          # 디스크·RAM·GPU 점검
bash server_kit/setup_env.sh                 # conda env + torch 2.6.0+cu124 + diffusers 포크(커밋 고정)
conda activate <env 경로>
bash server_kit/env_check.sh
huggingface-cli login                        # nvidia/Cosmos-Predict2.5-2B 라이선스 동의 필요
python server_kit/download_model.py          # 약 21GB

# 학습 (H, W는 16의 배수. 528×960은 24GB에서 여유가 약 1GB라 432×768 권장)
H=432 W=768 RUN=e1_r16 RANK=16 EPOCHS=110 CKPT_EPOCHS=28 GPU=0 bash server_kit/train_4090.sh

# 평가 영상
python server_kit/infer_lora.py --data_dir $PROJECT_ROOT/data/dataset/test --height 432 --width 768 \
    --out_dir eval/e1_r16 --lora_dir $PROJECT_ROOT/lora_checkpoints/<날짜>_e1_r16 --seeds 0 1
```

`server_kit/project.env`의 `HDD_BASE`만 바꾸면 경로가 모두 따라갑니다. 핀 버전은 `constraints_hanium.txt`이며 torch 2.6.0, transformers 4.52.4는 바뀌면 안 됩니다.

## 2차 학습 결과 (연구실 RTX 4090 24GB ×2, 2026-10-05 ~ 10-11)

**해상도 결정 (rank 16, 4클립 × 2 epoch)**

| 해상도 | 학습 메모리 (PyTorch 할당 피크) | step당 시간 | 판단 |
|---|---|---|---|
| 432×768 | 16.26GB | 약 14초 | 채택 |
| 528×960 | 22.57GB | 26.0초 | 여유 약 1GB, 공유 GPU에서 OOM 발생 |

**본 학습 (432×768, 18클립 × 110 epoch = 1,980 step, lr 1e-4, warmup 99)**

| | 시간 | 마지막 10줄 평균 loss | LoRA 크기 |
|---|---|---|---|
| rank 16 | 7시간 32분 | 0.0349 | 92MB |
| rank 32 | 7시간 56분 | 0.0239 | 184MB |

**평가 (test 첫 프레임, 동작당 시드 10개씩)**

- base 모델은 4개 모두 흰 로봇·사람 손 같은 엉뚱한 물체가 들어와 장면이 무너졌고, LoRA는 rank 16·32로 동작당 10개씩 만든 40개 영상 모두에서 로봇 두 대·수술포·패드가 유지됐습니다.
- 첫 프레임만으로는 절개·개복을 구분할 수 없는데(첫 프레임 평균 절대차: 같은 동작끼리 7.7~8.1, 다른 동작 사이 8.3), 캡션만 바꿔 넣으면 생성 동작이 캡션을 따라 바뀌는 모습이 보였습니다.
- 개복은 rank 32가 원본 타이밍에 더 가깝고, 절개는 rank 16이 원위치 복귀까지 나오는 경우가 있었습니다(10개 중 2개). 서비스·시연에는 rank 16을 사용했습니다.
- 위 관찰은 영상별 정지 프레임 6~7장을 기준으로 한 것입니다.

**추론 (432×768, 30 step, guidance 7.0)**: 영상당 194~226초. nvidia-smi 기준 GPU 피크 약 21.4GB이며, 미리 계산한 캡션 임베딩을 쓰면 9.5GB로 줄고 결과는 픽셀 단위로 같습니다(`control_server/tools/make_embeds.py`).

## 알려진 한계

- 학습 데이터가 한 위치(P1)에서만 촬영되었습니다. 새로 촬영한 장면에서의 일반화는 아직 검증하지 않았습니다.
- `train_lora_4090.py`에는 학습 재개 옵션이 없습니다. 같은 날 같은 `RUN` 이름으로 다시 실행하면 출력 폴더를 재사용하므로 이름을 바꿔야 합니다.
- 두 GPU로 한 실험을 돌리는 DDP 경로(`GPU=0,1`)는 검증하지 않았습니다.
