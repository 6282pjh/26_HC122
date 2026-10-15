# [2026년 한이음 드림업 공모전] 26_HC122

<p align="center">
  <img src="https://img.shields.io/badge/NVIDIA-Cosmos--Predict2.5--2B-76B900?style=flat&logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Fine--tuning-LoRA%20(rank%2016)-orange?style=flat">
  <img src="https://img.shields.io/badge/RL-Isaac%20Lab%20%2B%20PPO-76B900?style=flat&logo=nvidia&logoColor=white">
  <img src="https://img.shields.io/badge/Human--in--the--loop-2%E2%80%91step%20approval-blue?style=flat">
  <img src="https://img.shields.io/badge/Robot-Dual%20DOFBOT%20%2B%20Jetson-black?style=flat">
  <br><a href="https://www.youtube.com/watch?v=k7HaMJAGI6U"><img src="https://img.shields.io/badge/YouTube-%EC%8B%9C%EC%97%B0%20%EC%98%81%EC%83%81-FF0000?style=for-the-badge&logo=youtube&logoColor=white"></a>
</p>

---

## **💡1. 프로젝트 개요**

**1-1. 프로젝트 소개**
- **프로젝트 명** : 월드 모델 기반 행동 사전 시뮬레이션을 제공하는 수술 어시스턴트 로봇
- **프로젝트 정의** : 집도의의 음성 명령을 받으면 월드 모델(NVIDIA Cosmos)이 로봇 행동 이후의 미래 장면을 **실행 전에 영상으로 예측**하고, 집도의가 이를 검토·승인한 경우에만 양팔 로봇이 절개·개복 등의 수술 보조 동작을 수행하는 **Human-in-the-loop 수술 보조 시스템**

  <img width="700" alt="수술 실험 세트장 구성" src="images/setup.png" /></br>

**1-2. 개발 배경 및 필요성**
- **VLA 로봇의 블랙박스 문제** : 최근 VLA(Vision-Language-Action) 기반 로봇은 행동이 실제 환경에서 어떤 결과를 낼지 실행 전에 확인할 수 없습니다. 작은 오차도 조직 손상으로 이어지는 수술 환경에서는 치명적인 한계입니다.
- **기존 수술로봇의 한계** : 기존 수술로봇은 원격 조종 중심이어서 집도의의 지속적인 조작이 필요하며, 의료진의 부담을 근본적으로 줄이기 어렵습니다.
- **의료 현장의 구조적 문제** : 고령화와 필수의료 인력 부족으로 수술 현장의 업무 부담이 커지고 있어, 반복적인 보조 동작을 분담할 지능형 보조 로봇이 필요합니다.
- 따라서 본 프로젝트는 **"행동 결정 → 즉시 실행"** 사이에 **"미래 예측 → 인간 검토"** 단계를 끼워 넣어, 확인할 수 있는 반자동화 구조를 구현하였습니다.

**1-3. 프로젝트 특장점**
- **블랙박스 해소** : 로봇의 다음 행동을 월드 모델이 예측 영상으로 먼저 보여주어, 실행 전에 결과를 확인
- **예측계·제어계 분리** : 생성 영상은 판단 근거로만 쓰고 제어값으로 쓰지 않아, 생성 모델의 오류가 물리 동작으로 전달되지 않음
- **일반화된 절개 정책** : 매 에피소드마다 무작위 절개선을 부여하는 강화학습(Isaac Lab, PPO)으로, 처음 보는 절개선도 재학습 없이 수행
- **이중 안전 구조** : 3초 이내 2회 연속 승인 + 로봇 측 상태 시퀀스 검증으로 오조작·순서 오류 차단
- **실측 기반 설계** : 해상도별 메모리 측정, base vs LoRA 대조군 생성, 입력 전처리 비교 등 주요 설계 결정을 실측으로 확정

**1-4. 주요 기능**

<table>
  <tr>
    <td align="center"><b>① 월드 모델 행동 예측</b></td>
    <td align="center"><b>② 음성 명령 · 환경 인식</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/feature_worldmodel.png" width="300"></td>
    <td align="center"><img src="images/feature_stt.png" width="300"></td>
  </tr>
  <tr>
    <td align="center">실촬영 데이터로 LoRA 파인튜닝한 Cosmos-Predict2.5-2B가<br>현재 수술 장면 1장으로 약 5.8초의 미래 행동 영상 생성<br>(Image2World, 768×432, 16fps, 93프레임)</td>
    <td align="center">Whisper STT로 한국어 명령을 인식해<br>절개 / 개복 / 수술 완료 3종 동작으로 분류,<br>동시에 카메라가 현재 수술 환경을 자동 촬영</td>
  </tr>
  <tr>
    <td align="center"><b>③ Human-in-the-loop 웹 승인 콘솔</b></td>
    <td align="center"><b>④ 강화학습 기반 로봇 실행</b></td>
  </tr>
  <tr>
    <td align="center"><img src="images/feature_console.png" width="300"></td>
    <td align="center"><img src="images/feature_robot.png" width="300"></td>
  </tr>
  <tr>
    <td align="center">명령 → 생성 중 → 검토의 3화면 구조,<br>3초 이내 2회 연속 탭으로만 승인<br>(iPad Safari 실기 검증)</td>
    <td align="center">PPO 절개 정책 + Pixel→World→Robot Base 좌표 변환으로<br>양팔 DOFBOT이 절개·개복 동작을 협응 수행</td>
  </tr>
</table>

> 각 기능은 파트별 폴더에 독립적으로 구현되어 있습니다 — ① `world_model`·`control_server`, ② `control_server`·`camera_client`, ③ `control_server`, ④ `rl_policy`(시뮬레이션 학습·평가)·`vision`(절개선 검출·좌표 보정)

**1-5. 주요 실측 결과** 

| 항목 | 결과 |
|---|---|
| 월드 모델 LoRA 학습 | 실촬영 18클립(절개·개복), 432×768 · 93프레임 · rank 16, RTX 4090 24GB에서 학습 메모리 16.3GB, 1,980 step 7시간 32분 |
| 도메인 재현 | base 모델은 평가 영상 4개 모두 엉뚱한 물체가 들어와 장면이 무너짐 → LoRA 적용 후 생성 영상 40개 모두 로봇팔 2대·수술 시트·인공 피부 패드 유지 |
| 예측 영상 생성 | 영상당 약 3.5분(30 step). 캡션 임베딩을 미리 계산해 텍스트 인코더를 생략, GPU 피크 21.4GB → 9.5GB (결과 픽셀 동일) |
| 절개 정책 (PPO) | 64개 평가 환경 완주율 80~86%, 실제 도달 가능 방향 구간 94%, xy 정밀도 sub-mm |
| 좌표 보정 | 30점 Homography Calibration, 평균 재투영 오차 약 0.78 mm |
| End-to-End | 웹 → 카메라 → STT → 승인 → Jetson → DOFBOT 3종 동작 실기 검증 완료 |


**1-6. 기대 효과 및 활용 분야**
- **기대 효과** : AI 행동을 실행 전에 시각적으로 확인하게 하여 Physical AI의 예측 가능성·설명 가능성·신뢰성 향상, 반복적인 보조 업무 분담으로 의료진 부담 완화
- **활용 분야** : 의료기관의 Human-in-the-loop 수술 보조 시스템, 새로운 수술 동작을 실제 적용 전에 검증하는 교육·검증 플랫폼, 의료취약지역 의료진 지원 및 원격 협진

**1-7. 기술 스택**

| 구분 | 사용 기술 |
|------|-----------|
| **World Model** | ![NVIDIA Cosmos](https://img.shields.io/badge/Cosmos--Predict2.5--2B-76B900?style=flat&logo=nvidia&logoColor=white) ![Diffusers](https://img.shields.io/badge/Diffusers-FFD21E?style=flat&logo=huggingface&logoColor=black) `LoRA` `Image2World` |
| **Reinforcement Learning** | ![Isaac Lab](https://img.shields.io/badge/Isaac%20Sim%20%2F%20Isaac%20Lab-76B900?style=flat&logo=nvidia&logoColor=white) `skrl` `PPO` `Sim-to-Real` |
| **Speech / LLM** | ![Whisper](https://img.shields.io/badge/Whisper-412991?style=flat&logo=openai&logoColor=white) ![Groq](https://img.shields.io/badge/Groq%20API-F55036?style=flat) |
| **Deep Learning** | ![PyTorch](https://img.shields.io/badge/PyTorch%202.6-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![CUDA](https://img.shields.io/badge/CUDA%2012.4-76B900?style=flat&logo=nvidia&logoColor=white) `BF16` |
| **Web / Server** | ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML%2FCSS%2FJS-E34F26?style=flat&logo=html5&logoColor=white) ![Tailscale](https://img.shields.io/badge/Tailscale%20Funnel-242424?style=flat&logo=tailscale&logoColor=white) |
| **Vision** | ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white) ![FFmpeg](https://img.shields.io/badge/FFmpeg-007808?style=flat&logo=ffmpeg&logoColor=white) `HSV 검출` `Homography` |
| **Embedded / Robot** | ![Jetson](https://img.shields.io/badge/Jetson%20AGX%20Xavier-76B900?style=flat&logo=nvidia&logoColor=white) `Yahboom DOFBOT ×2` `I2C (smbus)` |
| **Camera** | `Orbbec Gemini E (영상 생성용)` `Logitech C920 (로봇 제어용)` |
| **Languages** | ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) |
| **Infra** | ![Ubuntu](https://img.shields.io/badge/Ubuntu%2020.04-E95420?style=flat&logo=ubuntu&logoColor=white) `RTX A6000 48GB (1차 학습)` `RTX 4090 ×2 (2차 LoRA 학습·추론, 강화학습)` `conda` `tmux` |
| **Project Management** | ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) ![Notion](https://img.shields.io/badge/Notion-000000?style=flat&logo=notion&logoColor=white) |

---

## **💡2. 팀원 소개**

| **팀장** | **멘티1** | **멘티2** | **멘티3** | **멘티4** | **멘토** |
|:---:|:---:|:---:|:---:|:---:|:---:|
| <img width="110" src="images/team/leader.png"> | <img width="110" src="images/team/mentee1.png"> | <img width="110" src="images/team/mentee2.png"> | <img width="110" src="images/team/mentee3.png"> | <img width="110" src="images/team/mentee4.png"> | <img width="110" src="images/team/mentor.png"> |
| • 통합환경 구축 <br> • 서버 연동 <br> • Jetson 연동 | • 월드모델 구성 <br> • LoRA 학습 | • 강화 학습 <br> • PPO 절개정책 | • 하드웨어 제작 <br> • 양팔 로봇 <br> • 도구 어댑터 | • 비전 인식 <br> • 절개선 검출 <br> • 좌표 보정 | • 프로젝트 멘토 <br> • 기술 자문 |
| ![Leader](https://img.shields.io/badge/Team-Leader-blue) | ![WM](https://img.shields.io/badge/World-Model-76B900) | ![RL](https://img.shields.io/badge/RL-PPO-orange) | ![HW](https://img.shields.io/badge/HW-DOFBOT-brown) | ![Vision](https://img.shields.io/badge/Vision-Calib-red) | ![Mentor](https://img.shields.io/badge/Mentor-Advisor-lightgrey) |

---

## **💡3. 시스템 구성도**

> 이 장의 구성도 · 상태 시퀀스 · 엔티티 관계도는 **전체 시스템 설계** 기준입니다. 저장소의 코드는 파트별로 독립 구현되어 있으며(각 폴더 README 참고), 서버와 Jetson은 JSON 상태 파일(`command.json`)과 polling으로 연동했습니다. Jetson 제어 코드는 저장소에 포함하지 않았습니다.

| **3-1. 서비스 흐름도** |
|---|
| [<img src="images/diagrams/service_flow.png" width="700" alt="서비스 흐름도">](images/diagrams/service_flow.png) |

집도의가 앱에서 음성으로 명령하면 ① Gemini E 카메라가 현재 수술대를 촬영해 서버로 올리고, ② 음성은 STT로 인식해 절개 · 개복 · 수술 완료 중 하나로 분류하고, 동작별 학습 캡션이 행동 프롬프트가 됩니다(키워드로 판단되지 않을 때만 LLM이 인식 오류를 보정). 서버의 LoRA 적용 Cosmos-Predict2.5-2B가 두 입력으로 시뮬레이션 영상을 생성해 돌려주면, 집도의가 검토 후 **승인한 경우에만** 다음 작업(로봇 실행)으로 진행합니다.

| **3-2. S/W 구성도** |
|---|
| [<img src="images/diagrams/sw_architecture.png" width="700" alt="S/W 구성도">](images/diagrams/sw_architecture.png) |

시스템은 **태블릿 · 카메라 노트북 · GPU 서버 · Jetson 제어부**의 4개 노드로 나뉘어 동작합니다. 카메라 노트북과 Jetson은 외부에서 직접 접속할 수 있는 주소가 없기 때문에, 서버가 상태 파일에 명령을 **게시**하고 각 장치가 1~1.5초 주기로 이를 **조회(polling)** 해 가는 단방향 구조로 설계했습니다. 서버는 명령을 게시할 수만 있고, 실제 실행 여부는 로봇이 최종 판단합니다. 태블릿 접속은 Tailscale Funnel을 통한 HTTPS로 제공됩니다.

월드 모델이 만든 예측 영상은 판단 근거로만 쓰고, 실제 로봇 제어는 별도 계통에서 수행합니다.

| 계통 | 입력 | 출력 | 역할 |
|---|---|---|---|
| **예측계** | 촬영 이미지 + 행동 프롬프트 | 미래 행동 예측 영상 | 집도의 판단 지원 |
| **제어계** | 실제 카메라 관측 + 승인된 명령 | 로봇 관절 제어 명령 | 실제 DOFBOT 실행 |

> 두 계통을 잇는 것은 로봇 제어 데이터가 아니라 **집도의의 승인 신호** 하나뿐입니다. 생성 영상은 픽셀 결과물이라 실제 좌표나 깊이 정보가 아니므로, 영상 속 로봇팔 위치를 제어값으로 쓰지 않습니다.

| **3-3. H/W 구성도** |
|---|
| [<img src="images/diagrams/hw_architecture.png" width="700" alt="H/W 구성도">](images/diagrams/hw_architecture.png) |

- 수술대 좌우에 6축 **Yahboom DOFBOT ×2** 배치 — 우측은 Scalpel Adapter로 절개, 좌측은 조직 고정·개복 보조
- 두 DOFBOT 확장보드가 동일한 I2C 주소(0x15)를 사용하므로 Jetson의 **Bus 1(좌) / Bus 8(우)** 로 물리적으로 분리해 독립 제어
- 3D 프린팅 **Scalpel Adapter / Opening Adapter** 자체 설계 — 절개 기준점을 그리퍼 중심에서 실제 칼날 끝(Blade Tip)으로 재정의
- 카메라 역할 분리 : **Logitech C920**(상부, 절개선 검출·로봇 제어) / **Orbbec Gemini E**(측면, 월드 모델 입력·학습 데이터)
- 그린스크린 · 암막 · LED 조명으로 촬영 환경 고정 → 영상 간 차이가 로봇 행동 변화만 되도록 통제

**로봇 안전 상태 시퀀스**

```mermaid
stateDiagram-v2
    direction LR
    [*] --> IDLE
    IDLE --> INCISION_DONE : incision (절개)
    INCISION_DONE --> OPEN_HOLD : laparotomy (개복)
    OPEN_HOLD --> COMPLETE : surgery_complete (수술 완료)
    COMPLETE --> [*]
```

> 현재 상태에서 허용되지 않는 명령은 로봇이 자체적으로 거부합니다. 서버가 잘못된 순서의 명령을 보내더라도 실제 동작으로 이어지지 않는 이중 안전 구조입니다.

| **3-4. 엔티티 관계도** |
|---|
| [<img src="images/diagrams/erd.png" width="700" alt="엔티티 관계도">](images/diagrams/erd.png) |

하나의 음성 명령(`USER_COMMAND`)에서 STT 결과 · 장면 이미지 · 검토 작업(`REVIEW_JOB`)이 파생되고, 검토 작업에 시뮬레이션 영상(`SIM_VIDEO`)과 승인(`APPROVAL`)이 연결됩니다. 승인된 작업만 로봇 명령(`ROBOT_COMMAND`)이 되며, 로봇의 응답(`ROBOT_ACK`)이 시스템 상태(`SYSTEM_STATE`)를 갱신합니다. 장면 이미지는 캘리브레이션 정보(`CALIBRATION_PROFILE`)를 거쳐 절개 목표 좌표(`TARGET_COORDINATE`)로 변환되어 로봇 명령에 첨부됩니다.

> 엔티티 관계도는 설계 기준입니다. 실제 서버 구현은 데이터베이스 없이 JSON 상태 파일(job 상태, `capture_command.json`, `command.json`)로 동작하며, 좌표 변환은 `vision` 폴더에서 별도로 실행합니다.


---

## **💡4. 작품 소개영상**
> <sub>이미지를 클릭하면 유튜브 시연 영상을 보실 수 있습니다.</sub>

[![월드 모델 기반 수술 어시스턴트 로봇 시연](https://img.youtube.com/vi/k7HaMJAGI6U/maxresdefault.jpg)](https://www.youtube.com/watch?v=k7HaMJAGI6U)

▶ 시연 영상 바로가기 : https://www.youtube.com/watch?v=k7HaMJAGI6U

---

## **💡5. 핵심 소스코드**
> <sub>각 파트의 핵심 부분을 실제 코드 그대로 캡처했습니다. 제목 옆 링크를 누르면 해당 줄로 이동합니다.</sub>

### 5-1. 음성 명령 분류 — control_server ([`remote_realtime_listener.py` L198–217](control_server/remote_realtime_listener.py#L198-L217))
<img src="images/code/code_01_classify.png" alt="음성 명령 분류 코드">

- 키워드(`절개` / `개복` / `수술`+`완료·끝`)로 **먼저** 판단하고, 맞는 키워드가 없을 때만 LLM이 STT 오인식(예: "개복"→"복구")을 보정합니다.
- LLM 답은 정규식으로 한 단어만 뽑고, `GENERATIVE_ACTIONS`(절개·개복)에 들 때만 받아들입니다. 승인 화면 없이 바로 실행되는 "수술 완료"는 LLM이 만들 수 없고 키워드로만 인정됩니다.
- LLM 호출이 실패하면 `None`을 돌려 키워드 결과만 사용합니다.

### 5-2. LoRA 월드 모델 예측 영상 생성 — control_server ([`remote_realtime_listener.py` L504–517](control_server/remote_realtime_listener.py#L504-L517))
<img src="images/code/code_02_generate.png" alt="예측 영상 생성 코드">

- `pipe`는 Cosmos-Predict2.5-2B에 2차 학습 LoRA(rank 16)를 불러와 합친 파이프라인입니다([L366–368](control_server/remote_realtime_listener.py#L366-L368)).
- `image`는 명령 시점에 촬영한 현재 장면 1장이며, 학습 클립과 같은 전처리(960:528 크롭 → bilinear 768×432)를 거칩니다.
- `text_args`에는 미리 계산한 캡션 임베딩을 넣어 텍스트 인코더(약 16GB)를 로드하지 않습니다. GPU 피크가 21.4GB에서 9.5GB로 줄었고 결과는 픽셀 단위로 같습니다.
- 432×768 · 93프레임 · 30 step · guidance 7.0 · 고정 seed로 생성하고, mp4 faststart 처리가 끝난 **뒤에만** 웹 콘솔을 검토 단계로 넘깁니다.

### 5-3. 승인 명령 ACK 동기화 — control_server ([`web_console/app.py` L339–355](control_server/web_console/app.py#L339-L355))
<img src="images/code/code_03_ack.png" alt="Jetson ACK 처리 코드">

- 서버는 승인된 명령을 `command.json`에 게시만 하고 Jetson이 주기적으로 가져갑니다. 승인할 때마다 고유 `job_id`를 새로 만들고, 임시 파일에 쓴 뒤 `os.replace`로 원자적으로 교체합니다([L66–83](control_server/web_console/app.py#L66-L83)).
- Jetson이 보낸 ACK의 `job_id`가 현재 명령과 다르면 `409`로 거부해, 늦게 도착한 이전 응답이 최신 상태를 덮어쓰지 못하게 합니다.

### 5-4. 24GB GPU에서의 LoRA 학습 — world_model ([`server_kit/train_lora_4090.py` L541–562](world_model/server_kit/train_lora_4090.py#L541-L562))
<img src="images/code/code_04_lora_train.png" alt="LoRA 학습 설정 코드">

- 원본 학습 스크립트는 매 step마다 텍스트 인코더(Qwen2.5-VL-7B, 약 16GB) · VAE · DiT를 모두 GPU에 올려, 1차(A6000 48GB) 학습 때 피크가 41.5GB였습니다.
- 캡션이 동작별 2문장으로 고정이고 VAE 인코딩이 결정적이므로, `cache_latents.py`로 한 번만 계산해 저장합니다. 학습 중에는 `text_encoder=None`, VAE는 CPU에 두고 GPU에는 DiT만 올립니다([L568](world_model/server_kit/train_lora_4090.py#L568)).
- 432×768 · 93프레임 · rank 16에서 학습 메모리 16.26GB, 1,980 step에 7시간 32분(RTX 4090). 작은 테스트 모델로 원본 스크립트와 나란히 돌려 step loss와 최종 LoRA 가중치가 같음을 확인했습니다.

### 5-5. 강화학습 절개 진행 판정 — rl_policy ([`dofbot_incision_env_v3.py` L536–560](rl_policy/dofbot/tasks/dofbot_incision_env_v3.py#L536-L560))
<img src="images/code/code_05_cut.png" alt="절개 진행 판정 코드">

- 칼끝이 선 가까이(`lateral`), 정해진 깊이(`z_err`), 공구 수직(`vertical_ok`), 칼날 방향 정렬(`blade_ok`)을 **동시에** 만족할 때만 "자르는 중"으로 봅니다.
- 이미 그은 끝에서 이어질 때(공간 연속), 직전 step에도 자르고 있었을 때(시간 연속)만 절개 길이 `_cut_s`가 늘어납니다. 점을 찍듯 콕콕 찌르거나 선 끝만 건드려 보상을 받는 행동을 막기 위한 조건입니다.
- 보상은 늘어난 절개 길이에 비례해 주고([L571](rl_policy/dofbot/tasks/dofbot_incision_env_v3.py#L571)), 중간에 칼을 들면 감점합니다. 시뮬레이션 평가: 무작위 절개선 64개 중 52개 완주(81%, [평가 기록](rl_policy/results/eval_incision_20261012.md)).

### 5-6. Pixel → World → 로봇 베이스 좌표 변환 — vision ([`coordinate_transform.py` L38–47](vision/surgical_pipeline/common/coordinates/coordinate_transform.py#L38-L47))
<img src="images/code/code_06_coord.png" alt="좌표 변환 코드">

- `pixel_to_world`: 30점 캘리브레이션으로 구한 Homography `H`로 C920 영상의 Pixel 좌표를 작업 평면 World 좌표(mm)로 바꿉니다.
- `world_to_robot`: World 좌표에서 각 팔 베이스 원점을 빼고 yaw만큼 역회전(`Rz(-yaw)`)해, 마주 보고 놓인 좌·우 DOFBOT 각자의 좌표로 바꿉니다.
- 30점 적합 잔차 평균 0.78mm (H를 구할 때 쓴 같은 30점 기준, [검증 기록](vision/results/verification_20261015.md)).

---

## **📁 레포지토리 구조**

```
26_HC122
├── images/                     # README 이미지
│   ├── diagrams/               # 서비스 흐름도, S/W·H/W 구성도, 엔티티 관계도
│   ├── code/                   # 핵심 소스코드 캡처
│   └── team/                   # 팀원 사진
├── control_server/             # 서버: Flask 웹 승인 콘솔, STT·명령 분류·Cosmos LoRA 워커, 촬영 브리지
│   ├── remote_realtime_listener.py
│   ├── web_console/            # app.py, 촬영 라우트, iPad 화면(templates, static)
│   ├── capture_bridge/
│   ├── tools/make_embeds.py
│   └── tests/smoke_test.py
├── camera_client/              # 카메라 노트북 촬영 클라이언트 (Orbbec Gemini E)
├── world_model/                # Cosmos-Predict2.5-2B LoRA 학습·추론
│   ├── tools/                  # 촬영 → 클립 제작 → 규격 검사 (노트북)
│   ├── server_kit/             # 환경 설치, 캐시, 학습, 추론 (GPU 서버)
│   ├── scripts/                # 평가 일괄 실행, 요약 시트
│   ├── results/                # 학습 설정 기록
│   └── dataset_c920/           # C920 원격조작 에피소드 100개의 관절값·메타데이터, 샘플 프레임
├── rl_policy/                  # Isaac Lab PPO 절개·개복 정책
│   ├── dofbot/tasks/           # 절개(v3)·개복 환경, 환경 등록, skrl PPO 설정(agents/)
│   ├── config/, setup.py       # Isaac Lab 확장 설치 정보 (pip install -e rl_policy)
│   ├── scripts/                # 학습, 평가, 렌더링, 교시 동작 재생
│   ├── motions/                # 개복 교시 동작(JSON)
│   ├── weights/                # 학습한 정책 가중치(npz)
│   └── results/                # 시뮬레이션 평가 결과
├── vision/                     # C920 절개선 검출(HSV), 30점 Homography 좌표 보정
│   ├── surgical_pipeline/
│   │   ├── camera/             # 절개선 검출·녹화, 30점 캘리브레이션 수집·좌표 변환 CLI
│   │   ├── common/             # 좌표 변환 함수, 베이스 yaw 측정
│   │   └── config/             # 캘리브레이션 측정 자료
│   └── results/                # 보정 수치 재계산·검증 기록
└── README.md
```

> 월드 모델 가중치, 학습 영상, 체크포인트, 녹음 파일은 용량·라이선스·개인정보 문제로 저장소에 포함하지 않습니다(강화학습 정책만 경량 npz로 포함). 각 폴더는 독립적으로 설치·실행하며, 폴더별 README에 설치 방법(의존성 파일), 실행 방법, 실측 결과를 정리했습니다.

---

## **📚 참고 문헌**
- NVIDIA et al. "Cosmos World Foundation Model Platform for Physical AI." arXiv:2501.03575 (2025).
- Ho, J., Jain, A., Abbeel, P. "Denoising Diffusion Probabilistic Models." arXiv:2006.11239 (2020).
- Hu, E. J. et al. "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685 (2021).
- Radford, A. et al. "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv:2212.04356 (2022).
- Schulman, J. et al. "Proximal Policy Optimization Algorithms." arXiv:1707.06347 (2017).
- Mittal, M. et al. "Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning." arXiv:2511.04831 (2025).

**사용 오픈소스** : [NVIDIA Cosmos](https://github.com/nvidia-cosmos) · [OpenAI Whisper](https://github.com/openai/whisper) (MIT) · [Isaac Lab](https://github.com/isaac-sim/IsaacLab) (BSD-3) · [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
