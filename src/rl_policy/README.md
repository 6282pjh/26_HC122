# rl_policy — 절개·개복 PPO 정책의 학습, 평가, 실제 로봇용 궤적 생성

NVIDIA Isaac Lab 에서 skrl PPO 로 학습한 정책 두 개와, 그 정책이 만든 동작을 실제 DOFBOT 양팔로
옮기는 코드다. 절개 정책은 오른팔 하나로 무작위 절개선을 긋고, 개복 정책은 양팔로 절개선을 벌린다.
`vision` 이 준 절개선을 받아 `jetson_control` 이 실행할 관절 궤적을 돌려준다
(입출력 계약: [`INTERFACE.md`](INTERFACE.md)).

## 흐름

```
[GPU 서버, Isaac Lab]
  task/ + agents/ ──train──▶ 정책 체크포인트 ──eval──▶ results/
                                   │
        절개: export_trajectory ──▶ apply_servo_margin ──▶ check_trajectory_safety ──▶ 롤아웃 CSV
        개복: export_opening_library ──▶ weights/opening_library.npz (위치별 궤적 1376개, 검사 통과분)

[Jetson, Isaac Lab 없음]
  vision 절개선 (Start/End, mm)
        │
  deploy.plan_opening(...)  ── 묶음에서 가장 가까운 위치의 궤적 선택 ──▶ 서보 범위·변화량 검사
        │
  궤적 (JSON / CSV, 서보 각도) ──▶ jetson_control 이 I2C 로 실행
```

정책을 로봇에서 매 주기 추론하지 않는다. 두 정책 모두 칼끝 위치나 경로 진행 거리처럼 시뮬레이션만
계산해 주는 값을 관측으로 쓰기 때문이다. 그래서 정책은 시뮬레이션에서 끝까지 실행하고, 로봇은 그
결과인 관절 궤적을 실행한다.

## 폴더

| 경로 | 역할 | 구분 |
|---|---|---|
| `INTERFACE.md` | 정책 입출력·배포 계약 | 새로 작성 |
| `task/dofbot/` | Isaac Lab 확장 패키지. 아래 환경과 과제 등록 | 기존 코드 |
| `task/dofbot/dofbot/tasks/direct/dofbot/dofbot_incision_env_v3.py`, `dofbot_incision_env_cfg_v3.py` | 절개 과제: 관측·행동·보상·종료·무작위 절개선 | 기존 코드 |
| `…/dofbot_opening_env.py`, `dofbot_opening_env_cfg.py` | 개복 과제: 교시 동작에서 경로 생성, 양팔 제어 | 기존 코드 |
| `…/agents/*.yaml` | skrl PPO 설정 (가이드의 `agents/`. 과제 등록이 패키지 안 경로를 참조해 여기에 둔다) | 기존 코드 |
| `scripts/train.py`, `play.py` | 학습, 재생 | 기존 코드 |
| `scripts/eval_incision.py`, `eval_opening.py` | 완주율 평가 | 기존 코드 |
| `scripts/export_trajectory.py`, `check_trajectory_safety.py`, `trim_trajectory.py` | 절개 롤아웃 추출, 시뮬레이션 안전 검사 | 기존 코드 |
| `scripts/apply_servo_margin.py` | 절개 롤아웃을 서보 끝단에서 띄움(재생 스크립트의 `--servo-margin` 과 같은 처리를 검사 전에 적용) | **새로 작성** (2026-10-12, 실행 확인) |
| `scripts/export_opening_trajectory.py`, `export_opening_library.py` | 개복 롤아웃 추출(위치 1개 / 격자 전체)과 검사 | 기존 코드 |
| `scripts/replay_json_in_env.py`, `clearance_check.py`, `reach_map.py`, `json_fk.py` | 교시 동작 검증, 양팔 간격 측정, 도달 범위 측정, 교시 JSON 순기구학 | 기존 코드 |
| `scripts/render_incision.py`, `render_opening.py` | 롤아웃 영상 | 기존 코드 |
| `scripts/export_policy_weights.py` | 체크포인트에서 정책만 추려 `.npz` 로 저장하고 PyTorch 와 대조 | **새로 작성** (2026-10-12, 실행 확인) |
| `deploy/` | 실제 로봇용: 궤적 생성(개복)·변환(절개)·안전 검사·dry-run. numpy 만 필요 | **새로 작성** (2026-10-12, dry-run 확인, 실기 미검증). `opening.py` 의 묶음 조회는 10-09 에 dry-run 한 코드를 옮긴 것 |
| `deploy/agx/` | Jetson 에서 Arm_Lib 로 재생하는 스크립트, 오른팔 관절 방향 측정값 | 기존 코드 |
| `motions/` | 교시 궤적 JSON 2개(출처: teach), 개복 도달 범위 맵 | 기존 자료 |
| `weights/` | 정책만 추린 `.npz` 2개, 개복 궤적 묶음 | 생성물 |
| `results/` | 평가·dry-run 요약 | 2026-10-12 재실행 |

기존 코드는 경로 참조만 이 폴더 구조에 맞게 바꿨다(에이전트 설정 파일 위치, 사용법 문구). 로직은 그대로다.

## 실행 환경

| 항목 | 값 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 × 2 |
| Isaac Sim / Isaac Lab | 5.1.0 / 2.3.2 |
| skrl | 2.1.0 |
| Python / PyTorch | 3.11.15 / 2.7.0+cu128 |
| deploy (Jetson) | Python 3, numpy. 재생 스크립트는 Yahboom `Arm_Lib` |

## 실행

Isaac Lab 이 설치된 파이썬에서, 이 폴더(`src/rl_policy`)를 작업 디렉터리로 한다.

```bash
export PYTHONPATH=$PWD/task/dofbot:$PYTHONPATH   # 과제 패키지 등록 (이 방법으로 실행 확인. pip install -e task/dofbot 도 가능하나 미확인)
export DOFBOT_ROBOT_USD=<dofbot 로봇 USD 경로>  # 로봇 모델은 저장소에 없다 (아래 한계 참고)

# 학습 (4096 환경)
python scripts/train.py --task=Dofbot-Incision-Direct-v3 --num_envs=4096 --headless --max_iterations 2000
python scripts/train.py --task=Dofbot-Opening-Direct-v0  --num_envs=4096 --headless --max_iterations 500

# 평가
python scripts/eval_incision.py <checkpoint.pt> --envs 64
python scripts/eval_opening.py  <checkpoint.pt> --envs 256

# 개복: 위치별 궤적 묶음 생성 (검사 포함)
python scripts/export_opening_library.py <개복 checkpoint.pt> --out weights/opening_library.npz

# 절개: 절개선 하나에 대한 롤아웃과 안전 검사
python scripts/export_trajectory.py <절개 checkpoint.pt> --heading 90 --length 0.05 --cx 0.17 --cy 0.0 --out traj.npz
python scripts/apply_servo_margin.py traj.npz --margin-deg 4 --out traj_margin.npz
python scripts/check_trajectory_safety.py traj_margin.npz      # "재생 가능" 판정이 나온 궤적만 로봇으로
```

Jetson 쪽 (Isaac Lab 불필요, `src/` 에서):

```bash
python -m rl_policy.deploy.dry_run opening --line <x1> <y1> <x2> <y2> --ref <기준 x1 y1 x2 y2> \
       --out out.json --csv-out out.csv                     # 모터 미구동
python rl_policy/deploy/agx/replay_opening_on_agx.py out.csv --dry-run
python rl_policy/deploy/agx/replay_opening_on_agx.py out.csv   # 실제 구동. 사람이 전원 스위치를 잡고
```

| 환경변수 | 용도 |
|---|---|
| `DOFBOT_ROBOT_USD` | 로봇 USD 파일 경로 |
| `DOFBOT_OPENING_JSON` | 개복 교시 JSON 을 다른 파일로 바꿀 때 (기본 `motions/opening_dual.json`) |

## 인터페이스

[`INTERFACE.md`](INTERFACE.md) 가 기준이다. 요점만:

- 입력: 절개선 Start/End 2점, mm, 오른팔(I2C 버스 8) 베이스 좌표.
- 출력: 서보 각도 궤적(JSON, 양팔 동기). `jetson_control` 은 `rl_policy.deploy.plan_opening(...)` 을 호출한다.
- 범위 밖이거나 검사에 실패하면 예외가 나고 궤적은 없다.

## 과제 정의

### 절개 (`Dofbot-Incision-Direct-v3`)

오른팔만 정책이 제어한다(행동 5, 관측 29). 왼팔은 피부 가장자리를 누르는 고정 자세이고 관측·보상에
들어가지 않는다. 매 에피소드 절개선을 무작위로 뽑는다(길이 40 ~ 70 mm, 방향 60 ~ 105°, 중심 x 135 ~ 235 mm,
y ±80 mm).

절개 판정은 네 조건을 모두 만족하는 것이다 (`dofbot_incision_env_v3.py:535`).

| 조건 | 기준 |
|---|---|
| 칼끝과 선 사이 수직 거리 | < 4 mm |
| 깊이 오차 | < 3 mm (목표: 피부면 아래 5 mm) |
| 공구 수직도 | cos > 0.6 |
| 칼날 모서리와 진행 방향 | \|cos\| > 0.90 |

진행 거리는 판정이 성립하는 동안에만, 이미 그은 지점에서 이어서만, 스텝당 3 mm 이내로 늘어난다
(`:551`). 한 점을 찍고 떼었다가 옆에서 다시 찍는 동작은 진행으로 인정되지 않는다.

| 보상 항목 | 식 | 가중치 |
|---|---|---|
| 절개 진행 | 이번 스텝에 늘어난 거리 / 선 길이 | 10000 |
| 완주 | 선 전체를 그었을 때 한 번 | 1000 |
| 접촉 이탈 | 긋던 중 판정이 깨진 스텝 | −200 |
| xy 접근 (거친 / 정밀) | 1 − tanh(거리 / 150 mm), 1 − tanh(거리 / 20 mm) | 0.5, 1.0 |
| 깊이 접근 (거친 / 정밀) | 1 − tanh(오차 / 50 mm), 1 − tanh(오차 / 8 mm), 선 위에 있을 때만 | 1.0, 1.5 |
| 수직 자세, 칼날 정렬 | 선에 가까울수록 크게 (1 − tanh(선까지 거리 / 30 mm) 를 곱함) | 0.5, 0.5 |
| 손끝·링크가 피부면 아래 | 침범 깊이 | −150 /m |
| 칼끝이 패치 바닥 아래 | 침범 깊이 | −150 /m |
| 관절 한계 근접, 행동 변화, 행동 크기 | | −2.0, −0.05, −0.005 |

가중치의 출처는 `dofbot_incision_env_cfg_v3.py:403-585` 다.

### 개복 (`Dofbot-Opening-Direct-v0`)

양팔을 정책 하나가 제어한다(행동 8 = 팔마다 관절 1 ~ 4, 관측 43). 기준 동작은 사람이 교시한
`motions/opening_dual.json` 이다: 양팔이 서로 반대쪽 옆으로 돌아 나와 절개선 위에서 모이고,
공구를 세워 내려간 뒤 각자 베이스 쪽으로 당겨 벌린다. 환경이 시작할 때 이 파일을 읽어 관절 공간에서
보간하고 시뮬레이션 순기구학으로 끝점 경로를 만든다(`dofbot_opening_env.py:150`, `:172`).
매 에피소드 절개선 위치를 교시 위치에서 x ±10 mm, y ±20 mm, 회전 ±8° 로 옮기고 경로도 같이 옮긴다.

| 보상 항목 | 내용 | 가중치 |
|---|---|---|
| 경로 진행 | 구간 가중(접근 1 : 1, 삽입 2, 벌림 4) 진행량. 이미 간 곳에서만 이어서 늘어난다 | 10000 |
| 유지 | 양팔이 끝점 6 mm 이내인 스텝 | 5 |
| 완료 | 60 스텝 연속 유지했을 때 한 번 | 1000 |
| 경로 이탈 | 삽입·벌림 구간에서 경로를 벗어난 스텝 | −200 |
| 양팔 동기 | 두 팔의 진행률 차이. 10% 이상 앞서면 진행이 멈춘다 | −1 |
| 경로 접근, 수직 자세 | 조밀 보상, 합계 스텝당 최대 1.5 | 0.3, 0.7, 0.5 |
| 제어점·링크가 바닥 아래, 양팔 끝점 근접 | | −150 /m, −150 /m, −50 |

가중치의 출처는 `dofbot_opening_env_cfg.py:117-146` 다.

## 실측 결과

모두 **시뮬레이션** 결과다. 조건과 정의는 각 근거 파일에 있다.

| 항목 | 값 | 조건 | 근거 |
|---|---|---|---|
| 절개 완주율 | 52 / 64 = 81% | 2026-09-01 학습 정책, 64 환경, 시드 1234, 무작위 절개선 | `results/eval_incision_20261012.md` |
| 절개율 (그은 비율) | 평균 87%, 중앙값 100% | 같음 | 같음 |
| 방향 구간별 완주 | 60~75° 18/20, 75~90° 16/20, 90~105° 15/24 | 같음 | 같음 |
| 개복 완료 | 256 / 256 | 2026-10-06 학습 정책, 256 환경, READY 자세에서 시작 | `results/eval_opening_20261012.md` |
| 개복 벌림 폭 | 135 mm (교시 133 mm), 완료 후 99% 유지 | 같음 | 같음 |
| 개복 궤적 묶음 | 1376 / 1377 위치 사용 가능, 양팔 뼈대 간격 최소 68 mm | 2.5 mm · 2° 격자 | `results/opening_library_20261009.md` |

"처음 보는 절개선"에 대해: 절개 평가의 절개선은 학습과 **같은 분포**에서 다른 시드로 뽑은 것이다.
학습 분포 밖의 절개선(방향 105° 초과 등)에 대한 평가는 하지 않았다.

## 실제 로봇 실행

두 동작 모두 PPO 정책을 시뮬레이션에서 실행해 얻은 관절 궤적을 실제 로봇에 배포하는 방식이다.

| 동작 | 무엇으로 움직였나 | 상태 |
|---|---|---|
| 개복 (양팔) | 개복 PPO 정책의 롤아웃 궤적을 양팔에서 재생 | 2026-10-08 실기 구동, 양팔 정상 동작 (교시 위치의 절개선) |
| 개복, 옮겨진 절개선 | `deploy.plan_opening` 이 절개선 위치에 맞는 롤아웃 궤적을 선택 | dry-run 통과 (`results/dryrun_20261012.md`). 실기는 교시 위치만 수행 |
| 절개 (오른팔) | 절개 PPO 정책의 롤아웃 궤적을 오른팔에서 재생 | 2026-08-31 실기 구동(칼날 미장착 안전 점검). 이 구동에서 손목 롤의 시뮬레이션–서보 오프셋(+135°)을 찾아 반영했다. 2026-10-12 끝단 여유 적용 궤적이 시뮬레이션 안전 검사와 dry-run 통과. 오프셋 반영 후 칼날 장착 구동은 수행하지 않음 |

실기 구동의 결과 파일은 남기지 않았다(구동 당시 기록은 구두 보고).

## 시연과의 차이

`motions/` 의 두 JSON 은 사람이 로봇을 움직여 기록한 궤적(teach)이고 정책의 출력이 아니다.
발표 시연에서 절개·개복에 어떤 궤적을 썼는지는 제어 담당이 `jetson_control` 에 적는다.
이 폴더의 코드는 정책이 만든 궤적을 내는 경로이며, 위 표의 실기 구동은 이 경로로 수행했다.

## 알려진 한계

- **절개 롤아웃은 끝단 여유 적용과 안전 검사를 거쳐야 한다.** 여유 없이 뽑은 궤적은 3번 관절이 가동범위
  끝에 붙어 검사에 걸리고, 절개선에 따라 칼끝이 수술대 윗면 아래로 내려가는 경우가 있다
  (`results/dryrun_20261012.md`). 검사를 통과한 궤적만 로봇으로 보낸다.
- **높이 기준.** 교시 동작을 순기구학으로 풀면 칼끝이 피부면 기준 −33 ~ −36 mm 에 있고, 절개 과제의 목표
  깊이는 −5 mm 다. 약 30 mm 차이의 원인(리그 치수 / 서보 영점)은 확인하지 않았다. 칼날을 장착한 실기
  전에 베이스판에서 피부면까지의 높이를 재서 맞춰야 한다. 개복 과제는 장면을 30 mm 내려 맞췄다.
- **개복의 대응 범위는 좁다.** 교시 절개선에서 x ±10 mm, y ±20 mm, 회전 ±8°. 벗어나면 거부한다.
- **로봇에서 정책을 매 주기 추론하지 않는다.** 시뮬레이션에서 실행한 결과를 궤적으로 배포한다(방식 B).
- 왼팔(버스 1)의 관절 방향은 오른팔과 같다고 가정했다. 영점 오프셋은 어느 팔도 측정하지 않았다.
- 개복 제어점은 닫힌 손가락 끝의 중점이다. 그리퍼에 쥔 도구의 길이는 반영하지 않았다.
- 시뮬레이션에 피부 변형 모델이 없다. 칼끝은 물체가 아니라 계산된 점이다.
- 로봇 USD 모델(약 85 MB)은 저장소에 없다. `DOFBOT_ROBOT_USD` 로 경로를 지정해야 환경이 뜬다.
- 원본 체크포인트(`.pt`)는 저장소에 없어, 평가·추출 스크립트는 클론만으로는 재실행할 수 없다.
- 보간 재생(약 16 Hz)은 실제 로봇에서 확인하지 않았다. 예전 방식은 `--no-smooth --period 0.35`.

## 루트 README 발췌 후보

- 파일: `task/dofbot/dofbot/tasks/direct/dofbot/dofbot_incision_env_v3.py` (535~567행)
- 설명: 절개 판정과 진행 거리 갱신이다. 네 조건을 만족하는 동안에만, 이미 그은 지점에서 이어서만
  진행을 인정한다. 점을 차례로 찍는 동작이 아니라 접촉을 유지한 채 긋는 동작을 학습하게 만든 부분이다.
- 다른 후보: `deploy/opening.py` (82~118행) — 절개선 → 위치 계산 → 궤적 선택 → 안전 검사.
- 루트 README 수정 필요 사항:
  1. **1-5 "64개 평가 환경 완주율 80~86%, 실제 도달 가능 방향 구간 94%, xy 정밀도 sub-mm"** —
     이 저장소의 정책으로 재실행한 값은 **64 환경 완주율 81% (52/64)** 로 80 ~ 86% 범위 안이다.
     "94%" 는 이전 설정(방향 범위 60 ~ 150°)에서 잰 값이라 이 저장소의 코드로는 재현되지 않는다.
     같은 성격의 현재 수치는 방향 구간별 완주율이다: 60 ~ 75° 구간 90% (18/20), 60 ~ 90° 구간 85% (34/40).
     "sub-mm" 는 칼끝과 선 사이 최소 거리이며 현재 평가에서도 0.0 mm 다.
  2. **1-3 "처음 보는 절개선도 재학습 없이 수행"** — 평가 절개선은 학습과 같은 분포에서 뽑았다.
     "학습 분포 안의 무작위 절개선"이 정확하다.
  3. **1-4 ④ "PPO 절개 정책 … 양팔 DOFBOT 이 절개·개복 동작을 협응 수행"** — 절개 정책은 오른팔만
     제어하고, 개복은 별도의 양팔 PPO 정책이다. "PPO 절개 정책과 양팔 PPO 개복 정책"이 정확하다.
     실기 구동 상태는 위 "실제 로봇 실행" 표와 같다.
  4. `src/rl_policy/README.md` 의 이전 안내 문구 "웨이포인트 8개" — 코드는 5개이고 판정은 연속 스트로크다.
