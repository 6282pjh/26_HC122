# rl_policy INTERFACE — 정책 입출력·배포 계약

제어(`jetson_control`)와 비전(`vision`)이 이 문서를 기준으로 맞춘다.
값은 모두 이 폴더의 코드에서 확인한 것이고 출처를 `파일:행` 으로 적었다.
환경 파일의 경로 앞부분 `task/dofbot/dofbot/tasks/direct/dofbot/` 는 생략하고 파일명만 쓴다.

## 0. 요약

| 항목 | 값 |
|---|---|
| 배포 방식 | **B (오프라인 롤아웃 → 궤적).** 정책은 시뮬레이션에서 실행하고 로봇은 그 결과인 관절 궤적을 실행한다 |
| 정책 수 | 2개. 절개(오른팔), 개복(양팔). 서로 다른 과제로 따로 학습했다 |
| Jetson 에서 되는 것 | 개복: 절개선 → 궤적 (미리 계산한 묶음에서 조회). 절개: 서버에서 만든 궤적의 변환·검사만 |
| Jetson 에서 안 되는 것 | 정책 추론 자체. 두 정책 모두 시뮬레이션만 줄 수 있는 관측을 쓴다 (3절) |
| 실기 구동 | 개복: 정책 궤적을 실제 양팔에서 구동, 정상 (2026-10-08, 교시 위치). 절개: 정책 궤적을 오른팔에서 구동 (2026-08-31, 칼날 미장착 점검). 상세는 8절 |

정책 파일 (이 저장소에는 정책 신경망과 관측 정규화 통계만 추린 `.npz` 를 넣었다):

| 파일 | 내용 | 크기 | sha256 |
|---|---|---|---|
| `weights/incision_policy.npz` | 절개 정책. 관측 29 → 은닉 512·512·256·128 (ELU) → 행동 5 | 1.65 MB | `53cf2ee707f19b1ce38c59e079a7452743c65197eb44e7cde6467d655584163c` |
| `weights/opening_policy.npz` | 개복 정책. 관측 43 → 같은 구조 → 행동 8 | 1.68 MB | `6c369b60ea38009ecef71c29ffa46b00fbae46847f46a6f09aa078c5e1b7e3e4` |
| `weights/opening_library.npz` | 개복 정책을 위치별로 실행해 얻은 궤적 묶음 (1376개) | 1.08 MB | `922a95bdb608fb918a192ddd78a4d124bf0ab1af1471f7c261da749eb0bd9bb7` |

원본 skrl 체크포인트(`.pt`, 절개 7.1 MB / 개복 5.4 MB, 가치 함수와 옵티마이저 상태 포함)는 저장소에 없다.
`scripts/` 의 평가·추출 스크립트는 원본 체크포인트를 입력으로 받는다. 보관 위치: 팀 공유 드라이브(요청 시 제공).

교시 궤적 `motions/incision_dual.json`, `motions/opening_dual.json` 의 출처는 **teach** 다.
이름 붙은 서보 웨이포인트와 이동 시간으로 이루어진 파일이고(`created_at` 2026-08-31),
이 폴더의 추출 스크립트가 쓰는 형식이 아니다. 발표 시연에 실제로 쓰인 파일과 같은 것인지는 **확인 필요**
(제어 담당).

## 1. 좌표계

| 이름 | 원점 | 축 방향 | 단위 | 시뮬레이션 ↔ 실제 |
|---|---|---|---|---|
| 오른팔 베이스 (정책 기준) | 오른팔(I2C 버스 8) 베이스 회전축, 높이는 인공피부 윗면 | +x: 왼팔 베이스 쪽, +z: 위, +y: 오른손 좌표계 | m (deploy 입력은 mm) | 시뮬레이션의 환경 원점과 같다. 로봇 루트는 z = +0.030 m (`dofbot_incision_env_cfg_v3.py:203`) |
| 왼팔 베이스 | x = 0.370 m, y = 0 | 오른팔을 마주 본다 | m | `dofbot_incision_env_cfg_v3.py:283` |

- 인공피부 패치: x 0.105 ~ 0.265 m, y −0.125 ~ +0.115 m, 윗면 z = 0, 두께 10 mm (`dofbot_incision_env_cfg_v3.py:303`, `:365`).
- **칼날 끝(절개 기준점)**: 손가락 링크 `rlink2` 원점에서 공구 축(손목 링크 `arm_link5` 의 +z) 방향으로
  60.5 mm (`dofbot_incision_env_cfg_v3.py:351`, 계산은 `dofbot_incision_env_v3.py:332`).
  60.5 = 링크 원점 → 손가락 끝 50.5 mm + 손가락 끝 → 칼날 10 mm(실측).
- **개복 제어점**: 닫힌 손가락 끝의 중점 (`dofbot_opening_env.py:172` 의 `_fk_reference`).
  그리퍼에 쥔 도구의 길이는 반영돼 있지 않다 (`opening_tool_length = 0`).
- **주의 — 높이 불일치 (미해결)**: 교시 동작을 순기구학으로 풀면 칼끝이 z = −33 ~ −36 mm 에서 긋는다.
  절개 정책의 목표 깊이는 −5 mm 다. 약 30 mm 차이의 원인(리그 치수 / 서보 영점)은 확인되지 않았다.
  개복 과제는 장면을 30 mm 내려 맞췄다 (`dofbot_opening_env_cfg.py:86`).

## 2. 정책 입력: 절개선

### 2-1. deploy 가 받는 형식 (비전이 맞출 것)

| 항목 | 값 |
|---|---|
| 형식 | **Start / End 2점.** 웨이포인트 보간은 필요 없다 |
| 좌표계 | 오른팔 베이스 (1절). 피부면 위의 (x, y) |
| 단위 | **mm** |
| 방향 | 선은 방향성이 없다. Start 와 End 를 바꿔도 같은 선이다 |

개복은 "교시한 절개선에서 얼마나 옮겨지고 돌아갔는가"만 쓴다. 그래서 **같은 검출기로 교시 위치의
절개선을 한 번 재서 기준선으로 넘기면**(`reference_ends_mm`), 검출기 좌표계의 원점이 오른팔
베이스와 달라도 상쇄된다. 축 방향(+x 가 왼팔 쪽)은 같아야 한다.
기준선을 넘기지 않으면 교시 절개 동작의 순기구학 값((154, +44) → (168, −29) mm)을 쓰는데,
1 ~ 2 cm 오차가 있을 수 있다.

### 2-2. 유효 범위

| 동작 | 범위 | 벗어나면 | 출처 |
|---|---|---|---|
| 개복 | 교시 절개선 기준 x ±10 mm, y ±20 mm, 회전 ±8° | `PlacementOutOfRange` 예외, 궤적 없음 | `dofbot_opening_env_cfg.py:92-94` |
| 절개 | 길이 40 ~ 70 mm, 방향 60 ~ 105°, 중심 x 135 ~ 235 mm, y ±80 mm, 깊이 5 mm | 학습 분포 밖. 추출 스크립트는 방향만 검사한다 | `dofbot_incision_env_cfg_v3.py:388-396` |

방향은 오른팔 베이스 +x 축에서 +y 쪽으로 잰 각이다. 교시 절개선은 약 101° 다.

### 2-3. 과제 내부 표현 (참고)

절개 과제는 선을 시작점 + 단위 방향 + 길이로 갖고, 표시용으로 5개 점을 둔다
(`dofbot_incision_env_cfg_v3.py:377`, 생성은 `dofbot_incision_env_v3.py:205`).
완주 판정은 점 통과가 아니라 선을 따라 연속으로 그은 거리다.

## 3. 관측

정규화: **있음.** skrl `RunningStandardScaler` (평균·분산, ±5 클리핑). 통계는 체크포인트의
`observation_preprocessor` 에 있고, `weights/*.npz` 에 `obs_mean`, `obs_var` 로 함께 들어 있다.

### 3-1. 절개 (29차원) — `dofbot_incision_env_v3.py:444-466`

| 인덱스 | 이름 | 의미 | 단위 | 실제 로봇에서 얻는 방법 |
|---|---|---|---|---|
| 0–4 | 관절각 | 오른팔 관절 1~5 | rad | 서보 읽기 → 5절 변환 |
| 5–9 | 관절속도 | 같은 관절 | rad/s | 서보 읽기의 차분 (직접 측정 불가) |
| 10–14 | 직전 행동 | 클리핑된 직전 행동 | − | 기록 |
| 15–17 | 칼끝 위치 | 시뮬레이션 월드 좌표 | m | **순기구학 필요** |
| 18–20 | 현재 목표까지 | 선 위 진행 지점 − 칼끝 | m | **순기구학 + 진행 거리 필요** |
| 21–23 | 다음 목표까지 | 진행 지점 15 mm 앞 − 칼끝 | m | 위와 같음 |
| 24 | 진행률 | 그은 거리 / 선 길이 | 0~1 | **절개 판정 필요** |
| 25 | 공구 수직도 | 공구 축과 −z 의 cos | − | 순기구학 필요 |
| 26–27 | 선 방향 | 진행 방향 단위벡터 (x, y) | − | 절개선에서 계산 |
| 28 | 칼날 정렬 | 칼날 모서리와 선 방향의 cos (부호 있음) | − | 순기구학 필요 |

### 3-2. 개복 (43차원) — `dofbot_opening_env.py:366-392`

팔마다 20개(오른팔 0–19, 왼팔 20–39), 공통 3개.

| 팔 내 인덱스 | 이름 | 의미 | 실제 로봇에서 얻는 방법 |
|---|---|---|---|
| 0–3 | 관절각 | 관절 1~4 (rad) | 서보 읽기 |
| 4–7 | 관절속도 | rad/s | 차분 |
| 8–11 | 직전 행동 | − | 기록 |
| 12–14 | 현재 경로점까지 | 경로 위 진행 지점 − 제어점 (m) | **순기구학 + 경로 진행 필요** |
| 15–17 | 앞 경로점까지 | 한 스텝 앞 지점 − 제어점 (m) | 위와 같음 |
| 18 | 진행률 | 구간 가중 경로 진행 0~1 | 위와 같음 |
| 19 | 공구 수직도 | cos | 순기구학 필요 |

공통: 40–41 벌림 방향 단위벡터, 42 유지 진행 비율.

**두 정책 모두 굵게 표시한 항목 때문에 Jetson 에서 폐루프로 돌릴 수 없다.** 순기구학과 과제 상태
(진행 거리)를 시뮬레이션이 계산한다. 그래서 배포 방식이 B 다.

## 4. 행동

| 정책 | 차원 | 의미 | 스케일·클리핑 | 출처 |
|---|---|---|---|---|
| 절개 | 5 | 오른팔 관절 1~5 의 위치 목표 델타 | [−1, 1] 클리핑 × 0.04 rad/스텝, 관절 한계로 제한 | `dofbot_incision_env_v3.py:185`, `dofbot_incision_env_cfg_v3.py:360` |
| 개복 | 8 | 오른팔 관절 1~4, 왼팔 관절 1~4 의 위치 목표 델타 | 같음 | `dofbot_opening_env.py:250` |

- 개복에서 관절 5(손목 롤)와 그리퍼는 행동이 아니다. 교시 값으로 고정한다 (서보 89 / 88, 그리퍼 180).
- 절개에서 그리퍼는 닫힌 값으로 고정한다.
- 제어 주기: 물리 120 Hz, decimation 2 → **정책 60 Hz** (`dofbot_incision_env_cfg_v3.py:129`, `:141`).
- 실제 로봇: 60 Hz 궤적을 6 스텝마다 하나씩 골라 명령으로 쓴다. 재생 스크립트는 그 사이를 보간해
  약 16 Hz 로 보낸다 (`deploy/agx/replay_opening_on_agx.py`, 기본값). 이 보간 재생은 실기 미검증.

## 5. Sim-to-Real 보정

서보 각도(도) ↔ 시뮬레이션 관절각. 구현: `deploy/servo.py`.

| 관절 | 변환 | 서보 범위 | 시뮬레이션 범위 |
|---|---|---|---|
| 1 ~ 4 | `sim_deg = 90 − servo_deg` (방향 반전) | 0 ~ 180° | −90 ~ +90° |
| 5 (손목 롤) | `sim_deg = servo_deg − 225` (반전 아님, 오프셋 +135°) | 0 ~ 270° | −225 ~ +45° (`dofbot_incision_env_cfg_v3.py:569`) |
| 그리퍼 | 정책 출력이 아님. 고정값 | 0 ~ 180° | − |

- 오른팔(버스 8)에서 측정한 값이다 (`deploy/agx/joint_calibration_bus8.json`).
  **왼팔(버스 1)은 따로 측정하지 않았고 같다고 가정했다** (`dofbot_opening_env_cfg.py:62`).
- 영점 오프셋은 측정하지 않았다. 방향과 5번 관절 오프셋만 측정했다.

안전 검사:

| 검사 | 기준 | 어디서 | 실패 시 |
|---|---|---|---|
| 칼끝이 수술대 아래 | 칼끝 z < −10 mm 인 스텝이 하나라도 | `scripts/check_trajectory_safety.py` (GPU, 시뮬레이션) | "실물 재생 금지" 판정 |
| 팔 링크·손끝이 수술대 아래 | 같음 | 같음 | 같음 |
| 관절이 끝단에 붙음 | 가동범위의 2% 이내인 스텝이 하나라도 | 같음 | 같음 |
| (절개) 끝단 여유 적용 | 관절 1~4 를 끝단에서 4° 안쪽으로, 그리퍼 고정. 위 검사 **전에** 적용 | `scripts/apply_servo_margin.py` | − |
| 개복 완료 | 벌림 도달 후 60 스텝 유지 | `scripts/export_opening_library.py`, `export_opening_trajectory.py` | 묶음에서 제외 / 파일 미작성 |
| 양팔 간격 | 링크 뼈대 최단 거리 ≥ 60 mm | 같음 | 같음 |
| 서보 범위 | 관절 1~4: 끝단에서 1° 이상 (절개 변환은 4°), 5: 0~270°, 그리퍼 0~180° | `deploy/safety.py` (Jetson) | `TrajectorySafetyError` |
| 명령 간 변화 | 서보 한 개가 한 행에 25° 이하 | `deploy/safety.py` | `TrajectorySafetyError` |

시뮬레이션이 필요한 검사의 결과는 궤적의 `sim_check` 필드에 실려 온다. `deploy` 는 그 검사를 다시
할 수 없으므로, `sim_check` 가 실패이거나 없는 절개 궤적은 실행하면 안 된다.

## 6. deploy API

```python
from rl_policy.deploy import (
    plan_opening, incision_trajectory_from_csv, validate_trajectory,
    save_trajectory, load_trajectory, trajectory_to_csv,
    PlacementOutOfRange, TrajectorySafetyError,
)
```

`src/` 가 파이썬 경로에 있어야 한다. 의존성은 numpy 뿐이다 (`deploy/requirements.txt`).
Isaac Lab, Isaac Sim, PyTorch 없이 import 된다 (확인: 8절).

| 함수 | 입력 | 반환 | 실패 |
|---|---|---|---|
| `plan_opening(line_start_mm, line_end_mm, reference_ends_mm=None, library=None)` | 절개선 양 끝점 (x, y) mm. 기준선 `((x, y), (x, y))` 선택 | 양팔 궤적 dict (7절), 검사 통과 상태 | `PlacementOutOfRange`, `TrajectorySafetyError` |
| `incision_trajectory_from_csv(path, rate_hz=3.0, grip_deg=180.0, end_stop_margin_deg=4.0, sim_check_passed=None, ...)` | `scripts/export_trajectory.py` 가 쓴 CSV (rad, 60 Hz) | 오른팔 궤적 dict | `ValueError` (sim 검사 실패 시), `TrajectorySafetyError` |
| `validate_trajectory(traj)` | 궤적 dict | 검사 보고 dict | `TrajectorySafetyError` |
| `trajectory_to_csv(traj, path)` | 양팔 궤적 dict | 재생 스크립트용 12열 CSV | − |

예외가 나면 궤적은 반환되지 않는다. **호출하는 쪽은 예외를 잡아 실행을 중단해야 하고, 다른 궤적으로
조용히 대체하면 안 된다.**

dry-run (모터 미구동):

```bash
cd src
python -m rl_policy.deploy.dry_run opening --line 154 44 168 -29 --out out.json
python -m rl_policy.deploy.dry_run opening --line 160 50 171 -22 --ref 154 44 168 -29 --out out.json --csv-out out.csv
python -m rl_policy.deploy.dry_run incision --csv <끝단 여유 적용·안전 검사 통과한 롤아웃.csv> --sim-check passed --out out.json
```

실제 구동용 스크립트(Arm_Lib 필요, Jetson 에서 실행): `deploy/agx/replay_opening_on_agx.py` (양팔, 12열 CSV),
`deploy/agx/replay_on_agx.py` (오른팔 절개, rad CSV). 둘 다 `--dry-run` 이 있다.

## 7. 궤적 형식

가이드 6-4-3 의 형식과 다른 점: 양팔이 동기화된 한 파일이고, 단위가 서보 각도이며, 그리퍼를 포함해
행당 6개 값이다.

```json
{
  "source": "rl_policy",
  "source_detail": "opening PPO policy (...) rolled out in simulation; pre-computed library ..., nearest grid cell",
  "created_at": "2026-10-12T08:16:02",
  "motion": "opening",
  "units": "servo_deg",
  "dt_s": 0.25,
  "joint_order": ["j1", "j2", "j3", "j4", "j5", "grip"],
  "line_robot_base_mm": {"start": [154.0, 44.0], "end": [168.0, -29.0]},
  "reference": "detector",
  "placement": {"requested": {"x_mm": 7.2, "y_mm": 13.0, "turn_deg": 2.4},
                "used":      {"x_mm": 7.5, "y_mm": 12.5, "turn_deg": 2.0}},
  "sim_check": {"opening_completed": true, "min_arm_skeleton_gap_mm_required": 60.0, "where": "scripts/export_opening_library.py"},
  "safety_check": {"passed": true, "rows": 29, "max_step_deg": 6.3, "angles_at_end_stop_margin": 29},
  "arms": {
    "right": {"i2c_bus": 8, "joints": [[85.0, 150.0, 45.0, 27.0, 89.0, 180.0], "..."]},
    "left":  {"i2c_bus": 1, "joints": [[85.0, 151.0, 45.0, 27.0, 88.0, 180.0], "..."]}
  }
}
```

- `joints` 의 각 행은 서보 1~5 와 그리퍼의 목표 각도(도)다. 양팔의 행 수는 같고 같은 행은 같은 시각이다.
- `dt_s` 는 행 사이 권장 시간이다. 개복 0.25 초(전체 약 7.5 초), 절개 1/3 초.
- 절개 궤적은 `arms` 에 `right` 만 있다. 왼팔(홀더)은 궤적에 포함되지 않는다.
- 실제 예: `results/dryrun_opening_taught_20261012.json`.

## 8. 검증 상태

| 항목 | 상태 | 날짜 | 근거 |
|---|---|---|---|
| 절개, 시뮬레이션 평가 64 env | 완료 — 완주 52/64 (81%) | 2026-10-12 | `results/eval_incision_20261012.md` |
| 개복, 시뮬레이션 평가 256 env | 완료 — 256/256 | 2026-10-12 | `results/eval_opening_20261012.md` |
| 개복 궤적 묶음 생성·검사 | 완료 — 1376/1377 사용 가능 | 2026-10-09 | `results/opening_library_20261009.md` |
| deploy 가 Isaac Lab 없이 import | 완료 | 2026-10-12 | `results/dryrun_20261012.md` |
| deploy dry-run, 개복 | 완료 (통과 2건, 범위 밖 거부 1건) | 2026-10-12 | `results/dryrun_20261012.md` |
| deploy dry-run, 절개 | 완료 — 끝단 여유 4° 적용 궤적 2개가 시뮬레이션 안전 검사 "재생 가능", deploy 검사 통과. 여유를 적용하지 않은 롤아웃은 검사에 걸린다 | 2026-10-12 | `results/dryrun_20261012.md`, `results/dryrun_incision_20261012.json` |
| 실제 로봇, 개복 (정책이 만든 궤적) | 구동 완료 — 교시 위치 궤적, 양팔 정상 동작 | 2026-10-08 | 결과 파일 없음 (구두 보고) |
| 실제 로봇, 개복 (교시 위치가 아닌 절개선) | 미검증 | − | − |
| 실제 로봇, 절개 (정책이 만든 궤적) | 구동 완료 — 칼날 미장착 점검. 손목 롤 오프셋 +135° 를 찾아 반영 (5절). 반영 후 칼날 장착 구동은 미수행 | 2026-08-31 | 결과 파일 없음 (구두 보고) |
| 보간 재생 (약 16 Hz) | 미검증 (가짜 서보로 타이밍만 확인) | − | − |
