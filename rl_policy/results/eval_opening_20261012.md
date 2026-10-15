# 개복 정책 평가 — 2026-10-12

| 항목 | 값 |
|---|---|
| 과제 | `Dofbot-Opening-Direct-v0` |
| 정책 | `weights/opening_policy.npz` 의 원본 체크포인트 (학습 실행 2026-10-06, 500 iteration, `agent_64000.pt`) |
| 명령 | `python scripts/eval_opening.py <checkpoint.pt> --device cuda:1 --envs 256` |
| 환경 수 / 시드 | 256 / 1234 |
| 시작 자세 | 모든 환경이 교시 READY 자세에서 시작 (`rsi_prob = 0`) |
| 절개선 위치 | 교시 위치에서 x ±10 mm, y ±20 mm, 회전 ±8° 무작위 |

## 정의

- **완료(COMPLETION)**: 양팔이 교시 경로를 끝까지 따라가 벌린 뒤, 양쪽 끝점이 목표에서 6 mm 이내인
  상태를 60 스텝(1초) 연속 유지한 환경 수. (`dofbot_opening_env.py` 의 `_get_rewards`)
- **벌림 폭**: 양팔 제어점(닫힌 손가락 끝 중점) 사이 거리의 벌림 방향 성분.
- **완료 후 유지**: 완료 판정 이후 남은 스텝 중 유지 조건을 만족한 비율.

## 결과 (스크립트 출력 그대로)

```
COMPLETION 256/256 = 100%   (spread reached and held 60 steps)
  steps: median 164 | fastest 159 | slowest 230
  PATH PROGRESS  right mean 100% median 100% | left mean 100% median 100%
  crashed or strayed before the end: 0/256
  opening width: 135 mm when completed, 136 mm at episode end (taught 133 mm)
  still holding afterwards: 99% of the remaining steps
where each arm stopped (leg of the taught motion it was on at its furthest):
  right READY_OPENING->OPEN_APPROACH:  0  OPEN_APPROACH->INSERT_ABOVE:  0  INSERT_ABOVE->INSERT_01:  0  INSERT_01->INSERT_02: 256
  left  READY_OPENING->OPEN_APPROACH:  0  OPEN_APPROACH->INSERT_ABOVE:  0  INSERT_ABOVE->INSERT_01:  0  INSERT_01->INSERT_02: 256
```

교시 동작 자체를 정책 없이 환경에 재생한 검증(`scripts/replay_json_in_env.py`): 16/16 완료,
끝점 오차 오른팔 3.0 mm / 왼팔 3.5 mm.

Isaac Lab 시뮬레이션 평가 결과다.
