# 절개 정책 평가 — 2026-10-12

| 항목 | 값 |
|---|---|
| 과제 | `Dofbot-Incision-Direct-v3` |
| 정책 | `weights/incision_policy.npz` 의 원본 체크포인트 (학습 실행 2026-09-01, `best_agent.pt`) |
| 명령 | `python scripts/eval_incision.py <checkpoint.pt> --device cuda:0 --envs 64` |
| 환경 수 / 시드 | 64 / 1234 (`scripts/eval_incision.py` 기본값) |
| 에피소드 | 환경당 1회, 최대 898 스텝 (60 Hz) |
| 절개선 | 매 환경 무작위: 길이 40~70 mm, 방향 60~105°, 중심 x 135~235 mm, y ±80 mm |

## 정의

- **완주(COMPLETION)**: 절개선 전체가 연속으로 그어진 환경 수 / 64.
  진행 거리 `_cut_s` 는 네 조건(선까지 수직 거리 < 4 mm, 깊이 오차 < 3 mm,
  공구 수직도 cos > 0.6, 칼날 정렬 |cos| > 0.90)을 만족하는 동안에만, 이미 그은 지점에서
  이어서만 늘어난다. `_cut_s` 가 선 길이 − 3 mm 에 이르면 완주.
  (`dofbot_incision_env_v3.py` 의 `_get_rewards`)
- **절개율(CUT FRACTION)**: 환경별 `_cut_s` / 선 길이의 최댓값.
- **off-line**: 방향 구간별, 칼끝과 절개선 사이 수직 거리의 에피소드 중 최솟값.
- **gates**: 방향 구간별, 각 조건이 열려 있던 스텝 수(898 중)의 최댓값.

## 결과 (스크립트 출력 그대로)

```
COMPLETION 52/64 = 81%
  steps: median 246 | fastest 156 | slowest 808
  mean blade align 0.856 | mean downward 0.811
  CUT FRACTION mean 87% | median 100% | best 100%   <- how much of the line was actually cut
by travel direction (gates are steps open out of 898):
  [ 60, 75)  n=20  done=18   cut= 94% off-line= 0.0mm down=1.00 blade=1.00   gates line=825 z=841 down=855 blade=852
  [ 75, 90)  n=20  done=16   cut= 88% off-line= 0.0mm down=0.99 blade=1.00   gates line=823 z=831 down=855 blade=829
  [ 90,105)  n=24  done=15   cut= 80% off-line= 0.0mm down=1.00 blade=1.00   gates line=824 z=833 down=853 blade=826
```

## 읽는 법

- 완주율 **81% (52/64)**, 절개율 평균 87% / 중앙값 100%.
- 방향 구간별 완주: 60~75° 18/20, 75~90° 16/20, 90~105° 15/24.
- Isaac Lab 시뮬레이션 평가 결과다.
