# rl_policy

NVIDIA Isaac Lab + skrl 기반 PPO 절개 정책 학습과 실제 로봇용 정책 실행부

- 매 에피소드 무작위 절개선(웨이포인트 8개) 추종 정책
- 64개 환경 평가 및 Sim-to-Real 전이

## 구성 (업로드 예정)

| 경로 | 내용 |
|---|---|
| `INTERFACE.md` | 정책 입출력·배포 계약 (절개선 입력 형식, 관측·행동 정의, 좌표계, Sim-to-Real 보정, deploy API) |
| `task/`, `agents/` | Isaac Lab 태스크(관측·행동·보상·종료·무작위 절개선), skrl PPO 설정 |
| `scripts/` | 학습 · 평가 · 내보내기 |
| `deploy/` | 실제 로봇용 정책 실행부 — `jetson_control`이 호출 (Isaac Lab 없이 동작) |
| `results/` | 평가 요약 |
