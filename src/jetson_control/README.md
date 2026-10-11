# jetson_control

Jetson AGX Xavier에서 동작하는 양팔 DOFBOT 제어

- 서버 승인 명령 조회 및 ACK 회신
- 상태 시퀀스 검증 : IDLE → INCISION_DONE → OPEN_HOLD → COMPLETE
- I2C Bus 1(좌) / Bus 8(우) 독립 제어

## 실행 모드 (업로드 예정)

| 모드 | 절개 동작 |
|---|---|
| `policy` (기본) | `vision` 절개선 검출 → `rl_policy/deploy` PPO 정책 → 안전 검사 → 실행 |
| `replay` | 기록해 둔 관절 궤적(JSON) 재생 — 발표 시연에서 사용 |
