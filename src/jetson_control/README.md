# jetson_control

Jetson AGX Xavier에서 동작하는 양팔 DOFBOT 제어

- 서버 승인 명령 조회 및 ACK 회신
- 상태 시퀀스 검증 : IDLE → INCISION_DONE → OPEN_HOLD → COMPLETE
- I2C Bus 1(좌) / Bus 8(우) 독립 제어
