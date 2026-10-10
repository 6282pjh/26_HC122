# gpu_server

GPU 서버(RTX A6000)에서 동작하는 웹 승인 콘솔과 예측 파이프라인

- Flask 웹 콘솔 : 명령 → 생성 중 → 검토 3화면, 2단계 승인, Jetson 명령 게시(`/robot/command`, `/robot/ack`)
- 리스너 : Whisper STT → 동작 분류 → LLM 프롬프트 변환 → Cosmos 예측 영상 생성
- 촬영 브리지 : 카메라 노트북과의 촬영 요청 상태 관리
