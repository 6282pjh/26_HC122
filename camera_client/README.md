# camera_client — Orbbec Gemini E 촬영 클라이언트

카메라를 연결한 노트북에서 동작합니다. 웹 콘솔에서 녹음을 시작하면 **현재 수술 장면 1장**을 서버로 올려 월드 모델의 입력으로 쓰고, 녹음이 끝날 때까지 장면을 녹화합니다.

## 동작

```
웹 콘솔 녹음 시작 ─▶ web_server: capture_command.json = {job_id, action: start, status: pending}
                                   ▲ 0.5초마다 GET /capture/pending
capture_client ─ ACK(POST /capture/ack) ─▶ 워밍업 5프레임 버림 → 첫 프레임 PNG 업로드(POST /capture/upload) → 녹화 시작
웹 콘솔 녹음 종료 ─▶ action: stop ─▶ capture_client 녹화 종료 (recordings/{job_id}.mp4)
```

- 같은 요청이 두 번 실행되지 않도록 **ACK가 받아들여진 경우에만** 촬영합니다.
- 첫 프레임 업로드가 실패해도 녹화는 계속합니다.
- 이전 녹화의 STOP을 놓친 상태에서 새 START가 오면 이전 녹화를 자동으로 끝내고 새로 시작합니다.

## 실행

```bash
pip install -r requirements.txt
python capture_client.py find      # 카메라 번호 확인 (Windows DSHOW)
python capture_client.py
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `HC122_SERVER_URL` | `http://127.0.0.1:5050` | 웹 콘솔 주소. 같은 기기에서 돌리면 터널을 거치지 않음 |
| `HC122_CAMERA_INDEX` | 1 | Gemini E RGB 장치 번호 |

학습 데이터 촬영(ffmpeg 녹화)은 [`../world_model/tools/record_take.py`](../world_model/tools/record_take.py)를 사용합니다. OpenCV DSHOW로 열면 화각이 좁아지는 문제가 있어 학습 촬영에는 ffmpeg dshow 입력을 직접 씁니다.
