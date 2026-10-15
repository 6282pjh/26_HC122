"""
capture_client.py — Orbbec Gemini E 촬영 클라이언트 (카메라를 연결한 노트북에서 실행)

control_server의 웹 콘솔에 0.5초마다 촬영 요청을 물어보고(/capture/pending),
START가 오면 ACK 후 첫 프레임을 업로드(/capture/upload)하고 녹화를 시작, STOP이 오면 녹화를 끝낸다.
업로드한 첫 프레임은 월드 모델의 입력 장면으로 쓰인다.

실행:
  python capture_client.py          # 폴링 시작
  python capture_client.py find     # 연결된 카메라 번호 확인
"""
import os
import sys
import time
import threading
from pathlib import Path

import requests
import cv2


# ============================================================
# 설정
# ============================================================

# 웹 콘솔(app.py)과 같은 기기에서 실행하면 localhost로 직접 통신한다.
# 외부 터널을 거치지 않으므로 네트워크·터널 상태와 무관하게 안정적으로 동작한다.
# 웹 콘솔이 다른 기기에 있으면 HC122_SERVER_URL로 주소를 지정한다.
BASE_URL = os.environ.get("HC122_SERVER_URL", "http://127.0.0.1:5050")
POLL_INTERVAL_SEC = 0.5

# Windows DSHOW 기준 Gemini E RGB 장치 번호. `python capture_client.py find`로 확인
CAMERA_INDEX = int(os.environ.get("HC122_CAMERA_INDEX", "1"))

FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
RECORD_FPS = 30.0

WARMUP_FRAMES = 5
REQUEST_TIMEOUT_SEC = 10

BASE_DIR = Path(__file__).resolve().parent
RECORDINGS_DIR = BASE_DIR / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 카메라 검색
# ============================================================

def find_camera_index(max_check=5):
    found_any = False

    for idx in range(max_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)

        if cap.isOpened():
            ret, frame = cap.read()

            if ret:
                h, w = frame.shape[:2]
                print(f"[INFO] index {idx}: {w}x{h}")
                found_any = True

            cap.release()

    if not found_any:
        print("[WARN] 카메라를 찾지 못했습니다.")


# ============================================================
# 서버 통신
# ============================================================

def get_pending_command():
    r = requests.get(
        f"{BASE_URL}/capture/pending",
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()
    return r.json()


def ack_command(job_id):
    r = requests.post(
        f"{BASE_URL}/capture/ack",
        json={"job_id": job_id},
        timeout=REQUEST_TIMEOUT_SEC,
    )
    r.raise_for_status()

    data = r.json()
    return bool(data.get("accepted"))


def upload_first_frame(job_id, frame):
    ok, encoded = cv2.imencode(".png", frame)

    if not ok:
        raise RuntimeError("PNG 이미지 인코딩 실패")

    files = {
        "image": (
            f"{job_id}.png",
            encoded.tobytes(),
            "image/png",
        )
    }

    r = requests.post(
        f"{BASE_URL}/capture/upload",
        data={"job_id": job_id},
        files=files,
        timeout=30,
    )

    r.raise_for_status()

    print(f"[CAMERA] 현재 환경 이미지 업로드 완료: {job_id}")


# ============================================================
# 실제 Gemini E 녹화
# ============================================================

class CameraRecorder:
    def __init__(self, camera_index):
        self.camera_index = camera_index

        self.cap = None
        self.writer = None
        self.thread = None

        self.stop_event = threading.Event()

        self.job_id = None
        self.output_path = None
        self.running = False


    def start(self, job_id):
        if self.running:
            print(
                f"[WARN] 이미 카메라 녹화 중입니다. "
                f"현재 job={self.job_id}"
            )
            return False

        print(f"[CAMERA] 녹화 준비: {job_id}")

        cap = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not cap.isOpened():
            raise RuntimeError(
                f"카메라 index {self.camera_index}를 열 수 없습니다."
            )

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            FRAME_WIDTH
        )
        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            FRAME_HEIGHT
        )
        cap.set(
            cv2.CAP_PROP_FPS,
            RECORD_FPS
        )

        # 노출 안정화를 위해 초기 프레임 버림
        first_frame = None

        for _ in range(WARMUP_FRAMES):
            ret, frame = cap.read()

            if ret:
                first_frame = frame

        # 실제 사용할 첫 프레임
        ret, frame = cap.read()

        if ret:
            first_frame = frame

        if first_frame is None:
            cap.release()
            raise RuntimeError(
                "Gemini E 첫 프레임 획득 실패"
            )

        actual_h, actual_w = first_frame.shape[:2]

        print(
            f"[CAMERA] 실제 녹화 해상도: "
            f"{actual_w}x{actual_h}"
        )

        output_path = (
            RECORDINGS_DIR /
            f"{job_id}.mp4"
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            RECORD_FPS,
            (actual_w, actual_h),
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(
                "MP4 VideoWriter를 열 수 없습니다."
            )

        # 첫 프레임도 영상에 기록
        writer.write(first_frame)

        # 생성 중 화면에 사용할 현재 환경 사진
        # 이게 실패해도(네트워크 잡음 등) 실제 녹화는 계속 진행되도록
        # 예외를 여기서 흡수함 — "현재 환경" 사진 하나 때문에
        # 전체 녹화가 죽으면 안 되기 때문
        try:
            upload_first_frame(job_id, first_frame)
        except Exception as e:
            print(
                f"[WARN] 현재 환경 이미지 업로드 실패 "
                f"(녹화는 계속 진행): {e}"
            )

        self.cap = cap
        self.writer = writer
        self.job_id = job_id
        self.output_path = output_path

        self.stop_event.clear()
        self.running = True

        self.thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
        )

        self.thread.start()

        print(
            f"[CAMERA] 녹화 START: {job_id}"
        )
        print(
            f"[CAMERA] 저장 위치: {output_path}"
        )

        return True


    def _record_loop(self):
        try:
            frame_interval = 1.0 / RECORD_FPS
            next_frame_time = time.perf_counter()

            while not self.stop_event.is_set():

                ret, frame = self.cap.read()

                if not ret:
                    print(
                        "[WARN] 카메라 프레임 읽기 실패"
                    )
                    time.sleep(0.01)
                    continue

                now = time.perf_counter()

                if now >= next_frame_time:
                    self.writer.write(frame)
                    next_frame_time = now + frame_interval

        except Exception as e:
            print(
                f"[ERROR] 녹화 스레드 오류: {e}"
            )

        finally:

            if self.writer is not None:
                self.writer.release()

            if self.cap is not None:
                self.cap.release()

            print(
                f"[CAMERA] 녹화 리소스 해제: "
                f"{self.job_id}"
            )


    def stop(self, job_id):
        if not self.running:
            print(
                f"[WARN] 현재 녹화 중인 카메라가 없습니다. "
                f"STOP={job_id}"
            )
            return False

        if job_id != self.job_id:
            print(
                f"[WARN] STOP job mismatch: "
                f"running={self.job_id}, "
                f"requested={job_id}"
            )
            return False

        print(
            f"[CAMERA] 녹화 STOP 요청: {job_id}"
        )

        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=5)

        saved_path = self.output_path

        self.running = False
        self.thread = None
        self.cap = None
        self.writer = None
        self.job_id = None
        self.output_path = None

        print(
            f"[CAMERA] 녹화 완료: {saved_path}"
        )

        return True


# ============================================================
# 서버 폴링
# ============================================================

def poll_loop(camera_index):

    recorder = CameraRecorder(camera_index)

    print("=" * 60)
    print("[INFO] Gemini E Capture Client")
    print(f"[INFO] CAMERA_INDEX = {camera_index}")
    print(f"[INFO] SERVER = {BASE_URL}")
    print(f"[INFO] 저장폴더 = {RECORDINGS_DIR}")
    print("=" * 60)

    while True:

        try:
            cmd = get_pending_command()

            if cmd.get("status") != "pending":
                time.sleep(POLL_INTERVAL_SEC)
                continue

            job_id = cmd.get("job_id")
            action = cmd.get("action")

            if not job_id:
                time.sleep(POLL_INTERVAL_SEC)
                continue

            print(
                f"[SERVER] 요청 수신: "
                f"job={job_id}, action={action}"
            )

            # 반드시 먼저 ACK해서 같은 요청 중복 실행 방지
            accepted = ack_command(job_id)

            if not accepted:
                print(
                    "[WARN] ACK 거절됨. "
                    "이미 처리 중인 요청일 수 있습니다."
                )
                time.sleep(POLL_INTERVAL_SEC)
                continue

            if action == "start":

                try:
                    # 이전 녹화가 STOP 누락 등으로 남아 있으면
                    # 새 명령을 막지 않고 자동 종료 후 새 job 시작
                    if recorder.running:

                        if recorder.job_id == job_id:
                            print(
                                f"[WARN] 동일 START 중복 요청 무시: {job_id}"
                            )

                        else:
                            old_job = recorder.job_id

                            print(
                                f"[RECOVERY] 이전 녹화가 남아 있음: "
                                f"{old_job}"
                            )

                            print(
                                f"[RECOVERY] 이전 녹화 자동 종료 후 "
                                f"새 job 시작: {job_id}"
                            )

                            recorder.stop(old_job)
                            recorder.start(job_id)

                    else:
                        recorder.start(job_id)

                except Exception as e:
                    print(
                        f"[ERROR] CAMERA START 실패: {e}"
                    )


            elif action == "stop":

                try:
                    recorder.stop(job_id)

                except Exception as e:
                    print(
                        f"[ERROR] CAMERA STOP 실패: {e}"
                    )

            else:
                print(
                    f"[WARN] 알 수 없는 action: {action}"
                )

        except requests.exceptions.RequestException as e:

            print(
                f"[WARN] 서버 통신 오류: {e}"
            )

        except KeyboardInterrupt:

            print("\n[INFO] 종료합니다.")

            if recorder.running:
                recorder.stop(recorder.job_id)

            break

        except Exception as e:

            print(
                f"[ERROR] 폴링 오류: {e}"
            )

        time.sleep(POLL_INTERVAL_SEC)


# ============================================================
# 실행
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "find":
        find_camera_index()

    else:
        poll_loop(CAMERA_INDEX)