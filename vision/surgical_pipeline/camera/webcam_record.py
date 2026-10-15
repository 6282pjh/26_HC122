import cv2
import os
from datetime import datetime
from pathlib import Path

CAMERA_INDEX = 0
WIDTH = 640
HEIGHT = 480
FPS = 30

SAVE_DIR = os.environ.get("HC122_VIDEO_DIR", str(Path.home() / "Videos"))
os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = os.path.join(SAVE_DIR, f"webcam_{timestamp}.avi")

# C920을 V4L2로 열기
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print("❌ 카메라 열기 실패")
    raise SystemExit

# C920 출력 형식
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)

if actual_fps <= 0:
    actual_fps = FPS

print("========================================")
print(" C920 WEBCAM RECORD")
print("========================================")
print(f"Camera     : {CAMERA_INDEX}")
print(f"Resolution : {actual_width} x {actual_height}")
print(f"Camera FPS : {actual_fps}")
print(f"Save       : {save_path}")
print("q : 종료 및 저장")
print("========================================")

# AVI + MJPG는 Jetson에서 호환성이 좋음
fourcc = cv2.VideoWriter_fourcc(*"MJPG")

writer = cv2.VideoWriter(
    save_path,
    fourcc,
    FPS,
    (actual_width, actual_height)
)

if not writer.isOpened():
    cap.release()
    print("❌ 녹화 파일 생성 실패")
    raise SystemExit

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret or frame is None:
        print("❌ 웹캠 프레임 읽기 실패")
        break

    # 실제 프레임 저장
    writer.write(frame)
    frame_count += 1

    # 아무 표시 없는 순수 웹캠 화면
    cv2.imshow("C920 Webcam Recording", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
writer.release()
cv2.destroyAllWindows()

print("")
print("========================================")
print(" 녹화 종료")
print("========================================")
print(f"저장 프레임 : {frame_count}")
print(f"저장 위치   : {save_path}")

if os.path.exists(save_path):
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"파일 크기   : {size_mb:.2f} MB")

if frame_count == 0:
    print("❌ 실제로 저장된 프레임이 없습니다.")
else:
    print("✅ 영상 저장 완료")
