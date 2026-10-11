# -*- coding: utf-8 -*-
"""
한이음 2차 학습 데이터 키트 공통 모듈.

학습 클립 규격(round2_finetune_plan.md 부록 B-1 기준):
  - 960x528 (가로 구도, 16의 배수), 정확히 93프레임, 16fps 표기
  - 정지 상태에서 시작
  - 파일명: inc_p1_t01.mp4 + metas/inc_p1_t01.txt
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# Windows 콘솔(cp949)에서 '≈', '×' 같은 기호를 출력하다 죽지 않도록
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass

# ---- 학습 클립 규격 ---------------------------------------------------------
TARGET_W = 960
TARGET_H = 528
TARGET_FRAMES = 93
TARGET_FPS = 16
CLIP_SECONDS = TARGET_FRAMES / TARGET_FPS  # 5.8125초

ACTION_PREFIX = {"incision": "inc", "laparotomy": "lap"}
PREFIX_ACTION = {v: k for k, v in ACTION_PREFIX.items()}

KIT_ROOT = Path(__file__).resolve().parent.parent
CAPTIONS_PATH = KIT_ROOT / "captions.json"


def load_captions():
    """captions.json에서 동작별 고정 캡션을 읽는다."""
    with open(CAPTIONS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data


def caption_for(action):
    data = load_captions()
    if action not in data["captions"]:
        raise KeyError(f"captions.json에 '{action}' 캡션이 없음")
    return data["captions"][action], data.get("status", "unknown")


def clip_name(action, pos, take):
    """inc_p1_t01 형식 이름을 만든다."""
    prefix = ACTION_PREFIX[action]
    pos = pos.lower()
    if not pos.startswith("p"):
        pos = "p" + pos
    return f"{prefix}_{pos}_t{int(take):02d}"


def probe(path):
    """OpenCV로 영상 기본 정보를 읽는다(프레임 수는 메타데이터 값)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없음: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames_meta": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration"] = info["frames_meta"] / info["fps"] if info["fps"] > 0 else 0.0
    return info


def count_frames_decoded(path):
    """메타데이터가 아니라 실제로 디코딩해서 프레임 수를 센다(정확값)."""
    cap = cv2.VideoCapture(str(path))
    n = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        n += 1
    cap.release()
    return n


def find_ffmpeg():
    """ffmpeg 실행 파일 경로. PATH → imageio-ffmpeg 순서로 찾고, 없으면 None."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg  # noqa: WPS433
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def parse_roi(text):
    """'x,y,w,h' 문자열 → 정수 튜플. None이면 None."""
    if not text:
        return None
    parts = [int(float(v)) for v in text.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI는 x,y,w,h 4개 값이어야 함")
    return tuple(parts)


def crop_to_aspect(frame, roi=None, aspect=TARGET_W / TARGET_H, crop_y_bias=0.5):
    """
    (선택) ROI로 먼저 자른 뒤, 960:528 비율이 되도록 가운데 기준으로 잘라낸다.
    crop_y_bias: 세로로 잘라낼 때 위(0.0)~가운데(0.5)~아래(1.0) 중 어디를 남길지.
    """
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y + h, x:x + w]
    h, w = frame.shape[:2]
    if w / h > aspect:  # 너무 넓음 → 좌우 자르기
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        frame = frame[:, x0:x0 + nw]
    elif w / h < aspect:  # 너무 높음 → 위아래 자르기
        nh = int(round(w / aspect))
        y0 = int(round((h - nh) * crop_y_bias))
        frame = frame[y0:y0 + nh, :]
    return frame


def to_target(frame, roi=None, crop_y_bias=0.5):
    frame = crop_to_aspect(frame, roi=roi, crop_y_bias=crop_y_bias)
    return cv2.resize(frame, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)


PIX_THR = 6.0  # 0~255 밝기 기준. 이보다 크게 바뀐 픽셀만 "움직임"으로 셈


def changed_ratio(gray, prev, pix_thr=PIX_THR):
    """이전 프레임 대비 밝기가 pix_thr보다 크게 바뀐 픽셀 비율(%)."""
    return float(100.0 * np.mean(np.abs(gray - prev) > pix_thr))


def prep_gray(frame, scale_w):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (scale_w, max(1, int(h * scale_w / w))), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)


def motion_energy(path, roi=None, scale_w=480, max_seconds=None):
    """
    프레임별 움직임 양을 구한다.
    반환: (times[s], energy[%], fps)
    energy[i] = frame[i-1]→frame[i] 사이에 크게 바뀐 픽셀 비율(%) (i=0은 0)

    평균 절대차 대신 "바뀐 픽셀 비율"을 쓰는 이유:
      정지 화면에서도 코덱(x264)이 센서 노이즈를 프레임마다 다르게 압축하면
      평균 절대차가 작은 동작 수준까지 올라감(합성 영상 실측). 픽셀별 임계값을 두면
      이런 자잘한 노이즈는 0으로 떨어지고, 실제로 물체가 움직인 영역만 남는다.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev = None
    energies = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_seconds is not None and i / fps > max_seconds:
            break
        if roi is not None:
            x, y, w, h = roi
            frame = frame[y:y + h, x:x + w]
        gray = prep_gray(frame, scale_w)
        energies.append(0.0 if prev is None else changed_ratio(gray, prev))
        prev = gray
        i += 1
    cap.release()
    e = np.array(energies, dtype=np.float32)
    t = np.arange(len(e)) / fps
    return t, e, fps


def smooth(x, win):
    if win <= 1 or len(x) == 0:
        return x
    k = np.ones(win, dtype=np.float32) / win
    return np.convolve(x, k, mode="same")
