# -*- coding: utf-8 -*-
"""
selftest.py — 키트가 이 컴퓨터에서 제대로 도는지 자동 점검.

1) 합성 촬영본(1080p 30fps 12초: 정지 3초 → 동작 7초 → 정지 2초)을 만든다
2) inspect_take 로직으로 동작 구간을 찾는다 (정답 3.0~10.0초, 허용 ±0.3초)
3) make_clips로 93프레임 클립 2개(타임랩스 1개 + 실시간 1개)를 만든다
4) 빨간 사각형 위치로 프레임이 원본의 올바른 시각에서 뽑혔는지 확인한다
5) check_clips로 규격 검사
결과물은 _selftest/ 폴더(지워도 됨).
사용: python tools/selftest.py   (또는 run_selftest.bat 더블클릭)
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import KIT_ROOT, find_ffmpeg, motion_energy, smooth  # noqa: E402

W, H, FPS, SEC = 1920, 1080, 30, 12


def x_at(t):
    if t < 3:
        return 400
    if t < 10:
        return 400 + int(1000 * (t - 3) / 7)
    return 1400


def make_synth(path):
    rng = np.random.default_rng(0)
    bg = np.full((H, W, 3), (60, 140, 60), np.uint8)
    cv2.rectangle(bg, (700, 500), (1300, 900), (200, 200, 220), -1)
    frames = []
    for i in range(SEC * FPS):
        t = i / FPS
        f = bg.copy()
        x = x_at(t)
        y = 650 + int(80 * np.sin(t * 3)) if 3 <= t < 10 else 650
        cv2.rectangle(f, (x, y - 40), (x + 120, y + 40), (30, 30, 200), -1)
        frames.append(np.clip(f + rng.normal(0, 2, f.shape), 0, 255).astype(np.uint8))
    ff = find_ffmpeg()
    if ff:
        p = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                              "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-crf", "20",
                              "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()
    else:
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for f in frames:
            vw.write(f)
        vw.release()
    return "ffmpeg" if ff else "opencv"


def red_x(frame):
    m = (frame[:, :, 2] > 150) & (frame[:, :, 1] < 90) & (frame[:, :, 0] < 90)
    xs = np.nonzero(m)[1]
    return int(xs.min() * 2) if len(xs) else -1  # 960 폭 → 1920 기준


def run(cmd):
    print("  $ " + " ".join(cmd[1:]))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return r.returncode, r.stdout + r.stderr


def main():
    out = KIT_ROOT / "_selftest"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    fails = []
    print(f"[환경] Python {sys.version.split()[0]}, OpenCV {cv2.__version__}, NumPy {np.__version__}, "
          f"ffmpeg={find_ffmpeg() or '없음(OpenCV mp4v로 대체)'}")

    take = out / "take_synth.mp4"
    enc = make_synth(take)
    print(f"[1] 합성 촬영본 생성 ({enc}): {take.name}")

    t, e, fps = motion_energy(take)
    s = smooth(e, max(1, int(round(0.2 * fps))))
    base, peak = np.percentile(s[1:], 20), np.percentile(s[1:], 99)
    mov = np.nonzero(s > max(0.02, base + 0.15 * (peak - base)))[0]
    on, off = (t[mov[0]], t[mov[-1]]) if len(mov) else (-1, -1)
    ok = abs(on - 3.0) <= 0.3 and abs(off - 10.0) <= 0.3
    print(f"[2] 동작 탐지: {on:.2f}~{off:.2f}초 (정답 3.00~10.00) → {'OK' if ok else '실패'}")
    if not ok:
        fails.append("동작 탐지")
    rc, txt = run([sys.executable, str(HERE / "inspect_take.py"), str(take)])
    if rc != 0:
        fails.append("inspect_take 실행")
        print(txt)

    ds = out / "dataset"
    py = sys.executable
    for args in (["--action", "incision", "--pos", "p1", "--take", "1", "--start", f"{on - 0.3:.2f}",
                  "--end", f"{off:.2f}"],
                 ["--action", "laparotomy", "--pos", "p1", "--take", "1", "--start", "2.70", "--end", "6.00"]):
        rc, txt = run([py, str(HERE / "make_clips.py"), str(take), *args, "--out", str(ds)])
        if rc != 0:
            fails.append("make_clips 실행")
            print(txt)
    print("[3] make_clips 2개 완료" if "make_clips 실행" not in fails else "[3] make_clips 실패")

    clip = ds / "train" / "videos" / "inc_p1_t01.mp4"
    if clip.exists():
        cap = cv2.VideoCapture(str(clip))
        xs = []
        while True:
            okf, f = cap.read()
            if not okf:
                break
            xs.append(red_x(f))
        cap.release()
        start, end = on - 0.3, off
        ts = [start + i * (end - start) / 92 for i in range(93)]
        err = max(abs(a - x_at(tt)) for a, tt in zip(xs, ts)) if len(xs) == 93 else 1e9
        ok = len(xs) == 93 and err <= 30
        print(f"[4] 프레임 위치: {len(xs)}프레임, 최대 오차 {err:.0f}px (허용 30px) → {'OK' if ok else '실패'}")
        if not ok:
            fails.append("프레임 위치")

    rc, txt = run([py, str(HERE / "check_clips.py"), "--root", str(ds)])
    print("[5] check_clips: " + ("OK (오류 0)" if rc == 0 else "실패"))
    if rc != 0:
        fails.append("check_clips")
        print(txt)

    print("\n" + ("=== 전부 통과 ===" if not fails else f"=== 실패: {', '.join(fails)} ==="))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
