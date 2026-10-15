# -*- coding: utf-8 -*-
r"""
record_take.py — Gemini E로 학습용 촬영 원본(take)을 녹화한다.

원본: 1차 학습 촬영 스크립트 record_video_ffmpeg.py (2026-08-31)
  - ffmpeg dshow 직접 입력(OpenCV/DSHOW 화각 축소 문제 우회), 1920x1080 mjpeg 30fps → H.264
바뀐 점(2026-10-04):
  1) -rtbufsize 1024M : "real-time buffer too full ... frame dropped!" 방지
  2) 어디서 실행해도 항상 <키트>\raw\ 에 저장
  3) 같은 이름이 있으면 덮어쓰지 않고 중단(-y → -n)
  4) 파일명 규칙 검사: take_p1_t01.mp4 (p1~p5, t01~t99)
  5) 녹화 중 q 키로 먼저 끝낼 수 있음(ffmpeg 기본 기능)
  인코딩 설정(libx264 기본 crf 23, yuv420p)은 원본과 동일

사용:
  python tools\record_take.py --check                 # 구도 확인 사진 → raw\preview_check.png
  python tools\record_take.py take_p1_t01 --duration 60
  python tools\record_take.py --next p1 --duration 60   # p1의 다음 번호 자동
"""
import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEVICE_NAME = "Orbbec Gemini E RGB Camera"
KIT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = KIT_ROOT / "raw"
NAME_RE = re.compile(r"^take_p[1-5]_t\d{2}$")

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(errors="replace")
    except Exception:
        pass


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if not exe:
        print("[오류] ffmpeg를 찾을 수 없음 (PATH 확인)")
        sys.exit(1)
    return exe


def dshow_input(fps=None):
    args = ["-f", "dshow", "-rtbufsize", "1024M", "-video_size", "1920x1080"]
    if fps:
        args += ["-framerate", str(fps)]
    args += ["-vcodec", "mjpeg", "-i", f"video={DEVICE_NAME}"]
    return args


def check_frame():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "preview_check.png"
    cmd = [ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", *dshow_input(),
           "-frames:v", "1", "-update", "1", str(out)]
    subprocess.run(cmd, check=True)
    print(f"[완료] 구도 확인 사진: {out}")


def next_name(pos):
    pos = pos.lower() if pos.lower().startswith("p") else "p" + pos
    nums = [int(p.stem.split("_t")[-1]) for p in RAW_DIR.glob(f"take_{pos}_t??.mp4")]
    return f"take_{pos}_t{(max(nums) + 1 if nums else 1):02d}"


def record(name, duration, fps):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"{name}.mp4"
    if out.exists():
        print(f"[중단] 이미 있음: {out}\n  → 번호를 올려서 다시 (또는 --next p1)")
        sys.exit(1)
    cmd = [ffmpeg_exe(), "-n", "-hide_banner", "-loglevel", "warning", "-stats", *dshow_input(fps),
           "-t", str(duration), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)]
    print(f"[녹화] {out.name}  ({duration}초, 먼저 끝내려면 q)")
    subprocess.run(cmd, check=True)
    print(f"[완료] {out}")
    print(f"  다음: python tools\\inspect_take.py raw\\{out.name} --action incision "
          f"--pos {name.split('_')[1]} --take {int(name[-2:])} --plot raw\\{name}.png")


def main():
    ap = argparse.ArgumentParser(description="Gemini E 학습용 원본 녹화 (항상 <키트>\\raw\\에 저장)")
    ap.add_argument("name", nargs="?", help="take_p1_t01 형식(.mp4 생략 가능)")
    ap.add_argument("--next", metavar="POS", help="해당 위치의 다음 번호로 자동 이름 (예: --next p1)")
    ap.add_argument("--duration", type=float, default=60.0, help="녹화 시간(초), 기본 60")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--check", action="store_true", help="녹화 대신 구도 확인 사진 1장")
    args = ap.parse_args()

    if args.check:
        check_frame()
        return 0
    if args.next:
        name = next_name(args.next)
    elif args.name:
        name = args.name[:-4] if args.name.lower().endswith(".mp4") else args.name
    else:
        ap.print_help()
        return 1
    if not NAME_RE.match(name):
        print(f"[중단] 파일명 규칙 아님: {name}  (예: take_p1_t01, 위치 p1~p5)")
        return 1
    record(name, args.duration, args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
