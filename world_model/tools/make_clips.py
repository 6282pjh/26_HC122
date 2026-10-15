# -*- coding: utf-8 -*-
"""
make_clips.py — 촬영 원본에서 학습 클립 1개를 만든다.

출력(기본 dataset/<split>/ 아래):
  videos/inc_p1_t01.mp4        960x528, 정확히 93프레임, 16fps
  metas/inc_p1_t01.txt         동작별 고정 캡션(captions.json)
  first_frames/inc_p1_t01.png  첫 프레임(추론 입력 이미지 비교용)
  clips_log.csv                어떤 원본의 몇 초를 어떻게 잘랐는지 기록

사용 예:
  python tools/make_clips.py raw/take.mp4 --action incision --pos p1 --take 1 --start 12.10 --end 19.50
  python tools/make_clips.py raw/take.mp4 --action laparotomy --pos p5 --take 1 --start 3.0 --end 9.2 --split test

프레임 선택:
  T = end - start
  T <= 5.75초   → 16fps로 추출(실시간), start부터 93프레임(5.75초 구간)
  T >  5.75초   → 추출 fps = 92 / T (타임랩스), 첫 프레임=start, 마지막 프레임=end
                 (93프레임 사이 간격은 92개. "93 ÷ T"가 아니라 "92 ÷ T")
  i번째 프레임 = 원본에서 (start + i / 추출fps)초에 가장 가까운 프레임
"""
import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLIP_SECONDS, KIT_ROOT, TARGET_FPS, TARGET_FRAMES, TARGET_H,  # noqa: E402
                    TARGET_W, caption_for, clip_name, find_ffmpeg, parse_roi, probe, to_target)


def pick_frames(path, src_fps, times, roi, crop_y_bias):
    """times(초) 각각에 가장 가까운 원본 프레임을 960x528로 변환해 반환."""
    want = [int(round(tt * src_fps)) for tt in times]
    cap = cv2.VideoCapture(str(path))
    # cap.set(POS_FRAMES)는 코덱/백엔드(특히 Windows)에 따라 키프레임 단위로 빗나갈 수 있어서
    # 처음부터 순서대로 grab()하며 건너뛴다(느리지만 프레임 위치가 정확함).
    idx = 0
    frames = []
    last = None
    repeated = 0
    j = 0
    while j < len(want):
        if idx < want[j]:
            if not cap.grab():
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        while j < len(want) and want[j] == idx:
            last = to_target(frame, roi=roi, crop_y_bias=crop_y_bias)
            frames.append(last)
            j += 1
        idx += 1
    cap.release()
    # 영상 끝을 넘은 부분만 마지막 프레임으로 채운다.
    while len(frames) < len(want) and last is not None:
        frames.append(last)
        repeated += 1
    return frames, repeated


def write_ffmpeg(ffmpeg, frames, out_path, crf):
    cmd = [ffmpeg, "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{TARGET_W}x{TARGET_H}", "-r", str(TARGET_FPS),
           "-i", "-",
           "-frames:v", str(len(frames)), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-r", str(TARGET_FPS), "-movflags", "+faststart", str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg 인코딩 실패")


def write_opencv(frames, out_path):
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (TARGET_W, TARGET_H))
    for f in frames:
        vw.write(f)
    vw.release()


def main():
    ap = argparse.ArgumentParser(description="촬영 원본 → 93f/16fps/960x528 학습 클립 + metas + 첫 프레임")
    ap.add_argument("video")
    ap.add_argument("--action", required=True, choices=["incision", "laparotomy"])
    ap.add_argument("--pos", required=True, help="절개선 위치 p1~p5")
    ap.add_argument("--take", required=True, type=int, help="촬영 번호(1부터)")
    ap.add_argument("--start", required=True, type=float, help="자르기 시작(초) — 정지 상태여야 함")
    ap.add_argument("--end", type=float, help="동작 끝(초). 생략하면 start부터 5.8125초(실시간)")
    ap.add_argument("--split", default="train", choices=["train", "test"], help="test = 학습 제외(held-out)")
    ap.add_argument("--out", default=str(KIT_ROOT / "dataset"), help="출력 루트 폴더")
    ap.add_argument("--roi", help="디지털 크롭 영역 x,y,w,h (원본 픽셀). 학습·추론에 같은 값 써야 함")
    ap.add_argument("--crop-y-bias", type=float, default=0.5,
                    help="16:9→960:528 세로 자를 때 위치(0=위 남김, 0.5=가운데, 1=아래 남김)")
    ap.add_argument("--crf", type=int, default=16, help="x264 화질(낮을수록 고화질), 기본 16")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-pad", action="store_true", help="영상 끝을 넘으면 마지막 프레임 반복 허용(기본: 중단)")
    args = ap.parse_args()

    src = Path(args.video)
    info = probe(src)
    src_fps = info["fps"]
    if src_fps <= 0:
        print("[오류] 원본 fps를 읽을 수 없음")
        return 1

    if args.end is None:
        span = CLIP_SECONDS
        ext_fps = TARGET_FPS
    else:
        span = args.end - args.start
        if span <= 0:
            print("[오류] end가 start보다 커야 함")
            return 1
        # 93프레임 = 간격 92개 → 첫 프레임 start, 마지막 프레임 end가 되려면 fps = 92 / T
        ext_fps = TARGET_FPS if span <= (TARGET_FRAMES - 1) / TARGET_FPS else (TARGET_FRAMES - 1) / span
    if ext_fps > src_fps + 1e-6:
        print(f"[주의] 추출 fps({ext_fps:.2f})가 원본 fps({src_fps:.2f})보다 큼 → 같은 프레임 중복 발생")

    times = [args.start + i / ext_fps for i in range(TARGET_FRAMES)]
    if times[-1] > info["duration"]:
        print(f"[주의] 마지막 프레임 시각 {times[-1]:.2f}초가 원본 길이 {info['duration']:.2f}초를 넘음")

    name = clip_name(args.action, args.pos, args.take)
    out_root = Path(args.out) / args.split
    vdir, mdir, fdir = out_root / "videos", out_root / "metas", out_root / "first_frames"
    for d in (vdir, mdir, fdir):
        d.mkdir(parents=True, exist_ok=True)
    vpath = vdir / f"{name}.mp4"
    if vpath.exists() and not args.overwrite:
        print(f"[중단] 이미 있음: {vpath}  (덮어쓰려면 --overwrite)")
        return 1

    roi = parse_roi(args.roi)
    frames, repeated = pick_frames(src, src_fps, times, roi, args.crop_y_bias)
    if len(frames) != TARGET_FRAMES:
        print(f"[오류] 프레임 {len(frames)}개만 확보(필요 {TARGET_FRAMES}) — start가 영상 길이를 넘었는지 확인")
        return 1
    if repeated and not args.allow_pad:
        print(f"[중단] 영상 끝을 넘어 {repeated}프레임이 모자람 → start를 앞당기거나 --end 지정 "
              f"(정지 프레임 반복으로 채우려면 --allow-pad)")
        return 1
    if repeated:
        print(f"[주의] 영상 끝을 넘어 마지막 프레임을 {repeated}번 반복함")

    ffmpeg = find_ffmpeg()
    if ffmpeg:
        write_ffmpeg(ffmpeg, frames, vpath, args.crf)
        encoder = "ffmpeg/libx264"
    else:
        print("[주의] ffmpeg 없음 → OpenCV mp4v로 저장 (pip install imageio-ffmpeg 권장)")
        write_opencv(frames, vpath)
        encoder = "opencv/mp4v"

    caption, cap_status = caption_for(args.action)
    (mdir / f"{name}.txt").write_text(caption + "\n", encoding="utf-8")
    cv2.imwrite(str(fdir / f"{name}.png"), frames[0])

    log = out_root / "clips_log.csv"
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["name", "split", "action", "source", "start_s", "end_s", "extract_fps", "speedup",
                        "roi", "crop_y_bias", "encoder", "caption_status", "created"])
        w.writerow([name, args.split, args.action, str(src), f"{args.start:.3f}", f"{times[-1]:.3f}",
                    f"{ext_fps:.4f}", f"{TARGET_FPS / ext_fps:.3f}", args.roi or "", args.crop_y_bias,
                    encoder, cap_status, dt.datetime.now().isoformat(timespec="seconds")])

    print(f"[완료] {vpath}")
    print(f"  {args.start:.2f}~{times[-1]:.2f}초 원본 구간, 추출 {ext_fps:.2f}fps (×{TARGET_FPS / ext_fps:.2f}), "
          f"{TARGET_FRAMES}프레임 @ {TARGET_FPS}fps, {encoder}")
    if cap_status != "final":
        print(f"  [주의] captions.json 상태가 '{cap_status}' — 캡션 확정 후 metas를 다시 써야 함 "
              f"(python tools/check_clips.py --rewrite-metas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
