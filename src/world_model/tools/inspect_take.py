# -*- coding: utf-8 -*-
"""
inspect_take.py — 촬영 원본(take) 1개를 분석해서 정지/동작 구간을 찾고,
93프레임 클립으로 자를 위치(start/end)와 make_clips.py 명령을 제안한다.

사용 예:
  python tools/inspect_take.py raw/take_inc_p1_01.mp4
  python tools/inspect_take.py raw/take.mp4 --roi 600,400,900,680 --plot out/take.png
  python tools/inspect_take.py raw/take.mp4 --action incision --pos p1 --take 1

원리:
  1) 프레임을 작게 줄이고 흑백+블러 → 이전 프레임보다 밝기가 6 이상 바뀐 픽셀 비율(%)
  2) 이동평균으로 다듬은 뒤, 임계값보다 크면 "동작", 작으면 "정지"
  3) 동작 구간마다 "동작 시작 preroll초 전(=정지 상태)"부터 "동작 끝"까지를 클립 후보로 제안
     - 길이 T <= 5.75초    → 16fps 그대로 추출(실시간)
     - 길이 T >  5.75초    → 추출 fps = 92 / T (타임랩스) → 동작 전체가 93프레임 안에 들어감
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CLIP_SECONDS, TARGET_FPS, TARGET_FRAMES, motion_energy,  # noqa: E402
                    parse_roi, probe, smooth)


def find_segments(mask, t, min_len_s, min_gap_s):
    """True 구간 목록 [(start_s, end_s)]. 짧은 끊김은 이어 붙이고 짧은 구간은 버린다."""
    segs = []
    start = None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            segs.append([start, i - 1])
            start = None
    if start is not None:
        segs.append([start, len(mask) - 1])
    if not segs:
        return []
    dt = t[1] - t[0] if len(t) > 1 else 1 / 30
    merged = [segs[0]]
    for s, e in segs[1:]:
        if (s - merged[-1][1]) * dt < min_gap_s:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(t[s], t[e]) for s, e in merged if (e - s) * dt >= min_len_s]


def main():
    ap = argparse.ArgumentParser(description="촬영 원본의 정지/동작 구간 탐지 + 자를 위치 제안")
    ap.add_argument("video")
    ap.add_argument("--roi", help="분석할 영역 x,y,w,h (원본 픽셀). 예: 오른팔 주변만 보기")
    ap.add_argument("--threshold", type=float, help="동작 판정 임계값(기본: 자동)")
    ap.add_argument("--smooth", type=float, default=0.2, help="이동평균 창 길이(초), 기본 0.2")
    ap.add_argument("--min-motion", type=float, default=0.5, help="이보다 짧은 동작은 무시(초)")
    ap.add_argument("--min-still", type=float, default=0.4, help="이보다 짧은 정지는 동작에 포함(초)")
    ap.add_argument("--preroll", type=float, default=0.3, help="동작 시작 몇 초 전부터 자를지(정지 프레임 확보)")
    ap.add_argument("--plot", help="motion energy 그래프 PNG 저장 경로(matplotlib 필요)")
    ap.add_argument("--action", choices=["incision", "laparotomy"], help="명령 제안용")
    ap.add_argument("--pos", default="p1", help="명령 제안용 절개선 위치 (p1~p5)")
    ap.add_argument("--take", type=int, default=1, help="명령 제안용 촬영 번호")
    args = ap.parse_args()

    path = Path(args.video)
    info = probe(path)
    print(f"[정보] {path.name}: {info['width']}x{info['height']}, {info['fps']:.2f}fps, "
          f"{info['frames_meta']}프레임(메타), {info['duration']:.2f}초")
    if (info["width"], info["height"]) != (1920, 1080):
        print("  [주의] 원본 규격(1080p)이 아님 — 촬영 설정 확인")
    if abs(info["fps"] - 30) > 1:
        print("  [주의] 원본 fps가 30이 아님 — 타임랩스 계산은 이 fps 기준으로 진행")

    roi = parse_roi(args.roi)
    t, e, fps = motion_energy(path, roi=roi)
    if len(e) < 3:
        print("[오류] 프레임이 너무 적음")
        return 1
    win = max(1, int(round(args.smooth * fps)))
    s = smooth(e, win)

    if args.threshold is not None:
        thr = args.threshold
    else:
        base = float(np.percentile(s[1:], 20))
        peak = float(np.percentile(s[1:], 99))
        thr = max(0.02, base + 0.15 * (peak - base))
        if peak < 0.05:
            print("[주의] 전체적으로 움직임이 거의 없음(정지 영상일 가능성)")
    print(f"[임계값] 바뀐 픽셀 {thr:.3f}%  (하위20% ≈ {np.percentile(s[1:], 20):.3f}%, "
          f"상위1% ≈ {np.percentile(s[1:], 99):.3f}%)")

    moving = find_segments(s > thr, t, args.min_motion, args.min_still)
    total = t[-1] + 1 / fps

    # 정지 구간 = 동작 구간의 여집합
    stills = []
    cur = 0.0
    for ms, me in moving:
        if ms - cur > 0.05:
            stills.append((cur, ms))
        cur = me
    if total - cur > 0.05:
        stills.append((cur, total))

    print("\n[타임라인]")
    events = [("정지", a, b) for a, b in stills] + [("동작", a, b) for a, b in moving]
    for kind, a, b in sorted(events, key=lambda x: x[1]):
        print(f"  {kind}  {a:6.2f} ~ {b:6.2f}초  ({b - a:5.2f}초)")

    if not moving:
        print("\n동작 구간을 찾지 못함 → --threshold를 낮추거나 --roi로 팔 주변만 지정해 봐.")
        return 0

    print(f"\n[클립 후보]  (클립 = {TARGET_FRAMES}프레임 @ {TARGET_FPS}fps 표기 = {CLIP_SECONDS:.4f}초)")
    for k, (ms, me) in enumerate(moving, 1):
        start = max(0.0, ms - args.preroll)
        span = me - start
        if span <= (TARGET_FRAMES - 1) / TARGET_FPS:
            ext_fps = TARGET_FPS
            end = start + (TARGET_FRAMES - 1) / TARGET_FPS
            note = "실시간 16fps (동작 뒤 정지 포함)"
        else:
            ext_fps = (TARGET_FRAMES - 1) / span
            end = me
            note = f"타임랩스 ×{TARGET_FPS / ext_fps:.2f}"
        if end > total:
            note += "  [주의] 영상 끝을 넘음 → make_clips가 중단함(촬영을 더 길게)"
        idx = min(len(s) - 1, int(round(start * fps)))
        still_at_start = s[idx] <= thr
        print(f"  #{k}: 동작 {ms:.2f}~{me:.2f}초({me - ms:.2f}초) → start={start:.2f}, end={end:.2f}, "
              f"추출 {ext_fps:.2f}fps, {note}"
              + ("" if still_at_start else "  [주의] start 시점이 정지 상태가 아님(preroll 조정)"))
        if args.action:
            print(f"      python tools/make_clips.py \"{path}\" --action {args.action} --pos {args.pos} "
                  f"--take {args.take} --start {start:.2f} --end {me:.2f}"
                  + (f" --roi {args.roi}" if args.roi else ""))
    if len(moving) > 1:
        print("\n  ※ 동작이 여러 덩어리면(예: 칼 집기 → 진입 → 절개), 학습에 쓸 구간을 골라서 "
              "--start는 첫 덩어리 직전 정지, --end는 마지막 덩어리 끝으로 직접 지정해도 됨.")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[그래프] matplotlib 없음 → 생략 (pip install matplotlib)")
            return 0
        fig, ax = plt.subplots(figsize=(12, 3.2))
        ax.plot(t, e, lw=0.6, alpha=0.4, label="energy")
        ax.plot(t, s, lw=1.4, label="smoothed")
        ax.axhline(thr, ls="--", lw=1, color="k", label="threshold")
        for ms, me in moving:
            ax.axvspan(ms, me, alpha=0.15, color="tab:red")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("changed pixels [%]")
        ax.set_title(path.name)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=120)
        print(f"[그래프] 저장: {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
