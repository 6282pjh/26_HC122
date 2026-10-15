# -*- coding: utf-8 -*-
"""
check_clips.py — 만든 학습 클립이 규격에 맞는지 전수 검사한다.

검사 항목(클립마다):
  - 파일명 형식 inc|lap_p1~p5_t01
  - 실제 디코딩 프레임 수 = 93, 해상도 = 960x528, fps = 16
  - metas/<이름>.txt 존재 + captions.json의 해당 동작 문장과 글자 하나까지 같은지
  - first_frames/<이름>.png 존재
  - 시작이 정지 상태인지(첫 3프레임 차분이 작아야 함), 클립 안에 움직임이 있는지
전체:
  - train/test에 같은 이름이 있는지(누수)
  - 동작×위치별 개수 표 (부록 B-1 목표: train 배치당·동작당 8개)

사용 예:
  python tools/check_clips.py                  # dataset/ 전체
  python tools/check_clips.py --root D:/hanium/dataset
  python tools/check_clips.py --rewrite-metas  # 캡션 확정 후 metas 일괄 갱신
종료 코드: 오류 있으면 1, 없으면 0 (경고는 0)
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (KIT_ROOT, PREFIX_ACTION, TARGET_FPS, TARGET_FRAMES, TARGET_H,  # noqa: E402
                    TARGET_W, changed_ratio, load_captions, prep_gray)

NAME_RE = re.compile(r"^(inc|lap)_p([1-5])_t(\d{2})$")
TRAIN_TARGET_PER_CELL = 8


def clip_motion(path):
    """클립 전체를 읽어 (프레임 수, 프레임 간 차분 배열)."""
    cap = cv2.VideoCapture(str(path))
    prev, diffs, n = None, [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        g = prep_gray(f, 480)
        if prev is not None:
            diffs.append(changed_ratio(g, prev))
        prev = g
        n += 1
    cap.release()
    return n, np.array(diffs, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="학습 클립 규격 전수 검사")
    ap.add_argument("--root", default=str(KIT_ROOT / "dataset"), help="dataset 루트(train/ test/ 포함)")
    ap.add_argument("--rewrite-metas", action="store_true", help="captions.json 기준으로 metas 전부 다시 쓰기")
    args = ap.parse_args()

    root = Path(args.root)
    caps = load_captions()
    cap_status = caps.get("status", "unknown")
    captions = caps["captions"]

    splits = [d for d in ("train", "test") if (root / d / "videos").is_dir()]
    if not splits:
        print(f"[오류] {root}/train/videos 또는 {root}/test/videos 가 없음")
        return 1

    errors = warns = 0
    names_by_split = {}
    table = Counter()
    print(f"[검사] {root}  (캡션 상태: {cap_status})\n")
    for split in splits:
        vdir = root / split / "videos"
        mdir = root / split / "metas"
        fdir = root / split / "first_frames"
        vids = sorted(vdir.glob("*.mp4"))
        names_by_split[split] = {v.stem for v in vids}
        print(f"== {split}: {len(vids)}개")
        for v in vids:
            msgs_e, msgs_w = [], []
            m = NAME_RE.match(v.stem)
            action = None
            if not m:
                msgs_e.append("파일명 형식 아님(inc|lap_p1~5_t01)")
            else:
                action = PREFIX_ACTION[m.group(1)]
                table[(split, m.group(1), "p" + m.group(2))] += 1

            c = cv2.VideoCapture(str(v))
            w, h = int(c.get(cv2.CAP_PROP_FRAME_WIDTH)), int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(c.get(cv2.CAP_PROP_FPS))
            c.release()
            n, d = clip_motion(v)
            if n != TARGET_FRAMES:
                msgs_e.append(f"프레임 {n}개(≠{TARGET_FRAMES})")
            if (w, h) != (TARGET_W, TARGET_H):
                msgs_e.append(f"해상도 {w}x{h}(≠{TARGET_W}x{TARGET_H})")
            if abs(fps - TARGET_FPS) > 0.01:
                msgs_e.append(f"fps {fps:.3f}(≠{TARGET_FPS})")

            if len(d) >= 4:
                start_e = float(np.mean(d[:3]))
                peak = float(np.percentile(d, 95))
                if peak < 0.05:
                    msgs_w.append(f"움직임 거의 없음(바뀐 픽셀 상위5% {peak:.3f}%)")
                elif start_e > max(0.02, 0.25 * peak):
                    msgs_w.append(f"시작이 정지 상태 아님(첫 3프레임 {start_e:.3f}% / 최대 {peak:.3f}%)")

            meta = mdir / f"{v.stem}.txt"
            if action and args.rewrite_metas:
                mdir.mkdir(parents=True, exist_ok=True)
                meta.write_text(captions[action] + "\n", encoding="utf-8")
            if not meta.exists():
                msgs_e.append("metas 없음")
            elif action:
                txt = meta.read_text(encoding="utf-8").strip()
                if not txt:
                    msgs_e.append("metas 비어 있음")
                elif txt != captions[action].strip():
                    msgs_e.append("metas 문장이 captions.json과 다름(--rewrite-metas)")
            if not (fdir / f"{v.stem}.png").exists():
                msgs_w.append("first_frames 없음")

            errors += bool(msgs_e)
            warns += bool(msgs_w)
            mark = "X " if msgs_e else ("! " if msgs_w else "OK")
            print(f"  [{mark}] {v.name}  {n}f {w}x{h} {fps:.0f}fps"
                  + ("" if not (msgs_e + msgs_w) else "  — " + "; ".join(msgs_e + msgs_w)))
        orphan = {p.stem for p in mdir.glob("*.txt")} - names_by_split[split] if mdir.is_dir() else set()
        if orphan:
            warns += 1
            print(f"  [! ] 영상 없는 metas: {sorted(orphan)}")
        print()

    if "train" in names_by_split and "test" in names_by_split:
        leak = names_by_split["train"] & names_by_split["test"]
        if leak:
            errors += 1
            print(f"[X ] train/test 이름 중복(누수 위험): {sorted(leak)}\n")

    print("[개수표]  (train 목표: 동작·위치당 8개, test: P1~P4 각 1개 + P5 동작당 2개)")
    for split in splits:
        for pre in ("inc", "lap"):
            row = "  ".join(f"p{i}:{table[(split, pre, f'p{i}')]:>2}" for i in range(1, 6))
            print(f"  {split:5s} {pre}  {row}")
    lacking = [f"{pre}_p{i}" for pre in ("inc", "lap") for i in range(1, 5)
               if table[("train", pre, f"p{i}")] < TRAIN_TARGET_PER_CELL]
    if "train" in splits and lacking:
        print(f"  train 목표 미달: {', '.join(lacking)}")

    if cap_status != "final":
        print(f"\n[! ] captions.json 상태 '{cap_status}' — 캡션 확정 전. 학습 전에 확정하고 --rewrite-metas 실행")
    print(f"\n결과: 오류 {errors}개 클립, 경고 {warns}개 클립")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
