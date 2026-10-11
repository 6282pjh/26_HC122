# -*- coding: utf-8 -*-
"""contact_sheet.py — 평가 영상을 한눈에 비교하는 요약 시트 생성 (O/X 판정 보조용, 판정 자체는 영상을 직접 볼 것).
클립마다 한 장: 행 = GT / base / 체크포인트별 / swap (시드별), 열 = 균등 간격 6프레임.
motion = 연속 프레임 간 평균 절대 차이(0~255 스케일) — 움직임 양의 대략적 지표.
사용: python scripts/contact_sheet.py r16
"""
import csv, sys
from pathlib import Path
import cv2, imageio.v3 as iio, numpy as np

R = sys.argv[1] if len(sys.argv) > 1 else "r16"
root = Path(__file__).resolve().parents[1]
out = root / "eval" / "contact_sheets"; out.mkdir(parents=True, exist_ok=True)
NCOL, W, H = 6, 320, 180

def load(p):
    return np.stack([f for f in iio.imiter(p)])

def row(p, label):
    v = load(p)
    g = v.astype(np.float32).mean(-1)
    motion = float(np.abs(np.diff(g, axis=0)).mean())
    idx = np.linspace(0, len(v) - 1, NCOL).round().astype(int)
    tiles = [cv2.resize(v[i], (W, H), interpolation=cv2.INTER_AREA) for i in idx]
    img = np.concatenate(tiles, axis=1)
    bar = np.full((22, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, f"{label}   frames={len(v)}  motion={motion:.2f}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], axis=0), motion, len(v)

rows_csv = []
for clip in ["inc_p1_t10", "lap_p1_t10"]:
    items = [(root / "data/dataset/test/videos" / f"{clip}.mp4", "GT (test clip)")]
    for s in (0, 1):
        items.append((root / "eval/base_432" / f"{clip}__base__s{s}.mp4", f"base s{s}"))
    for tag in (f"{R}_s504", f"{R}_s1008", f"{R}_s1980", f"{R}_s1980__swap"):
        for s in (0, 1):
            items.append((root / f"eval/e1_{R}" / f"{clip}__{tag}__s{s}.mp4", f"{tag} s{s}"))
    imgs = []
    for p, label in items:
        if not p.exists():
            print("[없음]", p); continue
        im, m, n = row(p, label)
        imgs.append(im); rows_csv.append([clip, label, n, f"{m:.3f}"])
    sheet = np.concatenate(imgs, axis=0)
    dst = out / f"{clip}__{R}.jpg"
    iio.imwrite(dst, sheet, quality=88)
    print("[저장]", dst, sheet.shape)
with open(out / f"motion_{R}.csv", "w", newline="") as fp:
    w = csv.writer(fp); w.writerow(["clip", "video", "frames", "motion"]); w.writerows(rows_csv)
for r in rows_csv:
    print(*r, sep="\t")
