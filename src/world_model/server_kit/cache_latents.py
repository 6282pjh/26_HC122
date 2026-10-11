# -*- coding: utf-8 -*-
"""
cache_latents.py — 학습 전에 한 번 실행: 클립의 VAE latent와 캡션의 텍스트 임베딩을 디스크에 저장.

왜 필요한가 (RTX 4090 24GB):
  원본 학습 스크립트는 매 step마다 텍스트 인코더(Qwen2.5-VL-7B, bf16 약 16~17GB)와 VAE를
  GPU에서 돌린다. DiT(2B)까지 합치면 24GB를 넘는다. 우리 캡션은 동작별 고정 2문장이고
  VAE 인코딩은 결정적(latent_dist.mean)이라, 미리 한 번 계산해 두면 결과가 원본과 같다.

출력 (<cache_dir>):
  index.json                       해상도·프레임·클립 목록·캡션 매핑
  prompt_embeds_<key>.pt           캡션별 임베딩 [1, 512, 100352] bf16 (약 100MB/개)
  <클립이름>.pt                     정규화된 latent [16, 24, H/8, W/8] float32

전처리는 학습 스크립트의 VideoDataset._get_frames 를 그대로 호출(원본과 동일한 리사이즈·정규화).

사용:
  python cache_latents.py --data_dir $PROJECT_ROOT/data/dataset/train \
      --cache_dir $PROJECT_ROOT/data/cache/train_528x960 --height 528 --width 960
  (메모리 부족 시) --text_on_cpu : 텍스트 인코더를 CPU에서 실행(캡션 2개라 수 분이면 끝)
"""
import argparse
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora_4090 import MockSafetyChecker, VideoDataset  # noqa: E402  (원본과 같은 전처리)
from diffusers import Cosmos2_5_PredictBasePipeline  # noqa: E402
from accelerate import PartialState  # noqa: E402  (VideoDataset이 accelerate 로거를 써서 초기화 필요)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="videos/ 와 metas/ 가 있는 폴더 (dataset/train 등)")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--height", type=int, default=528)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--num_frames", type=int, default=93)
    ap.add_argument("--model_id", default="nvidia/Cosmos-Predict2.5-2B")
    ap.add_argument("--revision", default="diffusers/base/post-trained")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text_on_cpu", action="store_true", help="텍스트 인코더를 CPU에서 실행")
    args = ap.parse_args()
    PartialState()
    assert args.height % 16 == 0 and args.width % 16 == 0, "해상도는 16의 배수"

    data_dir, cache_dir = Path(args.data_dir), Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted((data_dir / "videos").glob("*.mp4"))
    assert vids, f"{data_dir}/videos 에 mp4 없음"

    # 캡션 수집 (학습 스크립트와 같은 방식: metas/<이름>.txt 를 strip)
    caps = {}
    for v in vids:
        txt = (data_dir / "metas" / f"{v.stem}.txt").read_text(encoding="utf-8").strip()
        assert txt, f"빈 캡션: {v.stem}"
        caps[v.stem] = txt
    uniq = sorted(set(caps.values()))
    key_of = {c: hashlib.sha1(c.encode("utf-8")).hexdigest()[:10] for c in uniq}
    print(f"[캐시] 클립 {len(vids)}개, 서로 다른 캡션 {len(uniq)}개")
    for c in uniq:
        print(f"   {key_of[c]}: {c}")
    if len(uniq) != 2:
        print("   [주의] 캡션이 2개가 아님 — captions.json/metas 확인")

    t0 = time.time()
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id, revision=args.revision, torch_dtype=torch.bfloat16, safety_checker=MockSafetyChecker())
    dev = torch.device(args.device)

    # 1) 텍스트 임베딩
    tdev = torch.device("cpu") if args.text_on_cpu else dev
    pipe.text_encoder.to(tdev)
    emb_files = {}
    with torch.no_grad():
        for c in uniq:
            e = pipe._get_prompt_embeds(prompt=[c], device=tdev).to("cpu", torch.bfloat16)
            f = f"prompt_embeds_{key_of[c]}.pt"
            torch.save(e, cache_dir / f)
            emb_files[key_of[c]] = f
            print(f"[텍스트] {key_of[c]} {tuple(e.shape)}")
    pipe.text_encoder.to("cpu")
    pipe.text_encoder = None
    torch.cuda.empty_cache()
    if dev.type == "cuda":
        print(f"[텍스트] GPU 피크 {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}GB")
        torch.cuda.reset_peak_memory_stats(dev)

    # 2) VAE latent (원본 학습 루프와 같은 식: ((mu - mean) * (1/std)).float())
    vae = pipe.vae.to(dev)
    mean, inv_std = pipe.latents_mean.float().to(dev), pipe.latents_std.float().to(dev)
    ds = VideoDataset(dataset_dir=str(data_dir), num_frames=args.num_frames,
                      video_size=[args.height, args.width])
    clips = []
    with torch.no_grad():
        for v in vids:
            video = ds._get_frames(str(v))  # [C, T, H, W], [-1, 1]
            assert video.shape[1] == args.num_frames, f"{v.name}: 프레임 {video.shape[1]}"
            mu = vae.encode(video.unsqueeze(0).to(dev, vae.dtype)).latent_dist.mean
            lat = ((mu - mean) * inv_std).float().squeeze(0).cpu().contiguous()
            torch.save(lat, cache_dir / f"{v.stem}.pt")
            clips.append({"name": v.stem, "latent": f"{v.stem}.pt", "caption_key": key_of[caps[v.stem]]})
            print(f"[VAE] {v.stem} → {tuple(lat.shape)}")
    if dev.type == "cuda":
        print(f"[VAE] GPU 피크 {torch.cuda.max_memory_allocated(dev) / 2**30:.1f}GB")

    index = {
        "meta": {"height": args.height, "width": args.width, "num_frames": args.num_frames,
                 "model_id": args.model_id, "revision": args.revision,
                 "data_dir": str(data_dir), "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        "prompt_embeds": emb_files,
        "captions": {key_of[c]: c for c in uniq},
        "clips": clips,
    }
    (cache_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[완료] {cache_dir}  ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
