# -*- coding: utf-8 -*-
"""
infer_lora.py — 학습한 LoRA(또는 base)로 Image2World 영상을 생성해 평가용으로 저장.

기준: 포크의 examples/cosmos/eval_cosmos_predict25_lora.py 와 같은 호출
      (from_pretrained → load_lora_weights → fuse_lora → pipe(image=, prompt=, ...))
4090 대응: 전체 파이프라인(텍스트 인코더 16~17GB 포함)을 GPU에 한꺼번에 못 올리므로
      enable_model_cpu_offload() 로 텍스트 인코더 → DiT → VAE 순서대로 GPU에 올렸다 내림.
      (원본 eval 스크립트는 seed를 pipe에 넘기지 않음 → 여기서는 generator로 고정)

입력: dataset/test (또는 train) 폴더의 first_frames/<이름>.png + metas/<이름>.txt
출력: <out_dir>/<이름>__<tag>__s<seed>.mp4, <out_dir>/timing.csv

예:
  # base vs LoRA, test 클립 전부, 시드 2개
  python infer_lora.py --data_dir $PROJECT_ROOT/data/dataset/test --out_dir out/base --seeds 0 1
  python infer_lora.py --data_dir $PROJECT_ROOT/data/dataset/test --out_dir out/e1_ck500 \
      --lora_dir $PROJECT_ROOT/lora_checkpoints/20261005_e1_r16/checkpoint-<epoch> --seeds 0 1
  # 동작 구분 테스트: incision 첫 프레임에 laparotomy 캡션(또는 반대)을 넣음
  python infer_lora.py ... --swap_prompt
"""
import argparse
import csv
import time
from pathlib import Path

import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video, load_image


class MockSafetyChecker:  # 원본 eval 스크립트와 동일
    def to(self, *a, **k):
        return self

    def check_text_safety(self, *a, **k):
        return True

    def check_video_safety(self, video):
        return video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="first_frames/ 와 metas/ 가 있는 폴더")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--lora_dir", default=None, help="pytorch_lora_weights.safetensors 가 있는 폴더 (없으면 base)")
    ap.add_argument("--tag", default=None, help="파일명 태그 (기본: base 또는 lora 폴더명)")
    ap.add_argument("--height", type=int, default=528)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0, help="pipeline 기본값 7.0")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--only", nargs="*", default=None, help="이 이름들만 (예: inc_p1_t10)")
    ap.add_argument("--swap_prompt", action="store_true", help="inc↔lap 캡션을 바꿔 넣기 (동작 구분 테스트)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--vae_tiling", action="store_true", help="VAE 디코딩 OOM 시 사용")
    ap.add_argument("--model_id", default="nvidia/Cosmos-Predict2.5-2B")
    ap.add_argument("--revision", default="diffusers/base/post-trained")
    args = ap.parse_args()

    data, out = Path(args.data_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = sorted((data / "first_frames").glob("*.png"))
    if args.only:
        frames = [f for f in frames if f.stem in set(args.only)]
    assert frames, f"{data}/first_frames 에 png 없음"
    caps = {f.stem: (data / "metas" / f"{f.stem}.txt").read_text(encoding="utf-8").strip() for f in frames}
    by_prefix = {}
    for k, c in caps.items():
        by_prefix.setdefault(k.split("_")[0], c)
    tag = args.tag or (Path(args.lora_dir).name if args.lora_dir else "base")

    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        args.model_id, revision=args.revision, torch_dtype=torch.bfloat16, safety_checker=MockSafetyChecker())
    if args.lora_dir:
        pipe.load_lora_weights(args.lora_dir)
        pipe.fuse_lora(lora_scale=1.0)
        print(f"[LoRA] {args.lora_dir}")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        # [2026-10-05 수정] CPU offload 모드에서는 이 시점까지 CUDA가 초기화되지 않아,
        # 아래 reset_peak_memory_stats()가 "RuntimeError: Invalid device argument"로 죽었다(RTX 4090 실측).
        torch.cuda.init()
        pipe.enable_model_cpu_offload(gpu_id=args.gpu)
    else:
        print("[주의] GPU 없음 → CPU로 실행(테스트용)")
        pipe.to("cpu")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
    dev = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")

    log = out / "timing.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fp:
        w = csv.writer(fp)
        if new:
            w.writerow(["name", "tag", "seed", "prompt_from", "steps", "height", "width", "sec", "peak_gb"])
        for f in frames:
            name = f.stem
            prompt, src = caps[name], name
            if args.swap_prompt:
                other = "lap" if name.startswith("inc") else "inc"
                if other not in by_prefix:
                    print(f"[건너뜀] {name}: 바꿔 넣을 {other} 캡션 없음")
                    continue
                prompt, src = by_prefix[other], f"{other}(swap)"
            image = load_image(str(f))
            for seed in args.seeds:
                if use_cuda:
                    torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.time()
                video = pipe(image=image, prompt=prompt, height=args.height, width=args.width, num_frames=93,
                             num_inference_steps=args.steps, guidance_scale=args.guidance,
                             generator=torch.Generator("cpu").manual_seed(seed)).frames[0]
                sec = time.time() - t0
                p = out / f"{name}__{tag}{'__swap' if args.swap_prompt else ''}__s{seed}.mp4"
                export_to_video(video, str(p), fps=16)
                peak = torch.cuda.max_memory_allocated(dev) / 2**30 if use_cuda else 0.0
                w.writerow([name, tag, seed, src, args.steps, args.height, args.width, f"{sec:.1f}", f"{peak:.1f}"])
                fp.flush()
                print(f"[생성] {p.name}  {sec:.0f}초, GPU 피크 {peak:.1f}GB")


if __name__ == "__main__":
    main()
