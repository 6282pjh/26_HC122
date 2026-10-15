# -*- coding: utf-8 -*-
"""make_embeds.py — 캡션 2개 + DEFAULT_NEGATIVE_PROMPT의 텍스트 임베딩을 미리 계산해 저장한다.

파이프라인 __call__(pipeline_cosmos2_5_predict.py 686~697행)이 부르는 것과 같은 함수·같은 인자를 쓴다:
    self.encode_prompt(prompt=prompt, negative_prompt=negative_prompt,          # negative_prompt=None
                       do_classifier_free_guidance=self.do_classifier_free_guidance,  # guidance 7.0 → True
                       num_videos_per_prompt=num_videos_per_prompt,             # 1
                       prompt_embeds=None, negative_prompt_embeds=None,
                       device=device,                                           # self._execution_device
                       max_sequence_length=max_sequence_length)                 # 512
negative_prompt가 None이면 encode_prompt 381행이 DEFAULT_NEGATIVE_PROMPT(63~70행)를 쓴다.

사용: CUDA_VISIBLE_DEVICES=<빈 GPU> python tools/make_embeds.py [--cache_dir <학습 캐시>]
      기본값: --captions captions.json, --out assets/prompt_embeds  (web_server 폴더 기준)
필요 GPU: 텍스트 인코더를 올리므로 nvidia-smi 기준 약 17.5GB (RTX 4090 실측 피크 17,234MiB)
RTX 4090 24GB에서 2026-10-11 검증.
      학습 캐시 임베딩과 비트 단위 동일, 이 임베딩으로 만든 영상이 텍스트 인코더 경로와 픽셀 동일함을 확인.
      바꾼 것은 기본 경로 2개뿐이다.
"""
import argparse, hashlib, json, time
from pathlib import Path
import torch
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import DEFAULT_NEGATIVE_PROMPT


class NoSafetyChecker:
    def to(self, *a, **k): return self
    def check_text_safety(self, *a, **k): return True
    def check_video_safety(self, video): return video


ap = argparse.ArgumentParser()
ROOT = Path(__file__).resolve().parent.parent   # web_server
ap.add_argument("--captions", default=str(ROOT / "captions.json"))
ap.add_argument("--out", default=str(ROOT / "assets" / "prompt_embeds"))
ap.add_argument("--cache_dir", default=None, help="학습 캐시(index.json, prompt_embeds_*.pt) — 값 비교용")
ap.add_argument("--model_id", default="nvidia/Cosmos-Predict2.5-2B")
ap.add_argument("--revision", default="diffusers/base/post-trained")
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
caps = json.load(open(args.captions, encoding="utf-8"))["captions"]

t0 = time.time()
pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(args.model_id, revision=args.revision,
                                                     torch_dtype=torch.bfloat16, safety_checker=NoSafetyChecker())
torch.cuda.init()
pipe.enable_model_cpu_offload(gpu_id=0)
print(f"[로드] {time.time() - t0:.0f}초")
device = pipe._execution_device
report = {"device": str(device), "function": "pipe.encode_prompt", "max_sequence_length": 512, "items": {}}
neg_ref = None
with torch.no_grad():
    for action in ("incision", "laparotomy"):
        pe, ne = pipe.encode_prompt(prompt=caps[action], negative_prompt=None, do_classifier_free_guidance=True,
                                    num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None,
                                    device=device, max_sequence_length=512)
        pe, ne = pe.cpu(), ne.cpu()
        torch.save(pe, out / f"prompt_embeds_{action}.pt")
        if neg_ref is None:
            neg_ref = ne; torch.save(ne, out / "negative_prompt_embeds_default.pt")
        item = {"shape": list(pe.shape), "dtype": str(pe.dtype),
                "caption_sha1_10": hashlib.sha1(caps[action].encode("utf-8")).hexdigest()[:10],
                "negative_same_as_first_call_max_abs_diff": float((ne.float() - neg_ref.float()).abs().max())}
        if args.cache_dir:
            idx = json.load(open(Path(args.cache_dir) / "index.json", encoding="utf-8"))
            key = item["caption_sha1_10"]
            if key in idx["prompt_embeds"]:
                c = torch.load(Path(args.cache_dir) / idx["prompt_embeds"][key], map_location="cpu")
                d = (pe.float() - c.float()).abs()
                item["vs_train_cache"] = {"file": idx["prompt_embeds"][key], "cache_shape": list(c.shape), "cache_dtype": str(c.dtype),
                                          "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
                                          "bit_identical": bool(torch.equal(pe, c)),
                                          "embed_abs_mean": float(pe.float().abs().mean())}
            else:
                item["vs_train_cache"] = "캐시에 같은 캡션 키 없음"
        report["items"][action] = item
        print(action, json.dumps(item, ensure_ascii=False))
report["negative"] = {"shape": list(neg_ref.shape), "dtype": str(neg_ref.dtype),
                      "prompt_sha1_10": hashlib.sha1(DEFAULT_NEGATIVE_PROMPT.encode("utf-8")).hexdigest()[:10]}
report["sec_total"] = round(time.time() - t0, 1)
report["torch_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
(out / "make_embeds_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(report["negative"]), "peak_alloc_gb", report["torch_peak_alloc_gb"], "sec", report["sec_total"])
