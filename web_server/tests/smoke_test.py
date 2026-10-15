# -*- coding: utf-8 -*-
"""
smoke_test.py — web_server 워커를 실제 모델(Whisper + Cosmos + LoRA)로 한 번씩 돌려 보는 테스트.

웹 콘솔(app.py)을 같은 프로세스 안에서 띄우고, 녹음 wav 3개를 워커의 process_one()에 직접 넣는다.
녹음 파일은 저장소에 포함하지 않는다. tests/wavs/ 에 아래 이름으로 직접 녹음해 넣는다(16kHz 이상 mono wav).
  smoke_inc  : "절개선 따라 절개해줘"  + test 절개 첫 프레임  → incision 예측 영상
  smoke_lap  : "개복해 줘."           + test 개복 첫 프레임  → laparotomy 예측 영상
  smoke_done : "수술 끝났어"           → (워커 규칙상 '수술' 포함) surgery_complete, 영상 없음

출력은 전부 --out 폴더 안에만 만든다(작업 폴더 사본, 결과 json, nvidia-smi 기록).

사용 (conda activate 후, 빈 GPU 하나를 CUDA_VISIBLE_DEVICES로 지정):
  CUDA_VISIBLE_DEVICES=1 COSMOS_LORA_DIR=$PROJECT_ROOT/lora_checkpoints/20261005_e1_r16 \
  COSMOS_SEED=0 python tests/smoke_test.py \
      --test_frames $PROJECT_ROOT/data/dataset/test/first_frames \
      --out $PROJECT_ROOT/smoke_run_$(date +%m%d_%H%M)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent          # web_server/tests
CODE = HERE.parent                               # web_server

CASES = [
    # (job_id, wav 파일, 첫 프레임 png 이름 또는 None, 기대 동작)
    ("smoke_inc", "inc_절개선따라절개해줘.wav", "inc_p1_t10.png", "incision"),
    ("smoke_lap", "lap_개복해줘.wav", "lap_p1_t10.png", "laparotomy"),
    ("smoke_done", "done_수술끝났어.wav", None, "surgery_complete"),
]


def start_gpu_trace(path: Path):
    """nvidia-smi를 1초 간격으로 기록 (CUDA_VISIBLE_DEVICES의 첫 GPU 기준)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if shutil.which("nvidia-smi") is None:
        print("[주의] nvidia-smi 없음 → GPU 메모리 기록 생략")
        return None
    f = open(path, "w")
    return subprocess.Popen(
        ["nvidia-smi", "-i", vis, "--query-gpu=timestamp,memory.used,utilization.gpu",
         "--format=csv,noheader", "-l", "1"],
        stdout=f, stderr=subprocess.DEVNULL)


def gpu_guard(max_used_mb: int):
    """대상 GPU(CUDA_VISIBLE_DEVICES의 첫 번째)를 다른 프로세스가 쓰고 있으면 모델을 올리기 전에 멈춘다.
    Cosmos 추론 피크가 nvidia-smi 기준 약 21.4GB라서, 빈 GPU에서만 실행한다."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not vis:
        sys.exit("[중단] CUDA_VISIBLE_DEVICES로 GPU 한 장을 지정해야 함")
    if shutil.which("nvidia-smi") is None:
        sys.exit("[중단] nvidia-smi가 없어 GPU 사용량을 확인할 수 없음")
    gpu = vis.split(",")[0]
    q = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    used, total = [int(x) for x in q.stdout.strip().split(",")]
    procs = subprocess.run(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print(f"[GPU 확인] GPU {gpu}: 사용 {used} MiB / {total} MiB, 프로세스: {procs or '없음'}")
    if used > max_used_mb or procs:
        sys.exit(f"[중단] GPU {gpu}를 이미 사용 중 (사용 {used} MiB > 기준 {max_used_mb} MiB 또는 프로세스 있음). "
                 "완전히 빈 GPU에서만 실행할 것.")
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_frames", required=True, help="first_frames/ 폴더 (inc_p1_t10.png, lap_p1_t10.png)")
    ap.add_argument("--out", required=True, help="결과 폴더 (없어야 함)")
    ap.add_argument("--port", type=int, default=5059, help="테스트용 웹 콘솔 포트 (5050과 겹치지 않게)")
    ap.add_argument("--no_gpu_trace", action="store_true")
    ap.add_argument("--max_used_mb", type=int, default=1000,
                    help="대상 GPU에 이미 쓰이고 있는 메모리가 이보다 많으면 실행하지 않음 (기본 1000 MiB)")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    if out.exists():
        sys.exit(f"[중단] 결과 폴더가 이미 있음: {out} (덮어쓰지 않음)")
    gpu_guard(args.max_used_mb)   # 모델을 하나도 올리기 전에 확인

    # 1) 작업 폴더 사본 — 원본 코드 폴더에는 아무것도 쓰지 않는다
    work = out / "workdir"
    shutil.copytree(CODE, work, ignore=shutil.ignore_patterns("tests", "__pycache__", "*.pyc"))
    os.environ["HC122_WORKDIR"] = str(work)
    os.environ["WEB_CONSOLE_URL"] = f"http://127.0.0.1:{args.port}"
    sys.path[:0] = [str(work), str(work / "web_console")]

    trace = None if args.no_gpu_trace else start_gpu_trace(out / "gpu_mem_trace.csv")
    report = {"env": {k: os.environ.get(k) for k in
                      ["CUDA_VISIBLE_DEVICES", "COSMOS_LORA_DIR", "COSMOS_SEED", "COSMOS_OFFLOAD",
                       "WHISPER_DEVICE", "COSMOS_GPU"]},
              "events": [], "cases": []}

    def mark(name):
        report["events"].append({"t": round(time.time() - t_start, 1), "event": name})
        print(f"[{time.time() - t_start:7.1f}s] {name}", flush=True)

    t_start = time.time()
    try:
        # 2) 웹 콘솔을 같은 프로세스에서 띄움 (수술 완료 경로가 /robot/surgery_complete를 부르므로)
        import app as web
        threading.Thread(target=lambda: web.app.run(host="127.0.0.1", port=args.port, use_reloader=False),
                         daemon=True).start()
        time.sleep(1.5)
        mark("web_console_started")

        # 3) 워커 모델 로딩
        import remote_realtime_listener as L
        mark("import_done")
        models = L.load_all_models()
        mark("models_loaded")

        expected = L.COSMOS_EXPECTED_SEC
        for job_id, wav_name, frame_name, want in CASES:
            # 촬영 이미지 대신 test 첫 프레임을 캡처 폴더에 넣는다 (워커가 job_id로 찾음)
            if frame_name:
                shutil.copy(Path(args.test_frames) / frame_name, L.CAPTURES_DIR / f"{job_id}.png")
            wav = L.INCOMING_DIR / f"{job_id}.wav"
            shutil.copy(HERE / "wavs" / wav_name, wav)

            mark(f"{job_id}_start")
            t0 = time.time()
            expected = L.process_one(models, wav, expected)
            sec = round(time.time() - t0, 1)
            mark(f"{job_id}_end")

            st = L.read_job_status(job_id)
            video = L.OUTPUT_DIR / f"{job_id}_result.mp4"
            case = {
                "job_id": job_id, "expected_action": want, "sec": sec,
                "stage": st.get("stage"), "stt_text": st.get("stt_text"),
                "robot_action": st.get("robot_action"), "action_decided_by": st.get("action_decided_by"),
                "prompt_text": st.get("prompt_text"), "condition_frame": st.get("condition_frame"),
                "generation_elapsed_sec": st.get("generation_elapsed_sec"),
                "error_message": st.get("error_message"),
                "video": str(video) if video.exists() else None,
                "pass": st.get("robot_action") == want and st.get("stage") in ("review", "robot_direct"),
            }
            report["cases"].append(case)
            print(json.dumps(case, ensure_ascii=False, indent=1), flush=True)

        # 4) 웹 콘솔 쪽에서 본 상태 (승인 → Jetson 명령 생성까지)
        c = web.app.test_client()
        r = c.post("/decision/smoke_inc", json={"action": "approve"}).get_json()
        report["approve_smoke_inc"] = {"response": r, "robot_command": c.get("/robot/command").get_json()}
        mark("approve_checked")
    except Exception as e:  # 실패해도 그때까지의 기록은 남긴다
        import traceback
        report["exception"] = traceback.format_exc()
        print(report["exception"], flush=True)
    finally:
        if trace:
            trace.terminate()
        (out / "smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n결과: {out / 'smoke_report.json'}")
        print("PASS" if report.get("cases") and all(c["pass"] for c in report["cases"]) else "FAIL 또는 미완료")


if __name__ == "__main__":
    main()
