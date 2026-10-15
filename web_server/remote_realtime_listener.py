# -*- coding: utf-8 -*-
"""
remote_realtime_listener.py — 음성 명령 → 예측 영상 생성 워커 (GPU 서버에서 상시 실행)

web_console/app.py 와는 파일로만 주고받는다.
  inputs/incoming/{job_id}.wav          웹 콘솔이 넣는 녹음 파일 (이 워커가 감시)
  outputs/{job_id}.json                 job 상태 (웹 콘솔과 병합 기록)
  outputs/{job_id}_result.mp4           생성된 예측 영상
  capture_bridge/captures/{job_id}.png  Gemini E 촬영 이미지 (Cosmos 입력 프레임)

처리 흐름
  1) Whisper STT (한국어)
  2) 명령 분류
       - 키워드로 먼저 판단 (지연 0, 결과가 항상 같음)
       - 키워드가 안 맞을 때만 Groq LLM이 STT 오인식을 감안해 절개/개복 중 하나로 보정
         (예: "개복해줘"가 "복구해줘"로 인식된 경우)
       - "수술 완료"는 승인 화면 없이 바로 로봇에 전달되므로 키워드로만 인정한다
  3) 수술 완료 → /robot/surgery_complete 로 즉시 전달
  4) 절개/개복 → Cosmos-Predict2.5-2B + LoRA 로 Image2World 예측 영상 생성
       - 프롬프트는 LoRA 학습 캡션(captions.json)을 글자 그대로 사용
       - assets/prompt_embeds/ 에 미리 계산한 텍스트 임베딩이 있으면 텍스트 인코더(약 16GB)를 아예 로드하지 않는다
         (연구실 4090 실측: 결과 픽셀 동일, GPU 피크 21.9GB → 9.5GB, 첫 생성 384초 → 195초. tools/make_embeds.py)
       - 입력 프레임은 Gemini E 촬영 이미지 (없으면 assets/reference_frame.png)
       - 432×768, 93프레임, 16fps (2차 LoRA E1 학습 해상도와 동일, 환경변수로 변경 가능)
  5) faststart 처리 후 stage="review" → 웹 콘솔에서 집도의 승인

실행
  conda activate cosmos_surgical
  export COSMOS_LORA_DIR=/path/to/lora_checkpoints/production   # 없으면 base 모델로 생성
  export GROQ_API_KEY=...                                         # 없으면 LLM 보조 분류 생략
  python remote_realtime_listener.py

준비물: ffmpeg 명령(PATH), HuggingFace 로그인(게이트 모델), Whisper 모델(첫 실행 때 자동 다운로드 약 1.5GB)

*** 실행되는 동안 GPU를 계속 점유한다. 팀원들과 GPU 사용 스케줄을 미리 맞출 것 ***
"""

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import torch
import whisper
from PIL import Image
from diffusers import Cosmos2_5_PredictBasePipeline
from diffusers.utils import export_to_video, load_image

# ---------------------------------------------------------------------------
# 경로
# 작업 폴더 = 이 파일이 있는 폴더. 다른 위치를 쓰려면 HC122_WORKDIR 로 지정
# (web_console/app.py 도 같은 값을 써야 함)
# ---------------------------------------------------------------------------
WORKDIR = Path(os.environ.get("HC122_WORKDIR", Path(__file__).resolve().parent))
INCOMING_DIR = WORKDIR / "inputs" / "incoming"
PROCESSED_DIR = WORKDIR / "inputs" / "processed"
OUTPUT_DIR = WORKDIR / "outputs"
CAPTURES_DIR = WORKDIR / "capture_bridge" / "captures"
CAPTIONS_PATH = WORKDIR / "captions.json"
# 촬영 이미지가 없을 때 쓰는 기준 프레임: 2차 학습 test 클립(inc_p1_t10)의 첫 프레임, 960×528
FALLBACK_FRAME = WORKDIR / "assets" / "reference_frame.png"
# 미리 계산한 텍스트 임베딩 (tools/make_embeds.py 출력). 없으면 텍스트 인코더를 로드하는 기존 경로
EMBEDS_DIR = Path(os.environ.get("COSMOS_EMBEDS_DIR", WORKDIR / "assets" / "prompt_embeds"))

# ---------------------------------------------------------------------------
# 설정 (환경변수로 덮어쓰기 가능)
# ---------------------------------------------------------------------------
WEB_CONSOLE_URL = os.environ.get("WEB_CONSOLE_URL", "http://127.0.0.1:5050")

# GPU 배치: GPU가 2장이면 Cosmos는 COSMOS_GPU, Whisper는 다른 GPU에 둔다.
# (RTX 4090 실측: Cosmos 432×768 추론이 nvidia-smi 기준 약 21.5GB라 같은 GPU에 Whisper를 같이 올리면 위험)
# GPU가 1장이면 Whisper는 CPU에서 돈다. WHISPER_DEVICE 로 직접 지정 가능 (예: cuda:1, cpu)
COSMOS_GPU = int(os.environ.get("COSMOS_GPU", "0"))
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "")

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "medium")
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))
# Whisper 모델 저장 위치. 비우면 whisper 기본값(~/.cache/whisper). 공유 서버에서는 프로젝트 경로로 지정 권장
WHISPER_DOWNLOAD_ROOT = os.environ.get("WHISPER_DOWNLOAD_ROOT") or None

GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_TIMEOUT_SEC = float(os.environ.get("GROQ_TIMEOUT_SEC", "5"))

COSMOS_MODEL_ID = os.environ.get("COSMOS_MODEL_ID", "nvidia/Cosmos-Predict2.5-2B")
COSMOS_REVISION = os.environ.get("COSMOS_REVISION", "diffusers/base/post-trained")
COSMOS_LORA_DIR = os.environ.get("COSMOS_LORA_DIR", "")
# auto: GPU 메모리 40GB 미만(예: RTX 4090 24GB)이면 CPU offload, 이상(A6000 48GB)이면 전부 GPU 상주
COSMOS_OFFLOAD = os.environ.get("COSMOS_OFFLOAD", "auto")  # auto | on | off
COSMOS_STEPS = int(os.environ.get("COSMOS_STEPS", "30"))
COSMOS_GUIDANCE = float(os.environ.get("COSMOS_GUIDANCE", "7.0"))
COSMOS_SEED = int(os.environ.get("COSMOS_SEED", "42"))
# 웹 화면의 "남은 시간" 표시용 첫 예상값(초). 이후에는 실제 측정값으로 갱신된다.
# 기본값은 2차 실측(RTX 4090, 432×768·93f·30step, CPU offload) 영상당 194~226초 기준.
COSMOS_EXPECTED_SEC = float(os.environ.get("COSMOS_EXPECTED_SEC", "210"))

# LoRA 학습 해상도와 반드시 같아야 하는 값.
# 2차 E1(20261005_e1_r16 / e1_r32)은 432×768로 학습했다(run_config.txt의 res=432x768).
HEIGHT = int(os.environ.get("COSMOS_HEIGHT", "432"))
WIDTH = int(os.environ.get("COSMOS_WIDTH", "768"))
NUM_FRAMES = 93
FPS = 16
if HEIGHT % 16 or WIDTH % 16:
    raise ValueError("COSMOS_HEIGHT/WIDTH는 16의 배수여야 합니다.")

# 촬영 이미지가 job 상태에 기록될 때까지 기다리는 최대 시간(초)
CAPTURE_WAIT_SEC = float(os.environ.get("CAPTURE_WAIT_SEC", "5"))

POLL_INTERVAL_SEC = 2

GENERATIVE_ACTIONS = ("incision", "laparotomy")


# ---------------------------------------------------------------------------
# job 상태 파일
# ---------------------------------------------------------------------------
def status_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}.json"


def read_job_status(job_id: str) -> dict:
    path = status_path(job_id)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def write_job_status(job_id: str, **fields):
    """web_console과 공유하는 job 상태 파일에 진행 상황을 병합 기록한다."""
    data = read_job_status(job_id)
    data.update(fields)
    data["job_id"] = job_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = status_path(job_id).with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, status_path(job_id))


def move_to_processed(audio_path: Path):
    """처리 끝난 wav를 processed/로 옮겨 재처리를 막는다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    target = PROCESSED_DIR / audio_path.name
    if target.exists():
        target.unlink()
    shutil.move(str(audio_path), str(target))


# ---------------------------------------------------------------------------
# 1) 명령 분류
# ---------------------------------------------------------------------------
def classify_by_keyword(text: str):
    """키워드로 동작을 판단한다. 못 찾으면 None."""
    t = text.replace(" ", "")
    if "절개" in t:
        return "incision"
    if "개복" in t:
        return "laparotomy"
    # 수술 완료는 승인 없이 바로 실행되므로 "수술"과 함께 말한 경우만 인정
    if "수술" in t and ("완료" in t or "끝" in t):
        return "surgery_complete"
    return None


LLM_SYSTEM_PROMPT = """You classify Korean voice commands for a surgical assistant robot.
The text comes from speech recognition and may contain recognition errors
(for example "개복" misheard as "복구" or "예복", "절개" misheard as "절게").

Answer with exactly one word:
- incision   : the surgeon asks the robot to make an incision (절개)
- laparotomy : the surgeon asks the robot to open the incision (개복)
- unknown    : anything else, or if you are not sure

Output only that one word."""


def classify_by_llm(groq_client, text: str):
    """키워드가 실패했을 때만 호출. incision / laparotomy / None 중 하나를 돌려준다.
    surgery_complete 는 여기서 절대 만들지 않는다(승인 없이 실행되는 명령이라서)."""
    if groq_client is None:
        return None
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
        )
        answer = (response.choices[0].message.content or "").strip().lower()
    except Exception as e:
        print(f"[LLM] 분류 실패, 키워드 결과만 사용: {e}")
        return None

    match = re.search(r"\b(incision|laparotomy|unknown)\b", answer)
    label = match.group(1) if match else "unknown"
    print(f"[LLM] '{text}' -> {label}")
    return label if label in GENERATIVE_ACTIONS else None


def classify_command(groq_client, text: str):
    """(동작, 판단 근거) 를 돌려준다. 동작을 못 찾으면 (None, None)."""
    action = classify_by_keyword(text)
    if action:
        return action, "keyword"
    action = classify_by_llm(groq_client, text)
    if action:
        return action, "llm"
    return None, None


# ---------------------------------------------------------------------------
# 2) Cosmos 입력 프레임
# ---------------------------------------------------------------------------
# 학습 클립(make_clips.py 출력)과 평가 첫 프레임의 크기
CLIP_W, CLIP_H = 960, 528


def to_model_frame(img: Image.Image) -> Image.Image:
    """학습 때와 같은 전처리로 Cosmos 입력 프레임을 만든다.

    1) 학습 클립(make_clips.py)과 같은 방식: 가운데 기준 960:528 크롭 → 960×528 축소
    2) 학습 스크립트(VideoDataset, PIL 입력)와 같은 방식: PIL bilinear로 WIDTH×HEIGHT(768×432)

    2)를 여기서 하는 이유:
    파이프라인에 960×528을 그대로 넘기면 내부에서 텐서로 바꾼 뒤
    F.interpolate(기본 nearest)로 줄여서 학습 때(bilinear)와 첫 프레임이 달라진다
    (평균 차이 약 2.7/255). 미리 bilinear로 맞춰 넘기면 학습 경로와 차이 0.
    학습과 추론의 입력 전처리를 이 함수 하나로 맞춘다."""
    img = img.convert("RGB")
    target = CLIP_W / CLIP_H
    w, h = img.size
    if w / h > target:  # 너무 넓음 → 좌우 자르기
        nw = round(h * target)
        left = (w - nw) // 2
        img = img.crop((left, 0, left + nw, h))
    elif w / h < target:  # 너무 높음 → 위아래 자르기
        nh = round(w / target)
        top = round((h - nh) * 0.5)
        img = img.crop((0, top, w, top + nh))
    img = img.resize((CLIP_W, CLIP_H), Image.BOX)
    return img.resize((WIDTH, HEIGHT), Image.BILINEAR)


def find_condition_frame(job_id: str) -> Path:
    """Gemini E 촬영 이미지를 찾는다. 잠깐 기다려도 없으면 기준 프레임을 쓴다."""
    deadline = time.time() + CAPTURE_WAIT_SEC
    while True:
        source = read_job_status(job_id).get("source_image")
        if source and Path(source).is_file():
            return Path(source)
        candidate = CAPTURES_DIR / f"{job_id}.png"
        if candidate.is_file():
            return candidate
        if time.time() >= deadline:
            break
        time.sleep(0.5)
    print(f"[입력] 촬영 이미지 없음 → 기준 프레임 사용: {FALLBACK_FRAME.name}")
    return FALLBACK_FRAME


# ---------------------------------------------------------------------------
# 3) 영상 후처리
# ---------------------------------------------------------------------------
def apply_faststart(video_path: Path):
    """MP4의 moov atom을 앞으로 옮겨 iPad Safari에서 바로 재생되게 한다(재인코딩 없음)."""
    temp_path = video_path.with_name(video_path.stem + "_faststart_tmp.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-c", "copy",
             "-movflags", "+faststart", str(temp_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
        )
        os.replace(temp_path, video_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


# ---------------------------------------------------------------------------
# 모델 로딩 (한 번만 로드해서 계속 상주)
# ---------------------------------------------------------------------------
class _NoSafetyChecker:
    """포크의 LoRA 평가 스크립트(eval_cosmos_predict25_lora.py)와 같은 대체 객체.
    수술 도구·인공 피부 장면이 안전 필터에 걸려 생성이 막히는 것을 방지."""

    def to(self, *args, **kwargs):
        return self

    def check_text_safety(self, *args, **kwargs):
        return True

    def check_video_safety(self, video):
        return video


def load_captions() -> dict:
    with open(CAPTIONS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = data["captions"]
    missing = [a for a in GENERATIVE_ACTIONS if not captions.get(a)]
    if missing:
        raise ValueError(f"captions.json 에 캡션이 없음: {missing}")
    if data.get("status") != "final":
        print("[주의] captions.json 의 status가 final이 아님. "
              "LoRA 학습에 쓴 캡션과 글자까지 같은지 확인할 것.")
    return captions


_PROMPT_EMBEDS = {}   # {"incision": tensor, "laparotomy": tensor, "negative": tensor} — CPU에 보관


def _sha1_10(text: str) -> str:
    import hashlib
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def load_prompt_embeds(captions: dict) -> bool:
    """미리 계산한 임베딩을 읽는다. 쓸 수 있으면 True.
    captions.json 문장과 임베딩을 만들 때의 문장이 다르면(sha1 비교) 쓰지 않는다."""
    if os.environ.get("COSMOS_USE_EMBEDS", "auto") == "off":
        print("[임베딩] COSMOS_USE_EMBEDS=off → 텍스트 인코더 경로")
        return False
    if COSMOS_GUIDANCE <= 1.0:
        print("[임베딩] guidance ≤ 1.0이면 negative 임베딩 조건이 달라져 사용 안 함")
        return False
    report_path = EMBEDS_DIR / "make_embeds_report.json"
    files = {a: EMBEDS_DIR / f"prompt_embeds_{a}.pt" for a in GENERATIVE_ACTIONS}
    files["negative"] = EMBEDS_DIR / "negative_prompt_embeds_default.pt"
    if not report_path.is_file() or not all(f.is_file() for f in files.values()):
        print(f"[임베딩] {EMBEDS_DIR} 에 임베딩 없음 → 텍스트 인코더 경로 (tools/make_embeds.py로 만들 수 있음)")
        return False
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for action in GENERATIVE_ACTIONS:
        made_from = report.get("items", {}).get(action, {}).get("caption_sha1_10")
        if made_from != _sha1_10(captions[action]):
            print(f"[임베딩] {action} 캡션이 임베딩을 만든 문장과 다름 → 텍스트 인코더 경로. "
                  "캡션을 바꿨다면 tools/make_embeds.py를 다시 실행할 것")
            return False
    for key, path in files.items():
        _PROMPT_EMBEDS[key] = torch.load(path, map_location="cpu")
    print(f"[임베딩] 사용: {EMBEDS_DIR} (텍스트 인코더 로드 안 함)")
    return True


def load_cosmos_pipeline(use_embeds: bool = False):
    print(f"[초기화] Cosmos 로딩: {COSMOS_MODEL_ID} ({COSMOS_REVISION})")
    extra = {"text_encoder": None} if use_embeds else {}
    pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
        COSMOS_MODEL_ID,
        revision=COSMOS_REVISION,
        torch_dtype=torch.bfloat16,
        safety_checker=_NoSafetyChecker(),
        **extra,
    )

    if COSMOS_LORA_DIR:
        pipe.load_lora_weights(COSMOS_LORA_DIR)
        pipe.fuse_lora(lora_scale=1.0)
        print(f"[초기화] LoRA 적용: {COSMOS_LORA_DIR}")
    else:
        print("[주의] COSMOS_LORA_DIR 미지정 → base 모델로 생성")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU가 필요합니다.")
    # CPU offload 모드에서는 이 시점까지 CUDA가 초기화되지 않을 수 있다.
    # (연구실 4090 서버 실측: 초기화 전 CUDA 메모리 함수 호출 시 RuntimeError: Invalid device argument)
    torch.cuda.init()

    total_gb = torch.cuda.get_device_properties(COSMOS_GPU).total_memory / 2**30
    # 텍스트 인코더가 없으면 피크가 약 9.5GB라 24GB GPU에도 그대로 올릴 수 있지만,
    # 연구실에서 검증한 경로가 offload뿐이라 auto 기준은 그대로 둔다.
    use_offload = COSMOS_OFFLOAD == "on" or (COSMOS_OFFLOAD == "auto" and total_gb < 40)

    if use_offload:
        # 텍스트 인코더(약 16~17GB)까지 한 번에 못 올리는 GPU: 필요한 부품만 차례로 GPU에 올림
        pipe.enable_model_cpu_offload(gpu_id=COSMOS_GPU)
        print(f"[초기화] GPU {total_gb:.0f}GB → CPU offload 모드")
    else:
        # 전부 GPU 상주 (1차 DIS04 A6000에서 쓰던 방식)
        device = torch.device(f"cuda:{COSMOS_GPU}")
        Cosmos2_5_PredictBasePipeline._execution_device = property(lambda self: device)
        pipe.transformer.to(device)
        pipe.vae.to(device)
        if pipe.text_encoder is not None:
            pipe.text_encoder.to(device)
        print(f"[초기화] GPU {total_gb:.0f}GB → 전체 GPU 상주 모드")

    return pipe


_transcribe_fp16 = [True]
_generation_count = [0]


def check_prerequisites():
    """실행 전에 빠지기 쉬운 준비물을 확인한다 (연구실 서버에서 실제로 빠져 있던 것)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg 명령이 PATH에 없습니다. Whisper가 녹음을 읽을 때와 mp4 faststart에 필요합니다. "
            "예: conda install -c conda-forge ffmpeg")


def choose_whisper_device(use_embeds: bool) -> str:
    """Whisper를 올릴 장치.
    - GPU 2장: Cosmos와 다른 GPU
    - GPU 1장 + 임베딩 사용: Cosmos와 같은 GPU (실측 Cosmos 9.5GB + Whisper 4.9GB ≈ 14.4GB)
    - GPU 1장 + 텍스트 인코더 경로: CPU (21.9GB + 4.9GB > 24GB라 같은 GPU 불가)"""
    if WHISPER_DEVICE:
        return WHISPER_DEVICE
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpu >= 2 and COSMOS_GPU in (0, 1):
        return f"cuda:{1 - COSMOS_GPU}"
    if n_gpu >= 1 and use_embeds:
        return f"cuda:{COSMOS_GPU}"
    return "cpu"


def load_all_models():
    check_prerequisites()
    captions = load_captions()
    use_embeds = load_prompt_embeds(captions)

    whisper_device = choose_whisper_device(use_embeds)
    print(f"[초기화] Whisper {WHISPER_MODEL} 로딩 ({whisper_device})...")
    whisper_model = whisper.load_model(WHISPER_MODEL, device=whisper_device,
                                       download_root=WHISPER_DOWNLOAD_ROOT)
    # CPU에서는 fp16을 쓸 수 없다
    _transcribe_fp16[0] = whisper_device != "cpu"

    groq_client = None
    if os.environ.get("GROQ_API_KEY"):
        from groq import Groq
        groq_client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=GROQ_TIMEOUT_SEC)
        print(f"[초기화] Groq 보조 분류 사용: {GROQ_MODEL}")
    else:
        print("[초기화] GROQ_API_KEY 없음 → 키워드 분류만 사용")

    pipe = load_cosmos_pipeline(use_embeds)
    print("[초기화] 완료. 새 음성 파일을 기다립니다.\n")
    return whisper_model, groq_client, captions, pipe


# ---------------------------------------------------------------------------
# job 하나 처리
# ---------------------------------------------------------------------------
def send_surgery_complete(job_id: str, stt_text: str):
    """수술 완료: 예측 영상·승인 없이 Jetson 실행 큐로 바로 전달."""
    payload = json.dumps({"source_job_id": job_id}).encode("utf-8")
    req = urllib.request.Request(
        f"{WEB_CONSOLE_URL}/robot/surgery_complete",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    # 상태(stage=robot_direct 등)는 웹 콘솔이 기록하므로 여기서는 STT 결과만 보탠다.
    write_job_status(job_id, stt_text=stt_text, prompt_text="수술 완료")
    print(f"[ROBOT] surgery_complete 전달 완료: {result.get('job_id')}")


def generate_preview(pipe, captions: dict, job_id: str, robot_action: str,
                     stt_text: str, expected_sec: float) -> float:
    """Cosmos로 예측 영상을 만들고 review 단계로 넘긴다. 실제 걸린 시간(초)을 돌려준다."""
    prompt = captions[robot_action]
    frame_path = find_condition_frame(job_id)
    image = to_model_frame(load_image(str(frame_path)))

    started = time.time()
    write_job_status(
        job_id,
        stage="generating",
        stt_text=stt_text,
        prompt_text=prompt,
        robot_action=robot_action,
        condition_frame=str(frame_path),
        generation_started_at=started,
        generation_duration_sec=round(expected_sec),
    )
    print(f"[생성] {robot_action} / 입력 {frame_path.name} / 예상 {expected_sec:.0f}초")

    if _PROMPT_EMBEDS:
        # 미리 계산한 임베딩 경로: prompt는 None이어야 하고(같이 주면 check_inputs 에러),
        # 파이프라인이 device·dtype을 바꿔 주지 않으므로 직접 맞춘다.
        device = pipe._execution_device
        text_args = {
            "prompt": None,
            "prompt_embeds": _PROMPT_EMBEDS[robot_action].to(device=device, dtype=torch.bfloat16),
            "negative_prompt_embeds": _PROMPT_EMBEDS["negative"].to(device=device, dtype=torch.bfloat16),
        }
    else:
        text_args = {"prompt": prompt}   # negative_prompt 미지정 → 파이프라인 기본 negative (임베딩과 같은 조건)

    video = pipe(
        image=image,
        **text_args,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        num_inference_steps=COSMOS_STEPS,
        guidance_scale=COSMOS_GUIDANCE,
        generator=torch.Generator("cpu").manual_seed(COSMOS_SEED),
    ).frames[0]

    out_path = OUTPUT_DIR / f"{job_id}_result.mp4"
    export_to_video(video, str(out_path), fps=FPS)
    apply_faststart(out_path)  # review 전환보다 반드시 먼저
    elapsed = time.time() - started

    write_job_status(
        job_id,
        stage="review",
        generation_completed_at=time.time(),
        generation_elapsed_sec=round(elapsed, 1),
    )
    print(f"[완료] {out_path.name} ({elapsed:.0f}초)")
    return elapsed


def process_one(models, audio_path: Path, expected_sec: float) -> float:
    """wav 하나를 처리한다. 다음 '남은 시간' 표시에 쓸 예상 생성 시간을 돌려준다."""
    whisper_model, groq_client, captions, pipe = models
    job_id = audio_path.stem
    print(f"\n{'=' * 50}\n새 명령: {audio_path.name}\n{'=' * 50}")

    result = whisper_model.transcribe(
        str(audio_path),
        language="ko",
        fp16=_transcribe_fp16[0],
        beam_size=WHISPER_BEAM_SIZE,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=True,
    )
    stt_text = result["text"].strip()
    print(f"[STT] {stt_text}")

    robot_action, decided_by = classify_command(groq_client, stt_text)
    if robot_action is None:
        write_job_status(
            job_id,
            stage="error",
            stt_text=stt_text,
            error_message="절개/개복/수술 완료 명령을 인식하지 못했습니다.",
        )
        move_to_processed(audio_path)
        return expected_sec

    print(f"[분류] {robot_action} ({decided_by})")
    write_job_status(
        job_id,
        stage="stt_done",
        stt_text=stt_text,
        robot_action=robot_action,
        action_decided_by=decided_by,
    )

    if robot_action == "surgery_complete":
        try:
            send_surgery_complete(job_id, stt_text)
        except Exception as e:
            print(f"[ROBOT ERROR] surgery_complete 전달 실패: {e}")
            write_job_status(job_id, stage="error",
                             error_message=f"수술 완료 명령 전달 실패: {e}")
        move_to_processed(audio_path)
        return expected_sec

    try:
        elapsed = generate_preview(pipe, captions, job_id, robot_action, stt_text, expected_sec)
        # 다음 job의 남은 시간 표시는 직전 실측과 기존 예상의 평균으로 보정.
        # 로드 직후 첫 생성은 모델을 GPU로 옮기는 시간이 섞여 길게 나오므로(실측 384초 vs 이후 212초) 반영하지 않는다.
        if _generation_count[0] > 0:
            expected_sec = (expected_sec + elapsed) / 2
        _generation_count[0] += 1
    except Exception as e:
        print(f"[생성 실패] {e}")
        write_job_status(job_id, stage="error", error_message=f"예측 영상 생성 실패: {e}")
    move_to_processed(audio_path)
    return expected_sec


# ---------------------------------------------------------------------------
# 감시 루프
# ---------------------------------------------------------------------------
def watch_loop():
    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    models = load_all_models()
    expected_sec = COSMOS_EXPECTED_SEC

    # 워커 시작 전부터 있던 wav는 처리하지 않는다
    seen = {p.name for p in INCOMING_DIR.glob("*.wav")}
    print(f"[초기화] 기존 WAV {len(seen)}개는 무시합니다.")
    print("감시 시작:", INCOMING_DIR)

    try:
        while True:
            for wav in sorted(INCOMING_DIR.glob("*.wav")):
                if wav.name in seen:
                    continue
                # 업로드 중인 파일일 수 있으니 크기가 그대로인지 확인
                size1 = wav.stat().st_size
                time.sleep(0.5)
                if not wav.exists() or wav.stat().st_size != size1:
                    continue
                seen.add(wav.name)
                try:
                    expected_sec = process_one(models, wav, expected_sec)
                except Exception as e:
                    print(f"[처리 실패] {wav.name}: {e}")
            time.sleep(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n감시 종료.")


if __name__ == "__main__":
    watch_loop()
