# -*- coding: utf-8 -*-
"""
web_console/app.py — 낭만닥터 수술 어시스턴트 승인 콘솔 (Flask 버전)

remote_realtime_listener.py(워커)와는 아래 파일로만 주고받는다.
(Whisper / Groq / Cosmos 호출은 워커가 담당한다.)

  inputs/incoming/{job_id}.wav   -> worker가 감지해서 처리
  outputs/{job_id}_result.mp4    -> worker가 생성 완료 시 저장
  outputs/{job_id}.json          -> worker가 단계별 진행 상황을 기록.
                                     웹 콘솔은 이 파일로 단계·STT 결과·프롬프트를
                                     실시간 표시하고, 파일이 아직 없으면
                                     mp4 존재 여부로 완료를 판단한다.

실행:
    cd web_console
    python app.py
    (기본 포트 5050 — Gradio 7860과 겹치지 않게 분리)
"""
import json
import os
import subprocess
import tempfile
import uuid
import wave
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, render_template, send_file, abort
from flask_capture_routes import register_capture_routes

# ---------------------------------------------------------------
# 경로 설정 — 기존 worker와 동일한 기준
# 작업 폴더 = web_console/ 의 상위 폴더 (폴더 이름·위치가 바뀌어도 그대로 동작).
# 다른 위치를 쓰려면 환경변수 HC122_WORKDIR 로 지정 (listener와 같은 값이어야 함).
# ---------------------------------------------------------------
BASE_DIR = Path(os.environ.get("HC122_WORKDIR", Path(__file__).resolve().parent.parent))
INCOMING_DIR = BASE_DIR / "inputs" / "incoming"
OUTPUT_DIR = BASE_DIR / "outputs"

# Jetson 연동용 로봇 명령 파일
ROBOT_BRIDGE_DIR = BASE_DIR / "robot_bridge"
ROBOT_COMMAND_PATH = ROBOT_BRIDGE_DIR / "command.json"

INCOMING_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ROBOT_BRIDGE_DIR.mkdir(parents=True, exist_ok=True)

def read_robot_command():
    if not ROBOT_COMMAND_PATH.exists():
        return {
            "status": "idle",
            "consumed": True
        }

    try:
        with open(ROBOT_COMMAND_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "status": "error",
            "consumed": True
        }


def write_robot_command(job_id: str, action: str):
    """승인할 때마다 항상 새로운 고유 job_id로 명령을 생성한다.
    같은 job/action을 재승인해도 재사용하지 않는다 (Jetson 측 요구사항)."""
    unique_job_id = f"{job_id}_{uuid.uuid4().hex[:8]}"
    data = {
        "job_id": unique_job_id,
        "source_job_id": job_id,
        "action": action,
        "status": "approved",
        "consumed": False,
        "created_at": datetime.now().isoformat(),
    }
    tmp = ROBOT_COMMAND_PATH.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, ROBOT_COMMAND_PATH)
    print(f"[ROBOT] 새 명령 생성: {unique_job_id} (원본={job_id}) / {action}")
    return data


app = Flask(__name__)

def on_capture_uploaded(job_id, saved_path):
    """
    Gemini E에서 촬영한 현재 환경 이미지를 해당 job에 연결한다.
    여기서는 영상 생성/review 전환을 하지 않는다.
    STT 완료 후 별도의 generating 단계에서 처리한다.
    """
    print(f"[capture] 이미지 업로드됨: job_id={job_id}, path={saved_path}")

    write_status(
        job_id,
        source_image=str(saved_path),
        camera_status="captured",
        camera_captured_at=datetime.now().isoformat(),
    )


register_capture_routes(app, on_capture_uploaded=on_capture_uploaded)

# 최소 녹음 길이(초) — 이보다 짧으면 요청 자체를 막는다 (2단계에서 확인한 무음 문제 대비)
MIN_DURATION_SEC = 1.0


# ---------------------------------------------------------------
# 상태 파일 read/write 헬퍼
# ---------------------------------------------------------------
def status_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}.json"


def result_path(job_id: str) -> Path:
    return OUTPUT_DIR / f"{job_id}_result.mp4"


def read_status(job_id: str) -> dict:
    """worker가 남긴 상태 파일을 읽는다. 없으면 파일 존재 여부로 최소 추정한다."""
    sp = status_path(job_id)
    if sp.exists():
        try:
            with open(sp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    # 상태 파일이 아직 생기지 않은 초기 구간의 폴백
    if result_path(job_id).exists():
        return {"job_id": job_id, "stage": "review", "stt_text": "", "prompt_text": ""}
    if (INCOMING_DIR / f"{job_id}.wav").exists():
        return {"job_id": job_id, "stage": "queued", "stt_text": "", "prompt_text": ""}
    return {"job_id": job_id, "stage": "unknown", "stt_text": "", "prompt_text": ""}


def write_status(job_id: str, **fields):
    """decision(승인/거부/긴급정지) 등을 상태 파일에 병합 저장한다."""
    sp = status_path(job_id)
    data = read_status(job_id)
    if data.get("stage") == "unknown":
        data = {}
    data.update(fields)
    data["job_id"] = job_id
    data["updated_at"] = datetime.now().isoformat()
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# 화면 1 · 명령
# ---------------------------------------------------------------
@app.route("/")
def screen_ready():
    return render_template("ready.html")


@app.route("/source-image/<job_id>")
def source_image(job_id):
    """Gemini E가 촬영한 현재 수술 환경 이미지를 브라우저에 제공."""
    data = read_status(job_id)
    source = data.get("source_image")

    if not source:
        abort(404)

    path = Path(source)

    if not path.exists():
        abort(404)

    response = send_file(path)
    # iPad Safari가 이전 404/이미지를 캐시하지 않도록 강제한다.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/submit", methods=["POST"])
def submit_audio():
    """브라우저에서 녹음된 오디오(webm/opus 등)를 받아 wav로 변환 후 incoming에 저장."""
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"ok": False, "error": "오디오 데이터가 없습니다."}), 400

    # 카메라 캡처와 음성/STT를 동일 job으로 묶기 위해
    # 프론트에서 job_id가 넘어오면 그대로 사용한다.
    # 없으면 기존처럼 새 job을 생성한다.
    job_id = (request.form.get("job_id") or "").strip()

    if not job_id:
        job_id = (
            f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            f"{uuid.uuid4().hex[:6]}"
        )

    # 파일명으로 사용하므로 단순 안전 검증
    if not all(c.isalnum() or c in "_-" for c in job_id):
        return jsonify({
            "ok": False,
            "error": "잘못된 job_id 형식입니다."
        }), 400

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    wav_path = INCOMING_DIR / f"{job_id}.wav"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", tmp_path,
                "-ar", "44100", "-ac", "1", "-sample_fmt", "s16",
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_path)
        return jsonify({
            "ok": False,
            "error": "오디오 변환에 실패했습니다 (ffmpeg).",
            "detail": e.stderr.decode("utf-8", errors="ignore")[-800:],
        }), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    dur = 0.0
    try:
        with wave.open(str(wav_path), "rb") as w:
            dur = w.getnframes() / w.getframerate()
    except Exception:
        pass

    if dur < MIN_DURATION_SEC:
        wav_path.unlink(missing_ok=True)
        return jsonify({
            "ok": False,
            "error": f"녹음이 너무 짧습니다 ({dur:.1f}초). {MIN_DURATION_SEC:.0f}초 이상 말해주세요.",
        }), 400

    write_status(
        job_id,
        stage="queued",
        stt_text="",
        prompt_text="",
        created_at=datetime.now().isoformat(),
        duration_sec=round(dur, 2),
    )

    return jsonify({"ok": True, "job_id": job_id})


# ---------------------------------------------------------------
# 화면 2/3 · 생성 중 · 검토 (한 페이지, JS가 상태 보고 전환)
# ---------------------------------------------------------------
@app.route("/job/<job_id>")
def screen_job(job_id):
    st = read_status(job_id)
    if st["stage"] == "unknown":
        abort(404)
    return render_template("job.html", job_id=job_id)


@app.route("/status/<job_id>")
def api_status(job_id):
    st = read_status(job_id)
    st["has_video"] = result_path(job_id).exists()
    return jsonify(st)


@app.route("/video/<job_id>")
def api_video(job_id):
    p = result_path(job_id)
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="video/mp4")


@app.route("/decision/<job_id>", methods=["POST"])
def api_decision(job_id):
    """승인/거부/긴급정지 상태를 기록하고, 승인된 동작은 Jetson 실행 큐에 등록한다."""
    body = request.get_json(silent=True) or {}
    action = body.get("action")
    if action not in ("approve", "reject", "estop"):
        return jsonify({"ok": False, "error": "잘못된 action"}), 400

    label = {
        "approve": "실행 승인됨 · Jetson 실행 대기",
        "reject": "거부됨 · 다시 명령을 말해주세요",
        "estop": "긴급정지 요청됨",
    }[action]

    write_status(job_id, decision=action, decision_label=label,
                 decided_at=datetime.now().isoformat())

    # 승인된 job에 저장된 로봇 동작을 Jetson으로 전달
    if action == "approve":
        current = read_status(job_id)
        # 인식된 동작이 없는 job은 승인해도 로봇 명령을 만들지 않는다
        # (잘못된 동작이 로봇으로 전달되는 것을 막기 위함).
        robot_action = current.get("robot_action")
        if not robot_action:
            return jsonify({
                "ok": False,
                "error": "이 job에는 인식된 로봇 동작이 없어 승인할 수 없습니다."
            }), 400

        if robot_action not in ("incision", "laparotomy", "surgery_complete"):
            return jsonify({
                "ok": False,
                "error": f"지원하지 않는 로봇 동작: {robot_action}"
            }), 400

        write_robot_command(job_id, robot_action)

        write_status(
            job_id,
            robot_status="approved",
            robot_message="Jetson 실행 대기",
            robot_updated_at=datetime.now().isoformat(),
        )

    return jsonify({"ok": True, "decision": action, "label": label})



@app.route("/robot/command", methods=["GET"])
def api_robot_command():
    return jsonify(read_robot_command())


@app.route("/robot/ack", methods=["POST"])
def api_robot_ack():
    body = request.get_json(silent=True) or {}
    job_id = body.get("job_id")

    data = read_robot_command()

    if not job_id or data.get("job_id") != job_id:
        return jsonify({
            "ok": False,
            "error": "job_id mismatch"
        }), 409

    data["consumed"] = True
    robot_status = body.get("status", "done")
    data["status"] = robot_status
    data["acked_at"] = datetime.now().isoformat()

    robot_message = body.get("message", "")
    if robot_message:
        data["message"] = robot_message

    # Jetson ACK 상태를 원래 웹 job에도 반영
    source_job_id = data.get("source_job_id")
    if source_job_id:
        write_status(
            source_job_id,
            robot_status=robot_status,
            robot_message=robot_message,
            robot_updated_at=datetime.now().isoformat(),
        )
        print(
            f"[ROBOT ACK] {source_job_id} -> "
            f"{robot_status} / {robot_message}"
        )

    tmp = ROBOT_COMMAND_PATH.with_suffix(".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    os.replace(tmp, ROBOT_COMMAND_PATH)

    return jsonify({"ok": True})



@app.route("/robot/surgery_complete", methods=["POST"])
def api_surgery_complete():
    """개복(laparotomy) 이후 의사가 실제 수술을 마치면 호출하는 전용 엔드포인트.
    기존 job/승인 흐름과 분리된 독립 트리거이며, 항상 새 unique job_id로
    surgery_complete robot command를 생성한다."""
    body = request.get_json(silent=True) or {}
    source_job_id = body.get("source_job_id", "manual_surgery_complete")
    data = write_robot_command(source_job_id, "surgery_complete")

    # 수술 완료는 예측 영상/승인 화면을 거치지 않고 즉시 로봇 실행 상태 화면으로 전환한다.
    write_status(
        source_job_id,
        stage="robot_direct",
        robot_action="surgery_complete",
        robot_status="approved",
        robot_message="Jetson 실행 대기",
        robot_updated_at=datetime.now().isoformat(),
    )

    return jsonify({
        "ok": True,
        "job_id": data["job_id"],
        "source_job_id": source_job_id,
        "action": "surgery_complete",
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)