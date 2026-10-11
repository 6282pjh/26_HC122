"""
capture_bridge.py

노트북 카메라 <-> DIS04 서버 사이의 "캡처 요청" 상태를 파일 기반으로 관리하는 모듈.
"""

import json
import os
import threading
from datetime import datetime

CAPTURE_COMMAND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "capture_command.json"
)
CAPTURES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "captures"
)
os.makedirs(CAPTURES_DIR, exist_ok=True)

_lock = threading.Lock()


def _now():
    return datetime.now().isoformat()


def read_capture_command():
    with _lock:
        if not os.path.exists(CAPTURE_COMMAND_PATH):
            return {"status": "idle"}
        try:
            with open(CAPTURE_COMMAND_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"status": "idle"}


def write_capture_command(data):
    with _lock:
        tmp_path = CAPTURE_COMMAND_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, CAPTURE_COMMAND_PATH)


def trigger_capture(job_id, action="start"):
    """카메라 클라이언트에 start/stop 요청을 전달."""
    write_capture_command(
        {
            "status": "pending",
            "job_id": job_id,
            "action": action,
            "requested_at": _now(),
        }
    )
    return job_id


def mark_capture_running(job_id):
    cmd = read_capture_command()
    if cmd.get("job_id") == job_id and cmd.get("status") == "pending":
        cmd["status"] = "running"
        cmd["running_at"] = _now()
        write_capture_command(cmd)
        return True
    return False


def save_captured_image(job_id, file_storage):
    os.makedirs(CAPTURES_DIR, exist_ok=True)
    save_path = os.path.join(CAPTURES_DIR, f"{job_id}.png")
    file_storage.save(save_path)

    cmd = read_capture_command()

    # 중요:
    # 이미지 업로드가 끝나는 동안 같은 job_id의 STOP 요청이 들어올 수 있다.
    # 이때 STOP(pending)을 done으로 덮어쓰면 카메라 클라이언트가 STOP을 놓친다.
    # 따라서 현재 명령이 실제 START 처리 중일 때만 START를 done으로 변경한다.
    if (
        cmd.get("job_id") == job_id
        and cmd.get("action") == "start"
        and cmd.get("status") == "running"
    ):
        cmd["status"] = "done"
        cmd["saved_path"] = save_path
        cmd["completed_at"] = _now()
        write_capture_command(cmd)

    elif (
        cmd.get("job_id") == job_id
        and cmd.get("action") == "stop"
        and cmd.get("status") == "pending"
    ):
        # STOP 요청이 이미 들어온 상태면 절대 덮어쓰지 않는다.
        print(f"[capture] 이미지 업로드 완료 - pending STOP 보존: {job_id}")

    return save_path


def mark_capture_failed(job_id, error_message):
    cmd = read_capture_command()
    if cmd.get("job_id") == job_id:
        cmd["status"] = "error"
        cmd["error"] = error_message
        cmd["failed_at"] = _now()
        write_capture_command(cmd)
