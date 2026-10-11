"""
flask_capture_routes.py
기존 app.py에서 register_capture_routes(app)로 불러서 쓰는 캡처 라우트 모음.
"""

import uuid
from datetime import datetime

from flask import request, jsonify

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from capture_bridge.capture_bridge import (
    trigger_capture,
    mark_capture_running,
    read_capture_command,
    save_captured_image,
    mark_capture_failed,
)


def register_capture_routes(app, on_capture_uploaded=None):
    @app.route("/capture/trigger", methods=["POST"])
    def capture_trigger():
        """마이크 녹음 시작과 함께 카메라 녹화 START 요청."""
        data = request.get_json(silent=True) or {}

        job_id = (data.get("job_id") or "").strip()
        if not job_id:
            job_id = (
                f"web_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                f"_{uuid.uuid4().hex[:6]}"
            )

        trigger_capture(job_id, action="start")

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "action": "start",
        })


    @app.route("/capture/stop", methods=["POST"])
    def capture_stop():
        """마이크 녹음 종료와 함께 카메라 녹화 STOP 요청."""
        data = request.get_json(silent=True) or {}
        job_id = (data.get("job_id") or "").strip()

        if not job_id:
            return jsonify({
                "ok": False,
                "error": "job_id missing"
            }), 400

        trigger_capture(job_id, action="stop")

        return jsonify({
            "ok": True,
            "job_id": job_id,
            "action": "stop",
        })


    @app.route("/capture/pending", methods=["GET"])
    def capture_pending():
        return jsonify(read_capture_command())

    @app.route("/capture/ack", methods=["POST"])
    def capture_ack():
        data = request.get_json(silent=True) or {}
        job_id = data.get("job_id")
        if not job_id:
            return jsonify({"accepted": False, "error": "job_id missing"}), 400
        accepted = mark_capture_running(job_id)
        return jsonify({"accepted": accepted})

    @app.route("/capture/upload", methods=["POST"])
    def capture_upload():
        job_id = request.form.get("job_id")
        if not job_id:
            return jsonify({"ok": False, "error": "job_id missing"}), 400
        if "image" not in request.files:
            return jsonify({"ok": False, "error": "image file missing"}), 400

        try:
            saved_path = save_captured_image(job_id, request.files["image"])
        except Exception as e:
            mark_capture_failed(job_id, str(e))
            return jsonify({"ok": False, "error": str(e)}), 500

        if on_capture_uploaded is not None:
            on_capture_uploaded(job_id, saved_path)

        return jsonify({"ok": True, "job_id": job_id, "saved_path": saved_path})

    @app.route("/capture/status/<job_id>", methods=["GET"])
    def capture_status(job_id):
        cmd = read_capture_command()
        if cmd.get("job_id") != job_id:
            return jsonify({"status": "unknown"}), 404
        return jsonify(cmd)
