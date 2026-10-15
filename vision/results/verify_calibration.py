"""Recompute supplied planar calibration residuals without opening hardware.

Run from the repository root: python -B vision/results/verify_calibration.py
Only coordinate_transform is imported from the supplied scripts. In particular,
calibrate_base_yaw and webcam_record are compiled, never imported or executed.
"""

import hashlib
import json
import os
import platform
import py_compile
import re
import runpy
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

VISION = Path(__file__).resolve().parents[1]
PIPELINE = VISION / "surgical_pipeline"
CALIB = PIPELINE / "config/calibration/face_to_face_20260821"
os.environ["HC122_CALIB_DIR"] = str(CALIB)
sys.path.insert(0, str(PIPELINE))
sys.dont_write_bytecode = True

from common.coordinates.coordinate_transform import (  # noqa: E402
    load_homography, load_robot_bases, pixel_to_dual_robot,
    rotation_z_deg, world_to_robot, world_vector_to_robot,
)


def no_hardware(*args, **kwargs):
    raise AssertionError("Camera/video/GUI access is forbidden during verification")


def main():
    # Fail closed if a checked CLI accidentally enters a live mode.
    cv2.VideoCapture = no_hardware
    cv2.VideoWriter = no_hardware
    cv2.namedWindow = no_hardware
    scripts = sorted(VISION.rglob("*.py"))
    with tempfile.TemporaryDirectory() as temporary:
        for index, path in enumerate(scripts):
            py_compile.compile(str(path), cfile=str(Path(temporary) / f"{index}.pyc"), doraise=True)

    measured = json.loads((CALIB / "pixel_world_calibration_30.json").read_text(encoding="utf-8"))
    saved = json.loads((CALIB / "pixel_to_world_homography_30.json").read_text(encoding="utf-8"))
    points = measured["points"]
    assert len(points) == 30
    assert [p["name"] for p in points] == [f"P{i:02d}" for i in range(1, 31)]
    pixels = np.asarray([p["pixel"] for p in points], dtype=np.float64)
    worlds = np.asarray([p["world"] for p in points], dtype=np.float64)
    expected_grid = [[col * 35.0, row * 38.0, 0.0] for row in range(6) for col in range(5)]
    np.testing.assert_array_equal(worlds, expected_grid)
    assert measured["camera"] == {"index": 0, "resolution": [640, 480]}
    assert len(np.unique(pixels, axis=0)) == 30
    assert np.all((pixels >= 0) & (pixels < [640, 480]))
    H = load_homography(CALIB)
    assert H.shape == (3, 3) and np.all(np.isfinite(H)) and np.linalg.matrix_rank(H) == 3
    projected = np.column_stack((pixels, np.ones(30))) @ H.T
    assert np.all(np.abs(projected[:, 2]) > 1e-12)
    xy = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(xy - worlds[:, :2], axis=1)
    metrics = {"mean_error_mm": float(errors.mean()),
               "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
               "max_error_mm": float(errors.max())}
    stored_errors = np.asarray([p["reprojection_error_mm"] for p in points])
    np.testing.assert_allclose(errors, stored_errors, rtol=0, atol=1e-9)
    for key, value in metrics.items():
        np.testing.assert_allclose(value, saved[key + "_recomputed"], rtol=0, atol=1e-9)

    robots = load_robot_bases(CALIB)
    assert robots["left"]["base_origin_world_mm"] == [-124.0, 86.0, 27.0]
    assert robots["right"]["base_origin_world_mm"] == [254.0, 86.0, 27.0]
    assert robots["left"]["yaw_deg"] == -90.0 and robots["right"]["yaw_deg"] == 90.0
    examples = []
    for index, pixel in enumerate(pixels):
        world, left, right = pixel_to_dual_robot(*pixel, calib_dir=CALIB)
        np.testing.assert_allclose(world[:2], xy[index], rtol=0, atol=2e-5)
        for side, local in (("left", left), ("right", right)):
            robot = robots[side]
            recovered = rotation_z_deg(robot["yaw_deg"]) @ local + robot["base_origin_world_mm"]
            np.testing.assert_allclose(recovered, world, rtol=0, atol=1e-10)
        if index in (0, 29):
            examples.append({"name": points[index]["name"], "pixel": pixel.tolist(),
                             "world_mm": world.tolist(), "left_base_mm": left.tolist(),
                             "right_base_mm": right.tolist()})
    for robot in robots.values():
        np.testing.assert_allclose(world_to_robot(robot["base_origin_world_mm"], robot), 0, atol=1e-10)
        np.testing.assert_allclose(world_vector_to_robot([0, 0, -1], robot), [0, 0, -1], atol=1e-10)
        R = rotation_z_deg(robot["yaw_deg"])
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(R), 1, atol=1e-12)
    np.testing.assert_allclose(world_to_robot([70, 95, 0], robots["left"]), [-9, 194, -27], atol=1e-10)
    np.testing.assert_allclose(world_to_robot([70, 95, 0], robots["right"]), [9, 184, -27], atol=1e-10)

    cli_cases = [
        ("pixel_to_world_30.py", ["pixel", "398", "132"]),
        ("pixel_to_dual_robot_base_30.py", ["verify"]),
        ("pixel_to_dual_robot_base_30.py", ["pixel", "398", "132"]),
        ("pixel_to_dual_robot_base_30.py", ["world", "70", "95", "0"]),
        ("pixel_to_dual_robot_base_30.py", ["line", "315", "313", "315", "223"]),
    ]
    previous_argv = sys.argv[:]
    try:
        for name, args in cli_cases:
            script = PIPELINE / "camera/calibration" / name
            sys.argv = [str(script), *args]
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = previous_argv

    # Scan all deliverables, including documentation, without printing matched secrets.
    patterns = {
        "private_path": r"/home/[^/\s]+|[A-Za-z]:[\\/]Users[\\/]|OneDrive|/raid",
        "email": r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}",
        "network_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b|[\w.-]+\.ts\.net",
        "credential": r"(?:gsk_|hf_|sk-)[A-Za-z0-9]{20,}|(?:api[_-]?key|password|passwd|token)\s*=\s*['\"][^'\"]+",
    }
    findings, files = [], []
    forbidden = {".pyc", ".pyo", ".mp4", ".avi", ".mov", ".zip", ".pt", ".pth", ".ckpt"}
    for path in sorted(VISION.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(VISION).as_posix()
        assert path.stat().st_size <= 5_000_000, relative
        assert path.suffix.lower() not in forbidden, relative
        if path.suffix in {".py", ".md", ".json", ".txt"} and path != Path(__file__).resolve():
            content = path.read_text(encoding="utf-8")
            for name, pattern in patterns.items():
                if re.search(pattern, content, re.I):
                    findings.append({"file": relative, "category": name})
        if path.parent == CALIB:
            files.append({"file": relative, "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    assert not findings, findings
    image = cv2.imdecode(np.frombuffer(
        (CALIB / "pixel_world_calibration_30_snapshot.jpg").read_bytes(), dtype=np.uint8
    ), cv2.IMREAD_COLOR)
    assert image is not None and image.shape[:2] == (480, 640)
    report = {
        "verification_date": "2026-10-15", "status": "passed",
        "environment": {"os": platform.system(), "python": platform.python_version(),
                        "numpy": np.__version__, "opencv": cv2.__version__},
        "source_date": measured["created_at"], "source_homography_note": saved["note"],
        "method": "float64 homogeneous projection using supplied H; Euclidean XY residuals on the same 30 fitting correspondences",
        "point_count": len(points), "homography": H.tolist(), **metrics,
        "max_point": points[int(np.argmax(errors))]["name"],
        "stored_point_residual_max_difference_mm": float(np.max(np.abs(errors - stored_errors))),
        "stored_recomputed_metrics_match_atol_mm": 1e-9,
        "robots": robots, "examples": examples,
        "point_residuals": [{"name": p["name"], "projected_xy_mm": out.tolist(), "residual_mm": float(err)}
                            for p, out, err in zip(points, xy, errors)],
        "checks": {"syntax_files": len(scripts), "grid_and_pixels": "passed",
                   "base_origin_rotation_roundtrip": "passed", "offline_cli_cases": len(cli_cases),
                   "camera_gui_access_blocked": True, "sensitive_pattern_findings": findings,
                   "all_files_under_5MB": True, "forbidden_artifacts": "none",
                   "snapshot_resolution": [640, 480]},
        "evidence_files": files,
        "not_verified": ["camera capture and HSV detection", "motor movement and yaw measurement",
                         "independent physical validation", "robot tool-tip accuracy", "policy or executor integration"],
    }
    target = VISION / "results/calib_20261015.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("PASS: syntax, data, coordinates, offline CLI, size and sensitive-pattern checks")


if __name__ == "__main__":
    main()
