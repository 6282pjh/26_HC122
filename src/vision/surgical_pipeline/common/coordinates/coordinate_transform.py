#!/usr/bin/env python3

import json
import os
import math
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIB_DIR = Path(os.environ.get("HC122_CALIB_DIR", str(
    PROJECT_ROOT / "config" / "calibration" / "face_to_face_20260821"
)))


def rotation_z_deg(angle_deg):
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_homography(calib_dir=DEFAULT_CALIB_DIR):
    data = load_json(Path(calib_dir) / "pixel_to_world_homography_30.json")
    return np.asarray(data["pixel_to_world_homography"], dtype=np.float64)


def load_robot_bases(calib_dir=DEFAULT_CALIB_DIR):
    data = load_json(Path(calib_dir) / "robot_base_calibration_face_to_face.json")
    return data["robots"]


def pixel_to_world(u, v, H, world_z_mm=0.0):
    pixel = np.array([[[float(u), float(v)]]], dtype=np.float32)
    out = cv2.perspectiveTransform(pixel, H)
    return np.array([float(out[0,0,0]), float(out[0,0,1]), float(world_z_mm)], dtype=np.float64)


def world_to_robot(world_xyz, robot_info):
    origin = np.asarray(robot_info["base_origin_world_mm"], dtype=np.float64)
    yaw_deg = float(robot_info["yaw_deg"])
    return rotation_z_deg(-yaw_deg) @ (np.asarray(world_xyz, dtype=np.float64) - origin)


def world_vector_to_robot(vector_world, robot_info):
    return rotation_z_deg(-float(robot_info["yaw_deg"])) @ np.asarray(vector_world, dtype=np.float64)


def pixel_to_dual_robot(u, v, world_z_mm=0.0, calib_dir=DEFAULT_CALIB_DIR):
    H = load_homography(calib_dir)
    robots = load_robot_bases(calib_dir)
    world = pixel_to_world(u, v, H, world_z_mm)
    left = world_to_robot(world, robots["left"])
    right = world_to_robot(world, robots["right"])
    return world, left, right
