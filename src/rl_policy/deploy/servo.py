"""Simulated joint angles <-> DOFBOT servo degrees.

Measured on the right arm (I2C bus 8): joints 1-4 run inverted about 90 deg,
joint 5 is not inverted and sits 135 deg off the simulated zero. The left arm
(bus 1) has not been measured separately and is assumed to match.

Same mapping as the task reads the taught motion with
(task/.../dofbot_opening_env.py, DofbotOpeningEnv._load_motion) and as
agx/replay_on_agx.py applies (to_servo_deg + --j5-offset 135).
"""

import math

ARM_JOINTS = ("arm_joint1", "arm_joint2", "arm_joint3", "arm_joint4", "arm_joint5")
# Servo travel in degrees: joints 1-4, wrist roll, gripper.
SERVO_LIMITS = ((0.0, 180.0),) * 4 + ((0.0, 270.0), (0.0, 180.0))
INVERTED = (True, True, True, True, False)
J5_OFFSET_DEG = 135.0


def sim_to_servo(q_rad):
    """Five simulated arm-joint angles (rad) -> five servo angles (deg)."""
    out = []
    for i, q in enumerate(q_rad[:5]):
        deg = math.degrees(float(q))
        if i == 4:
            deg += J5_OFFSET_DEG
        out.append((-deg if INVERTED[i] else deg) + 90.0)
    return out


def servo_to_sim(servo_deg):
    """Five servo angles (deg) -> five simulated arm-joint angles (rad)."""
    out = []
    for i, s in enumerate(servo_deg[:5]):
        deg = float(s) - 90.0
        deg = -deg if INVERTED[i] else deg
        if i == 4:
            deg -= J5_OFFSET_DEG
        out.append(math.radians(deg))
    return out
