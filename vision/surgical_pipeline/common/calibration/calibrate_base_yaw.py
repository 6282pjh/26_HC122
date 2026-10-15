#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path
import smbus
from Arm_Lib import Arm_Device

PROJECT_ROOT=Path(__file__).resolve().parents[2]
OUTPUT=Path(os.environ.get("HC122_CALIB_DIR", str(
    PROJECT_ROOT / "config" / "calibration" / "face_to_face_20260821"
))) / "robot_base_calibration_face_to_face.json"
BASES={'left':{'origin':[-124.0,86.0,27.0],'bus':1},'right':{'origin':[254.0,86.0,27.0],'bus':8}}
STANDBY=[90,91,90,90,89,173]
DIRECTION=[90,126,90,90,89,173]
J1_PLUS=[105,126,90,90,89,173]
YES={'y','yes','네','예','ㅇ','맞아','맞음'}
NO={'n','no','아니오','아니요','ㄴ','아님','틀림'}

def ask(q):
    while True:
        a=input(q+' [네/아니오]: ').strip().lower()
        if a in YES:return True
        if a in NO:return False

def connect(bus):
    arm=Arm_Device(); time.sleep(.3)
    try: arm.bus.close()
    except Exception: pass
    arm.bus=smbus.SMBus(bus); return arm

def move(arm,pose,ms=1600):
    arm.Arm_serial_servo_write6_array(pose,ms); time.sleep(ms/1000+.4)

def norm(a):
    while a>180:a-=360
    while a<=-180:a+=360
    return a

def observed_direction():
    items=[(0.0,'World +X (P01->P05) 방향입니까?'),(90.0,'World +Y (P01->P26) 방향입니까?'),(180.0,'World -X 방향입니까?'),(-90.0,'World -Y 방향입니까?')]
    for ang,q in items:
        if ask(q): return ang
    return None

def calibrate(side):
    cfg=BASES[side]; arm=connect(cfg['bus'])
    try:
        input('칼날 제거/주변 안전 확인 후 Enter...')
        move(arm,STANDBY); input('STANDBY 안전 확인 후 Enter...')
        move(arm,DIRECTION)
        obs=observed_direction()
        if obs is None: raise RuntimeError('축 방향이 0/90/180/-90 중 하나로 판정되지 않았습니다.')
        yaw=norm(obs+90.0)  # test pose horizontal direction = Robot -Y
        print(f'Yaw={yaw:.1f} deg')
        input('J1 90->105 검증 전 Enter...'); move(arm,J1_PLUS,1200)
        ccw=ask('위에서 봤을 때 반시계방향(CCW)으로 회전했습니까?')
        move(arm,STANDBY)
        return {'i2c_bus':cfg['bus'],'i2c_address':'0x15','base_origin_world_mm':cfg['origin'],'yaw_deg':yaw,'direction_test_pose':DIRECTION,'direction_test_local_axis':'Robot -Y','direction_test_observed_world_angle_deg':obs,'joint1_90_to_105':'CCW' if ccw else 'not_CCW'}
    finally:
        try: move(arm,STANDBY,1000)
        except Exception: pass
        try: arm.bus.close()
        except Exception: pass

def main():
    if len(sys.argv)!=2 or sys.argv[1] not in ('left','right','both'): raise SystemExit('sudo python3 calibrate_base_yaw.py left|right|both')
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    data={'created_at':'2026-08-21','layout':'face_to_face','world_frame':{'origin':'P01','positive_x':'P01 -> P05','positive_y':'P01 -> P26','positive_z':'up','surface_z_mm':0.0},'robots':{}}
    if OUTPUT.exists():
        try: data=json.loads(OUTPUT.read_text(encoding='utf-8'))
        except Exception: pass
    for side in (['left','right'] if sys.argv[1]=='both' else [sys.argv[1]]):
        data['robots'][side]=calibrate(side); OUTPUT.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8'); print(f'Saved: {OUTPUT}')

if __name__=='__main__':main()
