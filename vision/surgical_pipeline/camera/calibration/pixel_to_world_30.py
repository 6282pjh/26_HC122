#!/usr/bin/env python3

import sys
import cv2
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT))
from common.coordinates.coordinate_transform import load_homography,pixel_to_world

CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480

def main():
    H=load_homography()
    if len(sys.argv)<2:
        print("Usage: pixel_to_world_30.py test | pixel U V [Z]"); return
    mode=sys.argv[1].lower()
    if mode=='pixel':
        if len(sys.argv) not in (4,5): raise SystemExit('pixel U V [Z]')
        z=float(sys.argv[4]) if len(sys.argv)==5 else 0.0
        p=pixel_to_world(float(sys.argv[2]),float(sys.argv[3]),H,z)
        print(f"World = ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) mm")
    elif mode=='test':
        cap=cv2.VideoCapture(CAMERA_INDEX); cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)
        last=None
        def cb(event,x,y,flags,param):
            nonlocal last
            if event==cv2.EVENT_LBUTTONDOWN:
                last=(x,y,pixel_to_world(x,y,H,0.0)); print(f"Pixel ({x},{y}) -> World ({last[2][0]:.2f}, {last[2][1]:.2f}, {last[2][2]:.2f}) mm")
        win='Pixel -> World Test'; cv2.namedWindow(win); cv2.setMouseCallback(win,cb)
        while True:
            ret,frame=cap.read()
            if not ret: break
            if last:
                x,y,p=last; cv2.circle(frame,(x,y),6,(0,255,0),-1); cv2.putText(frame,f"World({p[0]:.1f},{p[1]:.1f},{p[2]:.1f})",(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
            cv2.imshow(win,frame)
            if cv2.waitKey(1)&0xFF==ord('q'): break
        cap.release(); cv2.destroyAllWindows()
    else: raise SystemExit(f'Unknown mode: {mode}')

if __name__=='__main__': main()
