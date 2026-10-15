#!/usr/bin/env python3

import sys
import math
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT_ROOT))
from common.coordinates.coordinate_transform import load_homography,load_robot_bases,pixel_to_world,world_to_robot,world_vector_to_robot

CAMERA_INDEX=0
FRAME_WIDTH=640
FRAME_HEIGHT=480

def fmt(p): return f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})"

def convert(u,v,z=0.0):
    H=load_homography(); robots=load_robot_bases(); w=pixel_to_world(u,v,H,z); l=world_to_robot(w,robots['left']); r=world_to_robot(w,robots['right']); return w,l,r

def print_conv(u,v,w,l,r):
    print(f"Pixel: ({u:.1f},{v:.1f})")
    print(f"World: {fmt(w)} mm")
    print(f"Left Base: {fmt(l)} mm")
    print(f"Right Base: {fmt(r)} mm")
def verify():
    robots=load_robot_bases(); pts={'P01':[0,0,0],'P05':[140,0,0],'P26':[0,190,0],'P30':[140,190,0],'CENTER':[70,95,0]}
    for n,p in pts.items():
        p=np.asarray(p,float); print(f"[{n}] World {fmt(p)} | Left {fmt(world_to_robot(p,robots['left']))} | Right {fmt(world_to_robot(p,robots['right']))}")
    axis=np.array([0.,0.,-1.]); print('Blade axis Left',fmt(world_vector_to_robot(axis,robots['left'])),'Right',fmt(world_vector_to_robot(axis,robots['right'])))

def live():
    H=load_homography(); robots=load_robot_bases(); cap=cv2.VideoCapture(CAMERA_INDEX); cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT); last=None
    def cb(event,x,y,flags,param):
        nonlocal last
        if event==cv2.EVENT_LBUTTONDOWN:
            w=pixel_to_world(x,y,H,0.0); l=world_to_robot(w,robots['left']); r=world_to_robot(w,robots['right']); last=(x,y,w,l,r); print_conv(x,y,w,l,r)
    win='Pixel -> World -> Dual Robot Base'; cv2.namedWindow(win); cv2.setMouseCallback(win,cb)
    while True:
        ret,frame=cap.read()
        if not ret: break
        if last:
            x,y,w,l,r=last; cv2.circle(frame,(x,y),6,(0,255,0),-1); lines=[f'World {w[0]:.1f},{w[1]:.1f},{w[2]:.1f}',f'Left {l[0]:.1f},{l[1]:.1f},{l[2]:.1f}',f'Right {r[0]:.1f},{r[1]:.1f},{r[2]:.1f}']
            for i,t in enumerate(lines): cv2.putText(frame,t,(12,28+24*i),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,0),1)
        cv2.imshow(win,frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
    cap.release(); cv2.destroyAllWindows()

def main():
    if len(sys.argv)<2: raise SystemExit('test | verify | pixel U V [Z] | world X Y Z | line U1 V1 U2 V2')
    mode=sys.argv[1].lower(); robots=load_robot_bases()
    if mode=='verify': verify()
    elif mode=='test': live()
    elif mode=='pixel':
        if len(sys.argv) not in (4,5): raise SystemExit('pixel U V [Z]')
        z=float(sys.argv[4]) if len(sys.argv)==5 else 0.0; w,l,r=convert(float(sys.argv[2]),float(sys.argv[3]),z); print_conv(float(sys.argv[2]),float(sys.argv[3]),w,l,r)
    elif mode=='world':
        if len(sys.argv)!=5: raise SystemExit('world X Y Z')
        w=np.array(list(map(float,sys.argv[2:5]))); print('World',fmt(w),'Left',fmt(world_to_robot(w,robots['left'])),'Right',fmt(world_to_robot(w,robots['right'])))
    elif mode=='line':
        if len(sys.argv)!=6: raise SystemExit('line U1 V1 U2 V2')
        u1,v1,u2,v2=map(float,sys.argv[2:6]); w1,l1,r1=convert(u1,v1); w2,l2,r2=convert(u2,v2); print('START'); print_conv(u1,v1,w1,l1,r1); print('END'); print_conv(u2,v2,w2,l2,r2); d=w2-w1; length=float(np.linalg.norm(d[:2])); yaw=math.degrees(math.atan2(d[1],d[0])) if length else 0.0; print(f'Incision length={length:.3f} mm, world yaw={yaw:.3f} deg')
    else: raise SystemExit(f'Unknown mode: {mode}')

if __name__=='__main__':main()
