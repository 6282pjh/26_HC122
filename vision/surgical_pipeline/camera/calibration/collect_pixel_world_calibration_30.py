#!/usr/bin/env python3

import cv2
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(os.environ.get("HC122_CALIB_DIR", str(
    PROJECT_ROOT / "config" / "calibration" / "face_to_face_20260821"
)))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
GRID_COLS = 5
GRID_ROWS = 6
DX_MM = 35.0
DY_MM = 38.0
CALIBRATION_JSON = OUTPUT_DIR / "pixel_world_calibration_30.json"
HOMOGRAPHY_JSON = OUTPUT_DIR / "pixel_to_world_homography_30.json"
SNAPSHOT_IMAGE = OUTPUT_DIR / "pixel_world_calibration_30_snapshot.jpg"

world_points=[]
idx=1
for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        world_points.append({"name":f"P{idx:02d}","world":[col*DX_MM,row*DY_MM,0.0]})
        idx+=1
pixel_points=[]
latest_frame=None

def mouse_callback(event,x,y,flags,param):
    if event != cv2.EVENT_LBUTTONDOWN or len(pixel_points)>=len(world_points):
        return
    target=world_points[len(pixel_points)]
    pixel_points.append([int(x),int(y)])
    print(f"[{len(pixel_points):02d}/30] {target['name']} Pixel=({x},{y}) World={tuple(target['world'])}")

def calculate_homography():
    if len(pixel_points)!=len(world_points):
        raise RuntimeError("30개의 Calibration point가 모두 필요합니다.")
    src=np.asarray(pixel_points,np.float32).reshape(-1,1,2)
    dst=np.asarray([p['world'][:2] for p in world_points],np.float32).reshape(-1,1,2)
    H,_=cv2.findHomography(src,dst,method=0)
    pred=cv2.perspectiveTransform(src,H)
    errors=[float(np.linalg.norm(dst[i,0]-pred[i,0])) for i in range(len(world_points))]
    return H,errors,float(np.mean(errors)),float(np.sqrt(np.mean(np.square(errors)))),float(max(errors))

def save(frame):
    H,errors,mean,rmse,maxe=calculate_homography()
    data={"created_at":datetime.now().isoformat(),"camera":{"index":CAMERA_INDEX,"resolution":[FRAME_WIDTH,FRAME_HEIGHT]},"grid":{"cols":GRID_COLS,"rows":GRID_ROWS,"dx_mm":DX_MM,"dy_mm":DY_MM,"width_mm":140.0,"height_mm":190.0},"world_frame":{"origin":"P01","positive_x":"P01 -> P05","positive_y":"P01 -> P26","positive_z":"upward from surgical surface","surface_z_mm":0.0},"points":[]}
    for i,p in enumerate(world_points):
        data['points'].append({"name":p['name'],"pixel":pixel_points[i],"world":p['world'],"reprojection_error_mm":errors[i]})
    CALIBRATION_JSON.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    hdata={"created_at":datetime.now().isoformat(),"source":CALIBRATION_JSON.name,"pixel_to_world_homography":H.tolist(),"mean_error_mm":mean,"rmse_mm":rmse,"max_error_mm":maxe}
    HOMOGRAPHY_JSON.write_text(json.dumps(hdata,indent=2,ensure_ascii=False),encoding='utf-8')
    if frame is not None: cv2.imwrite(str(SNAPSHOT_IMAGE),frame)
    print(f"MEAN={mean:.3f} mm RMSE={rmse:.3f} mm MAX={maxe:.3f} mm")
    print(f"Saved: {CALIBRATION_JSON}")
    print(f"Saved: {HOMOGRAPHY_JSON}")
    print(f"Saved: {SNAPSHOT_IMAGE}")

def draw(frame):
    out=frame.copy()
    for i,(x,y) in enumerate(pixel_points):
        cv2.circle(out,(x,y),5,(0,255,0),-1); cv2.putText(out,world_points[i]['name'],(x+7,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,255,0),1,cv2.LINE_AA)
    if len(pixel_points)<30:
        p=world_points[len(pixel_points)]; text=f"CLICK {p['name']} ({len(pixel_points)+1}/30) World={tuple(p['world'])}"
    else: text="ALL 30 POINTS CAPTURED - press S"
    cv2.putText(out,text,(12,28),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2,cv2.LINE_AA)
    return out

def main():
    global pixel_points, latest_frame
    cap=cv2.VideoCapture(CAMERA_INDEX); cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_WIDTH); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_HEIGHT)
    if not cap.isOpened(): raise RuntimeError("Camera open failed")
    win="Pixel-World Calibration 30 Points"; cv2.namedWindow(win); cv2.setMouseCallback(win,mouse_callback)
    while True:
        ret,frame=cap.read()
        if not ret: break
        latest_frame=frame.copy(); cv2.imshow(win,draw(frame)); key=cv2.waitKey(1)&0xFF
        if key==ord('q'): break
        elif key==ord('u') and pixel_points: pixel_points.pop()
        elif key==ord('r'): pixel_points=[]
        elif key==ord('s'):
            if len(pixel_points)==30: save(draw(latest_frame))
            else: print(f"현재 {len(pixel_points)}개: 30개 모두 필요합니다.")
    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__': main()
