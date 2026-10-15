# vision — C920 절개선 검출 · 30점 좌표 보정

로봇이 실제로 자르려면 **카메라 화면 속 절개선이 로봇 기준으로 어디에 있는지** 알아야 합니다. vision은 수술대 위 Logitech C920 영상에서 절개선의 시작·끝을 찾고, 30점 Homography 캘리브레이션으로 이 Pixel 좌표를 작업 평면의 실제 좌표(mm)와 마주 보고 놓인 좌·우 DOFBOT 각각의 베이스 좌표로 변환합니다.

<img src="surgical_pipeline/config/calibration/face_to_face_20260821/pixel_world_calibration_30_snapshot.jpg" width="480" alt="30점 캘리브레이션 스냅샷">

## 처리 흐름

```
C920 영상 (640×480)
  └─ 고정 ROI → HSV 마스크(붉은색 두 구간) → opening·closing → 세로로 긴 contour 선택
       └─ 절개선 START(아래) · END(위) Pixel 좌표
            └─ Homography H (30점) ──▶ World 좌표 (mm, 피부 평면)
                 └─ 원점 이동 + yaw 역회전 ──▶ 좌 Base / 우 Base 좌표 (mm)
```

## 핵심 설계

**1. 절개선 검출**
절개선은 붉은 마커로 그려져 있습니다. OpenCV에서 빨간색은 색상(H) 축의 양 끝(0 부근과 179 부근)에 나뉘어 있으므로 `[0, H1_MAX]`와 `[H2_MIN, 179]` 두 구간의 마스크를 합칩니다. 2×2 opening으로 잡음을 지우고 3×3 closing으로 끊긴 선을 잇습니다. 그다음 세로로 긴 contour를 우선 선택하고, 지수 평활(`ALPHA=0.30`)로 프레임 사이 떨림을 줄입니다. 화면 아래쪽 끝이 START, 위쪽 끝이 END입니다.

**2. 30점 Homography 캘리브레이션**
피부 패드 평면 위에 5열×6행(35 mm × 38 mm 간격, 140×190 mm) 격자 30점을 두고, 각 점의 Pixel 위치를 클릭해 기록합니다. 30점 전체로 `cv2.findHomography`를 풀어 Pixel → World 변환 행렬 H를 구합니다.

```text
[a, b, c]^T = H [u, v, 1]^T
World = [a/c, b/c, z]
```

**3. 마주 보는 양팔의 베이스 좌표**
두 DOFBOT은 수술대 좌우에서 서로 마주 보고 있어, 같은 World 점도 팔마다 좌표가 다릅니다. 각 팔의 베이스 원점을 빼고 yaw만큼 거꾸로 회전시켜 팔 기준 좌표를 얻습니다. 방향 벡터(칼날 축 등)에는 회전만 적용합니다.

```text
Base        = Rz(-yaw) (World - base_origin)
Base_vector = Rz(-yaw) World_vector
```

| 좌표계 | 원점 | +X / +Y / +Z | 단위 |
|---|---|---|---|
| Pixel `(u,v)` | 영상 왼쪽 위 | u 오른쪽 / v 아래쪽 | px |
| World | 격자 P01 `(0,0,0)` | P01→P05 / P01→P26 / 피부 평면 위쪽 | mm |
| 좌 Base | World `(-124,86,27)`, yaw −90° | World −Y / World +X / World +Z | mm |
| 우 Base | World `(254,86,27)`, yaw +90° | World +Y / World −X / World +Z | mm |

두 확장보드가 같은 I2C 주소(`0x15`)를 쓰므로 좌측은 버스 1, 우측은 버스 8로 나눠 연결합니다. 베이스 yaw 방향은 `calibrate_base_yaw.py`로 실제 팔을 움직여 확인해 기록했습니다.

## 결과

**캘리브레이션 잔차** (2026-08-21 측정, H 적합에 사용한 30점의 평면 잔차)

| 항목 | 결과 |
|---|---|
| 평균 잔차 | **0.78 mm** (0.778569806) |
| RMSE | 0.90 mm |
| 최대 잔차 | 1.98 mm (P09) |

**변환 예시** — `pixel 398 132` 실행 결과

```text
Pixel: (398.0,132.0)
World: (0.036, 0.319, 0.000) mm
Left Base: (85.681, 124.036, -27.000) mm
Right Base: (-85.681, 253.964, -27.000) mm
```

H 행렬, 30점별 잔차, 좌표 변환 왕복 검사 결과는 [`results/calib_20261015.json`](results/calib_20261015.json)과 [`results/verification_20261015.md`](results/verification_20261015.md)에 정리했습니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `surgical_pipeline/camera/detect_incision_c920.py` | 절개선 검출, START·END Pixel 출력 |
| `surgical_pipeline/camera/detect_incision_c920_record.py` | 검출 화면 녹화 |
| `surgical_pipeline/camera/webcam_record.py` | C920 원본 화면 녹화 |
| `surgical_pipeline/camera/calibration/collect_pixel_world_calibration_30.py` | 30점 클릭 수집, H 추정, 잔차·스냅샷 저장 |
| `surgical_pipeline/camera/calibration/pixel_to_world_30.py` | Pixel → World 변환 CLI |
| `surgical_pipeline/camera/calibration/pixel_to_dual_robot_base_30.py` | Pixel/World/선분 → 좌·우 베이스 좌표 CLI |
| `surgical_pipeline/common/coordinates/coordinate_transform.py` | 좌표 변환 함수 |
| `surgical_pipeline/common/calibration/calibrate_base_yaw.py` | 베이스 yaw 방향 측정 (실제 모터 구동) |
| `surgical_pipeline/config/calibration/face_to_face_20260821/` | 캘리브레이션 측정 자료 (JSON 3개, 스냅샷) |
| `results/` | 캘리브레이션 검증 스크립트와 결과 |

<details>
<summary><b>▶ 실행 방법</b></summary>

모든 명령은 `surgical_pipeline/`에서 실행합니다 (코드가 이 폴더를 기준으로 경로와 `common.coordinates` import를 잡습니다).

```bash
python3 -m pip install -r vision/requirements.txt
cd vision/surgical_pipeline
```

**좌표 변환 (카메라 없이 실행)**

```bash
python3 -B camera/calibration/pixel_to_world_30.py pixel 398 132
python3 -B camera/calibration/pixel_to_dual_robot_base_30.py verify
python3 -B camera/calibration/pixel_to_dual_robot_base_30.py pixel 398 132
python3 -B camera/calibration/pixel_to_dual_robot_base_30.py world 70 95 0
python3 -B camera/calibration/pixel_to_dual_robot_base_30.py line 315 313 315 223
python3 -B ../results/verify_calibration.py
```

**검출·녹화 (C920 필요)**

```bash
python3 camera/detect_incision_c920.py           # s: START/END 출력, q: 종료
python3 camera/detect_incision_c920_record.py
python3 camera/webcam_record.py
python3 camera/calibration/pixel_to_world_30.py test
python3 camera/calibration/pixel_to_dual_robot_base_30.py test
```

| 설정 | 위치와 기본값 |
|---|---|
| 카메라 / 해상도 | 각 스크립트 상단 `CAMERA_INDEX=0`, 640×480 |
| ROI | `x=[250,385)`, `y=[170,355)` |
| HSV | `H1_MAX=9`, `H2_MIN=136`, `S_MIN=25`, `V_MIN=61` (실행 중 `HSV CONTROL` 트랙바로 조정) |
| contour | `MIN_AREA=8`, `MIN_HEIGHT=12`, `MAX_WIDTH=35`, `MIN_VERTICAL_RATIO=1.5` |
| 저장 위치 | `HC122_VIDEO_DIR`(녹화, 기본 `~/Videos`), `HC122_CALIB_DIR`(캘리브레이션 자료) |

**재캘리브레이션**

```bash
export HC122_CALIB_DIR="$HOME/hc122-calibration-new"     # 기존 자료를 덮어쓰지 않도록 새 폴더 지정
python3 camera/calibration/collect_pixel_world_calibration_30.py
# P01 → P30 순서로 클릭 (u: 한 점 취소, r: 초기화), 30점이 모이면 s로 저장
```

카메라 위치·각도·해상도, 작업 평면의 높이·기울기가 바뀌면 H를 다시 측정합니다.

**베이스 yaw 측정** (실제 모터 구동 — 칼날을 제거하고 주변 공간을 확보한 뒤 실행)

```bash
sudo -E python3 common/calibration/calibrate_base_yaw.py left    # right / both
```

</details>
