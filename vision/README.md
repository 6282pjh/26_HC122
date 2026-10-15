# vision — C920 절개선 검출 · 30점 좌표 보정

상부 Logitech C920 영상에서 HSV 색 조건으로 절개선을 찾아 **시작·끝 Pixel 좌표**를 출력하고, 30점 Homography로 Pixel → World(mm) → 좌·우 로봇 베이스 좌표로 변환하는 코드와 측정 자료입니다.

검출 프로그램과 좌표 변환 프로그램은 **따로 실행**합니다. 검출 창에서 `s`를 누르면 Pixel 좌표가 출력되고, 이 값을 좌표 변환 CLI에 넣어 World·베이스 좌표를 얻습니다.

## 파일 구성

| 경로 | 역할 |
|---|---|
| `surgical_pipeline/vision/detect_incision_c920.py` | 고정 ROI 안에서 HSV 마스크·contour 조건으로 절개선 검출, 화면 아래 START·위 END Pixel 출력 |
| `surgical_pipeline/vision/detect_incision_c920_record.py` | 검출 화면과 표시를 MP4로 녹화 |
| `surgical_pipeline/vision/webcam_record.py` | V4L2·MJPG로 C920 원본 화면을 AVI로 녹화 |
| `surgical_pipeline/vision/calibration/collect_pixel_world_calibration_30.py` | 30점 순서대로 클릭, H 추정, 대응점·오차·스냅샷 저장 |
| `surgical_pipeline/vision/calibration/pixel_to_world_30.py` | Pixel → World CLI 및 카메라 클릭 확인 |
| `surgical_pipeline/vision/calibration/pixel_to_dual_robot_base_30.py` | Pixel/World/선분 → 좌·우 베이스 좌표 출력, 기준 World 점 확인 |
| `surgical_pipeline/common/coordinates/coordinate_transform.py` | H·베이스 JSON 로드, 점·벡터 좌표 변환 함수 |
| `surgical_pipeline/common/calibration/calibrate_base_yaw.py` | **실제 모터를 움직여** 베이스 yaw 방향을 사람이 판정하는 측정 스크립트 |
| `surgical_pipeline/config/calibration/face_to_face_20260821/` | 2026-08-21 측정 JSON 3개와 30점 스냅샷 |
| `results/verify_calibration.py` | 카메라·로봇 없이 실행하는 문법·수치·좌표 변환·CLI·파일 검사 |
| `results/calib_20261015.json` | 재계산 수치, 30점별 잔차, 검사 항목 |
| `results/verification_20261015.md` | 검사 조건과 검증하지 않은 범위 |
| `requirements.txt` | 검출·좌표 변환에 필요한 패키지 |

## 설치와 실행 위치

모든 명령은 `surgical_pipeline/`에서 실행합니다. 코드가 이 폴더를 기준으로 경로와 import(`common.coordinates`)를 잡기 때문입니다. Python 3, NumPy, GUI 기능이 있는 OpenCV가 필요합니다.

```bash
python3 -m pip install -r vision/requirements.txt
cd vision/surgical_pipeline
```

장비에 이미 설치된 OpenCV가 있다면 덮어 설치하기 전에 환경을 확인하세요. `webcam_record.py`의 `CAP_V4L2`는 Linux용입니다. yaw 측정에만 `Arm_Lib`, `smbus`, I2C 접근 권한이 추가로 필요하며, 좌표 변환과 파일 검사에는 필요 없습니다.

### 장비 없이 실행하는 명령

```bash
python3 -B vision/calibration/pixel_to_world_30.py pixel 398 132
python3 -B vision/calibration/pixel_to_dual_robot_base_30.py verify
python3 -B vision/calibration/pixel_to_dual_robot_base_30.py pixel 398 132
python3 -B vision/calibration/pixel_to_dual_robot_base_30.py world 70 95 0
python3 -B vision/calibration/pixel_to_dual_robot_base_30.py line 315 313 315 223
python3 -B ../results/verify_calibration.py
```

`line`의 좌표는 격자 점을 사용한 계산 예시입니다. `pixel U V [Z]`의 Z 기본값은 World 0 mm이고, Z를 넣으면 그 높이로 계산합니다. H가 깊이를 측정하는 것은 아닙니다.

### 카메라가 필요한 명령

```bash
python3 vision/detect_incision_c920.py
python3 vision/detect_incision_c920_record.py
python3 vision/webcam_record.py
python3 vision/calibration/pixel_to_world_30.py test
python3 vision/calibration/pixel_to_dual_robot_base_30.py test
```

검출기는 `s`로 마지막 START/END를 출력하고 `q`로 종료합니다. 녹화 스크립트는 실행하면 바로 녹화를 시작하고 `q`로 저장·종료합니다. `webcam_record.py`는 최상위 코드에서 바로 카메라를 열기 때문에 다른 코드에서 import하지 않습니다.

| 설정 | 위치와 기본값 |
|---|---|
| 카메라 | 각 스크립트 상단 `CAMERA_INDEX=0` |
| 해상도 | `WIDTH/HEIGHT` 또는 `FRAME_WIDTH/FRAME_HEIGHT`: 640×480 |
| 녹화 FPS | 두 녹화 스크립트 `FPS=30` (실제 카메라 지원 여부는 미검증) |
| ROI | 두 검출 스크립트 상단: `x=[250,385)`, `y=[170,355)` |
| HSV | 두 검출 스크립트 상단: `H1_MAX=9`, `H2_MIN=136`, `S_MIN=25`, `V_MIN=61` |
| 실시간 HSV | `HSV CONTROL` 창의 트랙바 (변경값은 파일로 저장되지 않음) |
| contour | `MIN_AREA=8`, `MIN_HEIGHT=12`, `MAX_WIDTH=35`, `MIN_VERTICAL_RATIO=1.5` |
| smoothing | `ALPHA=0.30` (비녹화 검출기는 `main()` 내부) |

HSV의 H는 OpenCV의 0–179 범위이며 `[0,H1_MAX]`와 `[H2_MIN,179]` 마스크를 합칩니다. 2×2 opening, 3×3 closing 후 세로로 긴 contour를 우선합니다. 두 끝점의 X는 contour bounding box 중심으로 같게 고정됩니다.

### 저장 위치

| 환경변수 | 기본값과 적용 대상 |
|---|---|
| `HC122_VIDEO_DIR` | 사용자 홈의 `Videos`; 두 녹화 프로그램 |
| `HC122_CALIB_DIR` | `surgical_pipeline/config/calibration/face_to_face_20260821`; 수집·yaw 저장·좌표 로더 |

환경변수는 프로그램 실행 전에 설정합니다. 상대 경로는 현재 작업 디렉터리 기준입니다. 수집·yaw 프로그램은 같은 파일명이 있으면 덮어쓰므로, 새로 측정할 때는 별도 디렉터리를 지정해 기존 측정 자료를 보존하세요.

## 좌표계와 변환

| 좌표계 | 원점 | +X / +Y / +Z | 단위 |
|---|---|---|---|
| Pixel `(u,v)` | 영상 왼쪽 위 | u 오른쪽 / v 아래쪽 / Z 없음 | px |
| World | P01 `(0,0,0)` | P01→P05 / P01→P26 / 피부 평면 위쪽 | mm |
| 좌 Base | World `(-124,86,27)` | World −Y / World +X / World +Z | mm |
| 우 Base | World `(254,86,27)` | World +Y / World −X / World +Z | mm |

`robot_base_calibration_face_to_face.json`에 정의된 값입니다. 좌 yaw −90°, 우 yaw +90°, 좌 I2C 버스 1·우 버스 8, 주소는 둘 다 `0x15`입니다.

동차 좌표를 사용합니다.

```text
[a,b,c]^T = H [u,v,1]^T
World = [a/c, b/c, world_z_mm]
Base = Rz(-yaw) (World - base_origin_world_mm)
Base_vector = Rz(-yaw) World_vector
```

점에는 원점 이동과 회전이 모두 적용되고, 방향 벡터에는 회전만 적용됩니다. 피부면(World Z=0)은 베이스 기준 Z −27 mm입니다.

저장된 H:

```text
[-0.848260764540,  0.008093247409,  336.576406623147]
[ 0.004418032996,  0.854165097326, -114.184141138325]
[ 0.000017640143,  0.000054535517,    1.000000000000]
```

`pixel 398 132` 실행 결과:

```text
Pixel: (398.0,132.0)
World: (0.036, 0.319, 0.000) mm
Left Base: (85.681, 124.036, -27.000) mm
Right Base: (-85.681, 253.964, -27.000) mm
```

World `(70,95,0)`은 좌 Base `(-9,194,-27)`, 우 Base `(9,184,-27)`입니다.

## 30점 측정과 재캘리브레이션

1. 카메라·작업 평면·로봇 베이스를 고정하고 640×480 영상에서 격자 30점이 보이게 합니다.
2. 격자는 5열×6행, X 간격 35 mm, Y 간격 38 mm, 전체 140×190 mm입니다. P01이 World 원점이고 한 행에 P01→P05, 다음 행에 P06→P10 순서입니다. 스냅샷 기준으로 World +X는 화면 왼쪽을 향합니다.
3. 아래 명령을 실행하고 P01부터 P30까지 순서대로 클릭합니다. `u`: 한 점 취소, `r`: 전체 초기화, `q`: 종료.
4. 30점이 모두 모이면 `s`로 저장합니다. `cv2.findHomography(..., method=0)`로 전체 점을 사용해 H를 구하고, 각 점의 XY 잔차와 요약값·스냅샷을 저장합니다.
5. 새로 저장되는 요약 키는 `mean_error_mm`, `rmse_mm`, `max_error_mm`입니다. `results/verify_calibration.py`는 2026-08-21 자료의 `*_recomputed` 필드를 기준으로 작성되어 있어 새 측정 파일에는 그대로 쓸 수 없습니다.

```bash
# Linux 예시: 새 측정 파일을 분리해 저장
export HC122_CALIB_DIR="$HOME/hc122-calibration-new"
python3 vision/calibration/collect_pixel_world_calibration_30.py
```

카메라 위치·각도·해상도·크롭·렌즈 설정, 작업 평면의 높이·기울기·격자 위치가 바뀌면 H를 다시 측정합니다. 베이스 위치나 방향이 바뀌면 원점·yaw도 다시 측정합니다. 조명이 바뀌면 HSV·ROI와 검출 상태를 다시 확인합니다. 정확도를 따로 확인하려면 H 적합에 쓴 30점과 별개인 확인점을 측정해야 합니다.

![30점 측정 스냅샷 (2026-08-21)](surgical_pipeline/config/calibration/face_to_face_20260821/pixel_world_calibration_30_snapshot.jpg)

## 측정 자료와 검증 결과

`config/calibration/face_to_face_20260821/`의 JSON 3개와 스냅샷은 2026-08-21 측정 자료입니다. 점 JSON의 `validation_from_terminal`에 있는 독립 중간점 확인 기록은 점 목록·측정값이 함께 남아 있지 않아 아래 수치에 포함하지 않았습니다.

2026-10-15에 저장된 H로 30개 Pixel 점을 float64 동차 좌표로 투영하고, 대응 World XY와의 유클리드 거리를 계산했습니다.

| 항목 | 결과 |
|---|---|
| 점 개수 | 30 |
| 평균 잔차 | 0.778569806 mm |
| RMSE | 0.900923841 mm |
| 최대 잔차 | 1.981541886 mm (P09) |

이 값은 **H 적합에 사용한 같은 30점의 평면 잔차**입니다. 독립 검증 정확도, 3D 복원 정확도, 로봇 칼끝 정확도를 뜻하지 않습니다. 좌표 함수의 OpenCV 입력은 float32이며, float64 계산과 0.00002 mm 허용 오차로 일치를 확인했습니다.

통과한 검사(카메라·모터 없이 실행): Python 9개 문법, 30점·격자·H·저장된 점별 잔차와 재계산 요약값, 좌·우 원점과 회전, 30점 왕복 변환, 장비 없는 CLI 5종, JPG 디코딩·해상도. 상세 결과는 [`results/calib_20261015.json`](results/calib_20261015.json)과 [`results/verification_20261015.md`](results/verification_20261015.md)에 있습니다.

## yaw 측정 스크립트

`common/calibration/calibrate_base_yaw.py`는 Arm_Lib·smbus로 실제 로봇에 연결해 STANDBY, 방향 확인 자세, J1 90→105° 자세로 이동합니다. 사람이 World 방향과 위에서 본 CCW 회전을 확인해 베이스 JSON에 저장합니다. 종료할 때도 STANDBY 이동을 시도합니다. 원점은 `BASES` 상수에 넣어 둔 실측값이며, 이 스크립트가 자동으로 측정하지 않습니다.

실제 모터가 움직이므로 칼날 제거, 주변 공간, 전원 차단 수단, I2C 버스 연결을 확인한 뒤 실행합니다.

```bash
sudo -E python3 common/calibration/calibrate_base_yaw.py left
sudo -E python3 common/calibration/calibrate_base_yaw.py right
# both 인자를 주면 두 팔을 순서대로 측정합니다.
```

## 한계

- HSV는 조명·색·그림자·반사의 영향을 받습니다. 고정 ROI 밖의 절개선이나 폭·세로 비율 조건에 맞지 않는 선은 검출하지 못할 수 있습니다.
- 끝점 X를 같은 중심값으로 맞추기 때문에 기울어진 선의 실제 양 끝점을 그대로 복원하지 않습니다. smoothing은 과거 검출값의 영향을 받습니다.
- 검출에 실패한 프레임에서도 `last_start/last_end`를 비우지 않아, `s`가 이전에 성공한 좌표를 출력할 수 있습니다. 현재 화면과 함께 확인해야 합니다.
- H는 기준 평면에서만 유효합니다. 굴곡진 피부·높이 변화·렌즈 왜곡은 따로 모델링하지 않으며, 자동 보정이나 외삽 범위 검사는 없습니다.
- 카메라 캡처·HSV 검출 성능·녹화 코덱·실제 FPS, yaw 모터 측정, 베이스 실측, 독립 점 검증, 로봇 칼끝 정확도는 검증하지 않았습니다.
