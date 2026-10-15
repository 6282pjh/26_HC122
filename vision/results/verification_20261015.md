# 비전·캘리브레이션 파일 검증 — 2026-10-15

`surgical_pipeline/`의 Python 코드 8개와 2026-08-21 캘리브레이션 측정 자료(JSON 3개, JPG 1개)를 카메라·모터 없이 검사한 기록입니다. 측정 JSON·JPG는 수정하지 않았고, 원본 SHA-256을 `calib_20261015.json`에 기록했습니다.

## 재현

저장소 루트에서 실행:

```bash
python -B vision/results/verify_calibration.py
```

검증 환경은 Windows, Python 3.11.9, NumPy 2.4.1, OpenCV 4.13.0이며, 원래 Jetson 실행 환경과는 다릅니다. 이 스크립트는 실행할 때마다 `calib_20261015.json`을 다시 쓰므로(검증 환경 값 포함), 기록을 보존하려면 실행 후 변경분을 커밋하지 않습니다. `HC122_CALIB_DIR` 값과 관계없이 `config/calibration/face_to_face_20260821/` 자료를 검사합니다.

## 확인 결과

| 검사 | 결과 |
|---|---|
| Python 문법 | 코드 8개 + 검증 스크립트 1개, py_compile 통과 |
| 대응점 | P01–P30 30개, 중복 Pixel 없음, 640×480 범위, 5×6 World 격자 확인 |
| H | 3×3 유한 행렬, rank 3, 분모 유효 |
| 잔차 | 저장된 H로 float64 투영 후 World XY와 유클리드 거리 계산 |
| 저장 오차 | 점별 `reprojection_error_mm`와 H JSON의 `_recomputed` 요약 필드가 1e-9 mm 이내 일치 |
| 평균 / RMSE / 최대 | 0.778569806 / 0.900923841 / 1.981541886 mm, 최대 P09 |
| 베이스 | 좌 `(-124,86,27)`, yaw −90° / 우 `(254,86,27)`, yaw +90° |
| 좌표 변환 | 원점→0, 회전 직교성·행렬식, 30점 World↔Base 왕복, 방향 벡터, 중심점 확인 |
| CLI | Pixel→World, verify, Pixel→양팔, World→양팔, 선분 5종 통과 |
| 장비 차단 | CLI 검사 중 VideoCapture·VideoWriter·namedWindow 호출 시 실패하도록 차단 |
| 스냅샷 | JPG 디코딩, 640×480 확인 |
| 자료 크기 | 모든 파일 5 MB 이하 |

위 잔차는 H를 구할 때 사용한 같은 30점의 적합 잔차이며, 독립 검증 정확도나 로봇 칼끝 정확도가 아닙니다. 점 JSON의 `validation_from_terminal`에 있는 중간점 확인 기록은 점 목록·측정값이 함께 남아 있지 않아 수치 근거에서 제외했습니다.

## 검증하지 않은 범위

카메라와 모터는 실행하지 않았습니다. `calibrate_base_yaw.py`와 최상위 코드에서 바로 카메라를 여는 `webcam_record.py`는 import하지 않았습니다. 실제 HSV 검출 성능, 카메라·녹화 동작, 모터 yaw 측정, 실측 베이스 원점, 독립 측정점, 로봇 칼끝 정확도는 검증하지 않았습니다.
