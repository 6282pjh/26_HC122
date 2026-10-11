# vision

Logitech C920 기반 절개선 검출과 좌표 변환

- HSV 기반 절개선 Start / End Point 검출
- 30점 Homography Calibration : Pixel → World(mm) → 좌·우 Robot Base
- 출력 형식은 `rl_policy/INTERFACE.md`의 정책 입력(절개선)에 맞춤 (업로드 예정)
