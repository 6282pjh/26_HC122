# haneum_project

# [2025년 한이음 드림업 공모전]

<br/>

<div align="center">

# 🏥 AI 기반 수술 협업 및 의사결정 보조 시스템

### 의료 인력 부족 문제 완화를 위한  
### 실시간 수술 영상 분석 · 수술 단계 예측 · 협업 보조 플랫폼

<br/>

![Project Banner](./images/project_banner.png)

<br/>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![ROS2](https://img.shields.io/badge/ROS2-22314E?style=for-the-badge&logo=ros&logoColor=white)

</div>

<br/>

---

## 1. 프로젝트 개요

### 1-1. 프로젝트 소개

- **프로젝트 명** : AI 기반 수술 협업 및 의사결정 보조 시스템

- **프로젝트 정의** :  
  수술실 내 영상 데이터를 실시간으로 분석하여  
  수술 단계 인식, 위험 상황 감지, 다음 행동 예측, 협업 보조 정보를 제공하는  
  의료진 보조형 AI 플랫폼입니다.

- **핵심 목표** :  
  의료 인력 부족, 수술 교육 격차, 숙련도 편차 문제를 완화하기 위해  
  AI가 수술 상황을 분석하고 의료진에게 참고 정보를 제공하는 시스템을 구현합니다.

<br/>

---

### 1-2. 개발 배경 및 필요성

- 고령화와 의료 수요 증가로 인해 의료 인력 부족 문제가 점점 심화되고 있습니다.

- 수술실은 높은 숙련도와 빠른 판단이 요구되는 공간이지만,  
  신규 의료진이 실제 수술 경험을 충분히 축적하기까지 많은 시간이 필요합니다.

- 수술 중 발생하는 상황은 환자 상태, 수술 단계, 기구 사용, 의료진 판단이 복합적으로 작용하므로  
  실시간 상황 인식과 협업 지원 기술의 필요성이 커지고 있습니다.

- 본 프로젝트는 AI 기반 영상 분석과 예측 모델을 활용하여  
  수술 과정을 보조하고, 의료진 간 협업 효율을 높이며,  
  교육 및 훈련에도 활용 가능한 플랫폼을 목표로 합니다.

<br/>

---

### 1-3. 프로젝트 특장점

- **실시간 수술 영상 분석**
  - 수술 장면을 입력받아 주요 기구, 조직, 수술 단계 등을 인식합니다.

- **수술 단계 예측**
  - 현재 수술 장면을 기반으로 수술 진행 단계를 분류하고 다음 과정을 예측합니다.

- **위험 상황 감지**
  - 출혈, 시야 가림, 비정상적인 기구 움직임 등 위험 상황을 감지합니다.

- **의료진 협업 보조**
  - 수술실 내 보조 인력, 원격 의료진, 교육자에게 현재 상황 정보를 시각화하여 제공합니다.

- **시뮬레이션 기반 학습**
  - 실제 수술 영상 또는 가상 시뮬레이션 데이터를 활용하여 수술 훈련 및 피드백에 활용할 수 있습니다.

- **의사결정 참고 시스템**
  - AI가 판단을 대신하는 것이 아니라, 의료진이 참고할 수 있는 보조 정보를 제공합니다.

<br/>

---

### 1-4. 주요 기능

## ① 실시간 수술 영상 분석 기능

![Surgery Analysis](./images/surgery_analysis.png)

- 수술 카메라 또는 저장된 수술 영상을 입력받습니다.
- 영상 프레임 단위로 수술 기구, 조직, 동작 정보를 분석합니다.
- 분석 결과를 대시보드에 실시간으로 표시합니다.

<br/>

## ② 수술 단계 인식 및 다음 행동 예측 기능

![Phase Prediction](./images/phase_prediction.png)

- 현재 프레임 또는 일정 구간의 영상 데이터를 기반으로 수술 단계를 분류합니다.
- 이전 단계와 현재 단계를 함께 고려하여 다음 수술 흐름을 예측합니다.
- 의료진에게 현재 단계와 예상 진행 방향을 제공합니다.

<br/>

## ③ 위험 상황 감지 및 알림 기능

![Risk Detection](./images/risk_detection.png)

- 출혈, 시야 확보 실패, 특정 기구의 비정상적 접근 등 위험 가능성을 탐지합니다.
- 위험도 점수를 산출하고, 일정 기준 이상이면 대시보드에 경고를 표시합니다.
- 경고는 의료진 판단을 보조하기 위한 참고 정보로 제공됩니다.

<br/>

## ④ 협업 대시보드 기능

![Dashboard](./images/dashboard.png)

- 실시간 수술 화면
- 현재 수술 단계
- AI 분석 결과
- 위험도 점수
- 다음 예상 행동
- 의료진 메모 및 피드백

위 정보를 하나의 화면에서 확인할 수 있도록 구성합니다.

<br/>

---

### 1-5. 기대 효과 및 활용 분야

#### 기대 효과

- 의료 인력 부족 상황에서 보조 인력의 업무 부담 완화
- 수술 중 상황 인식 속도 향상
- 신규 의료진의 수술 교육 및 훈련 효율 향상
- 수술 영상 데이터 기반 피드백 시스템 구축
- 원격 협업 및 수술 교육 플랫폼으로 확장 가능

#### 활용 분야

- 대학병원 및 종합병원 수술실
- 수술 교육용 시뮬레이션 플랫폼
- 의료 AI 연구 및 데이터 분석
- 원격 수술 협업 보조 시스템
- 수술 로봇 및 의료 보조 로봇 연계 시스템

<br/>

---

### 1-6. 기술 스택

| 구분 | 사용 기술 |
|---|---|
| AI / Deep Learning | `PyTorch`, `YOLO`, `CNN`, `LSTM`, `Transformer` |
| Computer Vision | `OpenCV`, `Object Detection`, `Segmentation`, `Pose/Tool Tracking` |
| Backend | `FastAPI`, `Python`, `REST API`, `WebSocket` |
| Frontend | `React`, `JavaScript`, `HTML/CSS` |
| Database | `PostgreSQL`, `SQLite` |
| Simulation | `Unity`, `Isaac Sim`, `NVIDIA Omniverse` |
| Robot / System | `ROS2`, `Raspberry Pi`, `Jetson`, `Camera Module` |
| Communication | `WebRTC`, `Socket`, `HTTP`, `MQTT` |
| Project Management | `Git`, `GitHub`, `Notion`, `Figma` |

<br/>

---

## 2. 팀원 소개

| 이름 | 역할 | 담당 업무 |
|---|---|---|
| 팀장 | Project Manager / AI | 프로젝트 총괄, AI 모델 설계, 수술 단계 예측 |
| 팀원 1 | Backend | FastAPI 서버 구축, 데이터 처리, API 설계 |
| 팀원 2 | Frontend | React 대시보드 제작, UI/UX 구현 |
| 팀원 3 | Computer Vision | 수술 영상 분석, 객체 탐지, 위험 상황 감지 |
| 팀원 4 | Simulation / Robot | 시뮬레이션 환경 구축, ROS2 연동, 시스템 통합 |

<br/>

<div align="center">

| Project Manager | AI Engineer | Backend | Frontend | System |
|---|---|---|---|---|
| ![member](./images/member1.png) | ![member](./images/member2.png) | ![member](./images/member3.png) | ![member](./images/member4.png) | ![member](./images/member5.png) |
| 팀장 | 팀원 1 | 팀원 2 | 팀원 3 | 팀원 4 |

</div>

<br/>

---

## 3. 시스템 구성도

### 3-1. 전체 서비스 흐름도

![Service Flow](./images/service_flow.png)

1. 수술 영상 입력
2. 영상 프레임 전처리
3. AI 모델 기반 수술 장면 분석
4. 수술 단계 및 위험도 예측
5. 서버로 분석 결과 전송
6. 대시보드에서 실시간 시각화
7. 의료진이 참고 정보 확인

<br/>

---

### 3-2. S/W 구성도

![Software Architecture](./images/software_architecture.png)

```text
[Camera / Video Input]
          |
          v
[Preprocessing Module]
          |
          v
[AI Inference Server]
  ├── Object Detection
  ├── Phase Recognition
  ├── Risk Detection
  └── Action Prediction
          |
          v
[FastAPI Backend]
          |
          v
[React Dashboard]
