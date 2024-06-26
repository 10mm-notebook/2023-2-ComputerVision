## HAM10000 데이터셋을 이용한 VGGNet 기반 피부 병변 탐지 모델 개발 프로젝트

### 프로젝트 목적
피부암의 조기 진단을 돕기 위해 컴퓨터 비전 기술을 활용하여 피부 병변을 자동으로 분류하는 모델 개발

### 프로젝트 기간
2023.10~2023.12

### 프로젝트 팀
카피바라 팀 (이승열, 김승영, 민지수, 김민준)

### 사용 데이터
HAM10000 데이터셋 (5.9GB, 10,015장, 7클래스)

### 데이터 전처리
- Adasyn 오버샘플링 기법으로 클래스 간 데이터 불균형 해소
- 픽셀값 정규화, 원-핫 인코딩 등의 전처리 과정 수행

### 모델 구조
- 다양한 모델(ResNet, VGGNet 등) 성능 비교 후 VGGNet 선정 (사용 VGGNet 베이스라인 : https://www.kaggle.com/code/arthurgomesbubolz/skin-cancer-classifier-with-vggnet-inspired-cnn )
- 11개 층으로 구성된 수정된 VGGNet 아키텍처 사용
- He 초기화, Swish 활성화 함수, 0.3 드롭아웃률, 배치정규화 등 적용
- 입력 이미지 크기 56x56으로 리사이징

### 실험 과정
- 모델 구성요소(활성화 함수, 드롭아웃, 배치정규화 등) 변경하며 정확도 측정
- 하이퍼파라미터(옵티마이저, 학습률, 에폭 등) 튜닝

### 결과 (베이스라인 -> 최종 모델)
- Accuracy : 0.8522 -> 0.9785
- Precision : 0.8705 -> 0.9799
- Recall : 0.8405 -> 0.9775  
- F1 Score : 0.8552 -> 0.9787

## 최종 모델 파이프라인

![pipeline](https://github.com/10mm-notebook/2023-2-ComputerVision/assets/141313910/1d7195b8-2da0-4b42-96a2-204536c3903b)
