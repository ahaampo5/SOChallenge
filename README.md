# NSML Small Object Detection Competition

# Task Description

본 대회의 주제는 고해상도 이미지 내의 100x100 이하의 소형 객체 검출 문제입니다. 일반적인 Detection와 다르게 데이터 imbalance와 positive negative imbalance 문제가 동시에 존재하기 때문에 이를 해결하기 위한 실험이 필요합니다.

<img src=https://user-images.githubusercontent.com/60084351/145232746-43cfc489-1980-469a-a8da-a44ff9d9e6af.png width='400'>

<img src=https://user-images.githubusercontent.com/60084351/145232835-6407fb91-329b-4e47-876b-ed89c99c015e.png width='400'>

<img src=https://user-images.githubusercontent.com/60084351/145232567-360cdb0c-2168-43c7-8726-31a613019df0.png width='400'>

### Data
- 학습 데이터 : 17173 장
- 테스트 데이터 : 1000 장


## 실행 방법
```bash
# 명칭이 'CUBOX_SOC'인 데이터셋을 사용해 세션 실행하기
$ nsml run -d CUBOX_SOC
# 메인 파일명이 'main.py'가 아닌 경우('-e' 옵션으로 entry point 지정)
# 예: nsml run -d CUBOX_SOC -e main.py
$ nsml run -d CUBOX_SOC -e [파일명]

# 세션 로그 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml logs -f [세션명]

# 세션 종료 후 모델 목록 및 제출하고자 하는 모델의 checkpoint 번호 확인하기
# 세션명: [유저ID/데이터셋/세션번호] 구조
$ nsml model ls [세션명]

# 모델 제출 전 제출 코드에 문제가 없는지 점검하기('-t' 옵션)
$ nsml submit -t [세션명] [모델_checkpoint_번호]

# 모델 제출하기
# 제출 후 리더보드에서 점수 확인 가능
nsml submit [세션명] [모델_checkpoint_번호]
```

## Contribution
- NSML 환경에 맞춘 mmdetection 실험 환경 구축
  - Docker 실험 환경 세팅
  - MMdet Train, Inference 함수 구현
- EDA
  - Imbalance 문제 개선 (Loss 실험, sampler 실험)
