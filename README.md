# Baseline code
주제1. Small Object Detection 모델 개발

<img src=https://user-images.githubusercontent.com/60084351/145232746-43cfc489-1980-469a-a8da-a44ff9d9e6af.png width='200'>

<img src=https://user-images.githubusercontent.com/60084351/145232835-6407fb91-329b-4e47-876b-ed89c99c015e.png width='200'>

<img src=https://user-images.githubusercontent.com/60084351/145232567-360cdb0c-2168-43c7-8726-31a613019df0.png width='200'>

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
