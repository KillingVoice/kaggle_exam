# kaggle_exam

# Kaggle Deep Voice 데이터 예제 코드
# 기본 CNN 구조를 활용한 Deep Voice Detect 코드

# Raw Data ->
    ./AUDIO ->
        /FAKE -> FAKE_AUDIOES.wav (counts = 56)
        /REAL -> REAL_AUDIOES.wav (counts = 8)

# 데이터 처리 -> mfcc 17824 + gtcc 2개로 총 17826개의 feature

train.py : 모델 학습
Features_Extracter.py : 특징 추출 및 데이터 전처리
model.pth = 모델(ep = 14 , lr = 0.001, Adam)

#Train Loss
![Figure_1](https://github.com/KillingVoice/kaggle_exam/assets/162958984/99c7c105-549d-466d-81a9-0b53e255c5db)
