"""/**
 * @author ljm00
 * @email ljm000701@naver.com
 * @create date 2024-05-15 02:54:30
 * @modify date 2024-05-15 02:54:30
 * @desc [description]
 */"""



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
# import torchaudio
# from transformers import AutoConfig
# from transformers import Wav2Vec2FeatureExtractor
# from transformers import Wav2Vec2ForSequenceClassification
import matplotlib.pyplot as plt
import glob

### 음성 신호 특징 추출 LIB ###
import librosa
import numpy as np
import moviepy.editor as mp
from gammatone.gtgram import gtgram

#mfcc 특징 추출
def mfcc(audio_path, sr = 16000,n_mfcc=25): #sr(샘플 rate) - 기본 음성값으로 설정, n_mfcc = 추출 개수
    y, sr = librosa.load(audio_path, sr=sr)
    mfccs = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=n_mfcc)
    return mfccs 

#gtcc 특징 추출
def gtcc(audio_path, sr = 16000, n_gtcc=25):
    y, sr = librosa.load(audio_path, sr=sr)
    # Gammatone filterbank 적용 (대체 방법)
    hop_time = int(sr * 0.01)  # 10ms의 hop 길이 지정
    window_time = int(sr * 0.025)  # 25ms 윈도우 길이 지정
    gt_features = gtgram(y, sr, window_time, hop_time, channels=n_gtcc, f_min=50)
    gtccs = librosa.feature.mfcc(S=librosa.power_to_db(gt_features), sr=sr, n_mfcc=n_gtcc)
    return gtccs

#FAKE 데이터 불러오기 & 데이터 전처리
fake_data_files = glob.glob("./AUDIO/REAL/*.wav")
mfcc_f = []
gtcc_f = []
num=1
for data in fake_data_files:
    fake_data_mfcc = mfcc(data)
    print(fake_data_mfcc.shape) #특징 추출 개수를 보기 위한 임의 코드
    fake_data_gtcc = gtcc(data)
    print(fake_data_gtcc.shape)
    #mfcc,gtcc 각 25개 features 총 50개
    #각 인물마다 7개씩 1-7,8-14 ...
    #biden mfcc는 18751 gtcc는 2 -7
    #linus mfcc 17824 gtcc 2 -14
    #musk  mfcc 18751 gtcc 2 -21
    #obama mfcc 18751 gtcc 2 -28
    #taylor mfcc 18751 gtcc 2 -35
    #trump mfcc 18751 gtcc 2 -42
    result = np.concatenate((fake_data_mfcc,fake_data_gtcc),axis=1) # 0은 fake분류 Feature
    total_data = pd.DataFrame(result)
    total_data = total_data.transpose()
    total_data.to_csv("./data_real_"+str(num)+".csv")
    num= num+1



# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


