#!/usr/bin/env python
# coding: utf-8

# # 월간 데이콘 항공편 지연 예측 AI 경진대회
# https://dacon.io/competitions/official/236094/overview/description

# ## 23-1학기 빅데이터처리 기말 프로젝트
# ### 팀원
# - 이시내
# - 유동혁
# - 김명학
# - 박준수
# --- 
# ### 이 후 수정 방법:
# 1. ['Delay_per']의 평균을 내서 상위 50퍼는 Delayed로 간주, ['Delay_num']에 삽입
#     - 0.6821020302점 달성.
# 2. labeled['Delay_num']의 평균을 내서 (약 0.17647), 상위 20%의 ['Delay_per']를 Delayed로 간주
#     - 0.7751130567점 달성
# 3. 전처리 이후 train['Delay_per'](dtype = float)만 이용해서 no tune xgboost 실행
#     - 0.7464493418점 달성
# 4. 전처리 과정 중 unlabeled의 전체를 한번에 채워넣는 것이 아니라, unlabeled를 여러 조각으로 나눠 조금씩 labeled에 합침
#     - 0.6857301445점 달성
# 5. 나눠서 학습하는 과정에서 labeled['Delay_per']를 기준으로 학습하고 전체 평균을 기준으로 train['Delay_num']을 채워넣음
#     - 0.6561362148점 달성
# 6. EDT, EAT 데이터 전처리 후 학습
#     - 이는 `항공운항지연예측 Final-model` 파일에 저장함
#     - public: 0.6337836841점 , private: 0.7080757848점
# 7. 대회 종료 후, [private점수 1위의 코드](https://dacon.io/competitions/official/236094/codeshare/8341)를 참고하여 수정한 모델
#     - 이는 `항공운항지연예측 with 1st.ipynb` 파일에 저장함
#     - public: 0.6234701617점, praivate: 0.7169214489점
# 

# In[1]:


# 경고메세지 끄기
import warnings
warnings.filterwarnings(action='ignore')

# 프로그램 시간 측정
import math
import time
import datetime
time_start =  time.time()
print('시작 시각:', time.strftime('%Y.%m.%d - %H:%M:%S'))


# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import random
import os
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score


# In[3]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정


# ## 데이터 불러오기

# In[4]:


def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('./train.csv', 'train')
csv_to_parquet('./test.csv', 'test')


# In[5]:


train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('sample_submission.csv', index_col = 0)


# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


sample_submission.info()


# ## 데이터 전처리 과정

# #### 1. Delay열을 제외한 열에 존재하는 결측값을 채웁니다.

# ##### 1-1. EDT, EAT의 전처리를 진행합니다. 

# In[9]:


#EDT, EAT가 모두 결측값인 행 제거
train = train.dropna(subset=['Estimated_Departure_Time', 'Estimated_Arrival_Time'], how='all')


# In[10]:


train.info()


# In[11]:


# Estimated Departure Time, Estimated Arrival Time을 분으로 환산
# EDT, EAT는 hhmm의 형태인 4자리 정수이기 때문에, 아래 함수를 통해 분 형태로 바꿉니다.
def convert_time(time):
    if pd.isna(time):
        return None
    else:
        hours = time // 100
        minutes = time % 100
        return hours * 60 + minutes

train['Estimated_Departure_Time'] = train['Estimated_Departure_Time'].apply(convert_time)
train['Estimated_Arrival_Time'] = train['Estimated_Arrival_Time'].apply(convert_time)
test['Estimated_Departure_Time'] = test['Estimated_Departure_Time'].apply(convert_time)
test['Estimated_Arrival_Time'] = test['Estimated_Arrival_Time'].apply(convert_time)


# In[12]:


#test 데이터셋에서 결측값을 채우기 위해 EDT, EAT가 모두 결측값인 행이 제거된 test_filtered 사용
test_filtered = test.dropna(subset=['Estimated_Departure_Time', 'Estimated_Arrival_Time'])

#EDT, EAT가 모두 결측값인 행의 결측값 채우기 위해 test_filtered를 사용
# Origin_Airport와 Destination_Airport로 그룹화합니다.
grouped_df = test_filtered.groupby(['Origin_Airport', 'Destination_Airport'])

# 각 그룹의 Estimated_Arrival_Time과 Estimated_Departure_Time 각각의 평균을 계산
mean_departure = grouped_df['Estimated_Departure_Time'].mean()
mean_arrival = grouped_df['Estimated_Arrival_Time'].mean()

mean_departure_dict = mean_departure.reset_index().set_index(['Origin_Airport', 'Destination_Airport'])['Estimated_Departure_Time'].to_dict()
mean_arrival_dict = mean_arrival.reset_index().set_index(['Origin_Airport', 'Destination_Airport'])['Estimated_Arrival_Time'].to_dict()

print("Mean Departure Dictionary:")
print(mean_departure_dict)
print("\nMean Arrival Dictionary:")
print(mean_arrival_dict)


# In[13]:


def fill_arrival_time_test(row):
    if pd.isna(row['Estimated_Arrival_Time']) and pd.isna(row['Estimated_Departure_Time']):
        key = (row['Origin_Airport'], row['Destination_Airport'])
        if key in mean_arrival_dict:
            return mean_arrival_dict[key]
        else:
            return None
    else:
        return row['Estimated_Arrival_Time']

# 결측치가 있는 행에서 계산된 값을 사용하여 Estimated_Arrival_Time을 채웁니다.
test['Estimated_Arrival_Time'] = test.apply(fill_arrival_time_test, axis=1)


# In[14]:


#train 데이터에서 같은 Origin_Airport와 Destination_Airport 사이의 소요 시간의 평균 계산
mean_diff = train.groupby(['Origin_Airport', 'Destination_Airport']).apply(
    lambda group: (group['Estimated_Arrival_Time'] - group['Estimated_Departure_Time']).mean()
).to_dict()

print(mean_diff)


# In[15]:


#Origin_Airport와 Destination_Airport가 같고 Estimated_Departure_Time만 확인가능할 경우 공항사이 결리는 시간 평균 더하기
def fill_arrival_time(row):
    if pd.isna(row['Estimated_Arrival_Time']):
        key = (row['Origin_Airport'], row['Destination_Airport'])
        if key in mean_diff:
            return (row['Estimated_Departure_Time'] + mean_diff[key])%1440 #시차계산
        else:
            return None
    else:
        return row['Estimated_Arrival_Time']

# 결측치가 있는 행에서 계산된 값을 사용하여 Estimated_Arrival_Time을 채웁니다.
train['Estimated_Arrival_Time'] = train.apply(fill_arrival_time, axis=1)
test['Estimated_Arrival_Time'] = test.apply(fill_arrival_time, axis=1)

print(train)
print(test)


# In[16]:


#Origin_Airport와 Destination_Airport가 같고 Estimated_Arrival_Time만 확인가능할 경우 공항사이 결리는 시간 평균 빼기
def fill_departure_time(row):
    if pd.isna(row['Estimated_Departure_Time']):
        key = (row['Origin_Airport'], row['Destination_Airport'])
        if key in mean_diff:
            return (row['Estimated_Arrival_Time'] - mean_diff[key])%1440 #시차계산
        else:
            return None
    else:
        return row['Estimated_Departure_Time']

# 결측치가 있는 행에서 계산된 값을 사용하여 Estimated_Departure_Time을 채웁니다.
train['Estimated_Departure_Time'] = train.apply(fill_departure_time, axis=1)
test['Estimated_Departure_Time'] = test.apply(fill_departure_time, axis=1)

print(train)
print(test)


# ##### 1-2.  나머지 질적 변수의 결측값을 최빈값으로 대체합니다.

# In[17]:


# 질적 변수의 결측값을 최빈값으로 대체합니다.
NaN_mode_col = ['Origin_State','Destination_State','Airline','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_mode_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)

print('Nan_mode_Done.')


# #### 2. LabelEncoder를 이용해, 질적 변수들을 수치화합니다.
# 

# In[18]:


qual_col = ['Origin_Airport', 'Origin_State', 'Destination_Airport', 'Destination_State', 'Airline', 'Carrier_Code(IATA)', 'Tail_Number']

for i in qual_col:
    le = LabelEncoder()
    le=le.fit(train[i])
    train[i]=le.transform(train[i])
    
    for label in np.unique(test[i]):
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test[i]=le.transform(test[i])
print('qual_col Done.')


# #### 3. Delay 열에 결측값이 없는 행들과 있는 행들을 분리합니다.

# In[19]:


labeled = train.dropna(subset=['Delay'])
unlabeled = train[train['Delay'].isnull()]

print(train.shape)
print(labeled.shape)
print(unlabeled.shape)


# #### 4. Delay 열의 값이 string 형태이기 때문에 이를 0 또는 1로 변환합니다.

# In[20]:


column_number = {}
for i, column in enumerate(sample_submission.columns):
    column_number[column] = i
# ==> column_number: {'Not_Delayed': 0, 'Delayed': 1}
    
def to_number(x, dic):
    return dic[x]

labeled.loc[:, 'Delay_num'] = labeled['Delay'].apply(lambda x: to_number(x, column_number))
# Delay 열의 값에 따라서, Not_Delayed면 0, Delayed면 1이 Delay_num 열에 저장됨
print('Delay_num Done.')

# 위 전체 과정은 아래 코드와 같은 의미임
# labeled['Delay_num'] = labeled['Delay'].apply(lambda x: 1 if x == 'Delayed' else 0)


# #### 5. 레이블이 없는 데이터의 `['Delay_num']`열을 채우기 위해 아래 과정을 거칩니다.
# 1. 먼저 `unlabeled`를 여러 구간으로 쪼갭니다. 여기서는 20개의 구간으로 쪼갰습니다.
# 2. 해당 항공편이 지연될 확률을 의미하는 `['Delay_per']`열을 만듭니다.
# 3. `labeled`를 XGBoost 모델에 학습시켜, `unlabeled`의 하나의 구간의 `['Delay_per']`열을 채워넣습니다.
# 4. `['Delay_per']`열이 채워진 `unlabeled` 구간을 `labeled`에 합칩니다.
# 5. 3, 4의 과정을 `unlabeled`데이터에 결측치가 없을 때까지 반복합니다.
# 6. `['Delay_per']`열의 평균값을 구해, 이 평균값보다 지연될 확률이 낮은 항공편은 <i>Not_Delayed</i>로 간주하고, 높은 항공편은 <i>Delayed</i>로 간주해 `['Delay_num']`열의 결측치를 채워 넣습니다.

# In[21]:


labeled['Not_Delay_per'] = labeled['Delay_num'].apply(lambda x: 1 if x == 0 else 0).astype('float64')
labeled['Delay_per'] = labeled['Delay_num'].astype('float64')

num_of_gugan = 20
for i in range(num_of_gugan):
    gugan_size = int(round(len(unlabeled)/num_of_gugan, 0))
    L = gugan_size * i
    R = gugan_size * (i+1)
    
    small_unlabeled = unlabeled.iloc[L:R]
    
    # 레이블이 있는 데이터의 입력 변수와 출력 변수를 각각 labeled_x와 labeled_y로 저장
    labeled_x = labeled.drop(columns=['ID', 'Delay', 'Delay_num', 'Not_Delay_per', 'Delay_per'])
    labeled_y = labeled['Delay_per']

    # 레이블이 없는 데이터의 입력 변수를 unlabeled_x로 저장
    unlabeled_x = small_unlabeled.drop(columns=['ID', 'Delay'])

    # XGBoost 모델의 입력 데이터 형식인 DMatrix로 변환
    dtrain = xgb.DMatrix(labeled_x, label=labeled_y)
    dtest = xgb.DMatrix(unlabeled_x)

    # XGBoost 모델의 목적 함수와 클래스 개수 설정
    params = {
        'objective': 'multi:softprob',
        'num_class': len(sample_submission.columns)
    }

    bst = xgb.train(params, dtrain)
    small_unlabeled[['Not_Delay_per','Delay_per']] = bst.predict(dtest)
    labeled = pd.concat([labeled, small_unlabeled])
    print(f'{i}: 구간 {L} ~ {R} 완료, labeled size: {len(labeled)}')

per_mean = np.mean(labeled['Delay_per'])
labeled['Delay_num'] = labeled['Delay_num'].fillna((labeled['Delay_per'] > per_mean).astype(int))
train = labeled
print(train.shape)


# #### 전처리가 완료된 데이터를 csv, parquet 형태로 저장합니다.

# In[22]:


# 전처리 코드를 여러 번 실행 하지 않기 위해서 csv, parquet 형식으로 한번 분리하였음
train_pre = pd.DataFrame(data=train, columns=train.columns, index=train.index)
train_pre.to_csv('train_pre.csv', index=False)

test_pre = pd.DataFrame(data=test, columns=test.columns, index=test.index)
test_pre.to_csv('test_pre.csv', index=False)

csv_to_parquet('./train_pre.csv', 'train_pre')
csv_to_parquet('./test_pre.csv', 'test_pre')


# In[23]:


# 한 번 이상 전처리 코드를 실행했다면, 이 셀부터 실행하면 됨
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import random
import os
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

train = pd.read_parquet('./train_pre.parquet')
test = pd.read_parquet('./test_pre.parquet')
sample_submission = pd.read_csv('sample_submission.csv', index_col = 0)


# ## 모델 훈련 과정

# #### 1. train 데이터의 입력 변수, 출력 변수,  test 데이터의 출력변수를 저장합니다.

# In[24]:


train_x = train.drop(columns=['ID', 'Delay', 'Delay_num', 'Not_Delay_per', 'Delay_per'])
train_y = train['Delay_num'].astype('int64')
test_x = test.drop(columns=['ID'])


# 모든 입출력 변수의 데이터 타입이 int, float인 것을 확인합니다.

# In[25]:


train_x.info()


# In[26]:


train_y.sample(10)


# In[27]:


test_x.info()


# #### 2. 하이퍼 파라미터 튜닝을 수행합니다.¶
# 베이지안 최적화를 사용하여 XGBoost 모델의 최적 매개변수를 찾습니다.

# In[28]:


# XGBoost 모델의 입력 데이터 형식인 DMatrix로 변환
dtrain = xgb.DMatrix(train_x, label=train_y)
dtest = xgb.DMatrix(test_x)

# XGBoost 모델의 목적 함수와 클래스 개수 설정
params = {
    'objective': 'multi:softprob',
    'num_class': len(sample_submission.columns)
}

# XGBoost 모델의 교차 검증 함수 정의 (전처리 과정에 있는 함수 정의와 동일함)
def xgb_cv(max_depth, learning_rate, n_estimators):
    params = {
        'max_depth': int(max_depth),
        'learning_rate': learning_rate,
        'n_estimators': int(n_estimators)
    }
    xgb_clf = xgb.XGBClassifier(**params)
    cv_result = cross_val_score(xgb_clf, train_x, train_y, cv=3)
    return cv_result.mean()

# 베이지안 최적화 객체 생성
xgb_bo = BayesianOptimization(xgb_cv, {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators': (100, 1000)
})

# 베이지안 최적화 수행
xgb_bo.maximize()
print("hyper-parameter tuning for model Done.")


# #### 3. 찾아낸 최적의 parameters를 이용해서, XGBoost 학습을 진행하여 모델을 훈련시킵니다.

# In[29]:


# 최적 매개변수 저장
best_params = xgb_bo.max['params']

# 최적 매개변수 중 max_depth와 n_estimators를 정수로 변환
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# 목적 함수와 클래스 개수 설정
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = len(sample_submission.columns)

# 최적 매개변수로 XGBoost 모델 훈련 및 모델 내보내기
bst = xgb.train(best_params, dtrain)
bst.save_model('model.bst')

# 레이블이 없는 데이터에 대한 예측값 생성
y_pred = bst.predict(dtest)


# #### 4. 최종 제출 파일을 생성합니다.

# In[30]:


# 예측값을 submission 데이터 프레임으로 저장
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)

# submission 데이터 프레임을 CSV 파일로 출력
submission.to_csv('FlightDelayPrediction_submission_pre_tune.csv', index=True)


# 해당 제출 파일으로, 2023년 04월 19일 21시 기준으로 0.635점으로 2등을 달성하였습니다.
# ![image.png](attachment:image.png)

# In[31]:


time_end = time.time()
sec = (time_end - time_start)
print("수행 시간:", datetime.timedelta(seconds=sec))
print('종료 시각:', time.strftime('%Y.%m.%d - %H:%M:%S'))

