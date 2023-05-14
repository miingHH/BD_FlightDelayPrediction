#!/usr/bin/env python
# coding: utf-8

# # 월간 데이콘 항공편 지연 예측 AI 경진대회
# https://dacon.io/competitions/official/236094/overview/description

# ## 빅데이터처리 기말 프로젝트
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
# 6. EDT, EAT 데이터 전처리 후 동일 과정으로 학습시킨 후 실행
#     - 0.0655098점 달성

# In[5]:


# 경고메세지 끄기
import warnings
warnings.filterwarnings(action='ignore')

# 프로그램 시간 측정
import math
import time
import datetime
time_start =  time.time()
print('시작 시각:', time.strftime('%Y.%m.%d - %H:%M:%S'))


# In[6]:


pip install pandas numpy scikit-learn xgboost bayesian-optimization


# In[7]:


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


# In[8]:


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정


# ## 데이터 불러오기

# In[9]:


def csv_to_parquet(csv_path, save_name):
    df = pd.read_csv(csv_path)
    df.to_parquet(f'./{save_name}.parquet')
    del df
    gc.collect()
    print(save_name, 'Done.')

csv_to_parquet('./train.csv', 'train')
csv_to_parquet('./test.csv', 'test')


# In[10]:


train = pd.read_parquet('./train.parquet')
test = pd.read_parquet('./test.parquet')
sample_submission = pd.read_csv('sample_submission.csv', index_col = 0)


# In[11]:


train.info()


# In[12]:


test.info()


# In[13]:


sample_submission.info()


# ## 데이터 전처리 과정

# #### 1. Delay열을 제외한 열에 존재하는 결측값을 채웁니다.

# ##### 1-1. EDT, EAT 전처리하기

# In[14]:


#EDT, EAT가 모두 결측값인 행 제거
train = train.dropna(subset=['Estimated_Departure_Time', 'Estimated_Arrival_Time'], how='all')


# In[15]:


train.info()


# In[16]:


#EDT, EAT 분으로 환산

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


# In[17]: 


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


# In[18]:


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


# In[20]:


#train 데이터에서 같은 Origin_Airport와 Destination_Airport 사이의 거리의 평균 계산
mean_diff = train.groupby(['Origin_Airport', 'Destination_Airport']).apply(
    lambda group: (group['Estimated_Arrival_Time'] - group['Estimated_Departure_Time']).mean()
).to_dict()

print(mean_diff)


# In[21]:


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


# In[22]:


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


# ##### 1-2.  나머지 질적 변수의 결측값을 최빈값으로 대체

# In[ ]:


NaN_mode_col = ['Origin_State','Destination_State','Airline','Carrier_Code(IATA)','Carrier_ID(DOT)']

for col in NaN_mode_col:
    mode = train[col].mode()[0]
    train[col] = train[col].fillna(mode)
    
    if col in test.columns:
        test[col] = test[col].fillna(mode)

print('Nan_mode_Done.')


# #### 2. LabelEncoder를 이용해, 질적 변수들을 수치화합니다.
# 

# In[24]:


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

# In[25]:


labeled = train.dropna(subset=['Delay'])
unlabeled = train[train['Delay'].isnull()]

print(train.shape)
print(labeled.shape)
print(unlabeled.shape)


# #### 4. Delay 열의 값이 string 형태이기 때문에 이를 0 또는 1로 변환합니다.

# In[26]:


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


# #### 5. 레이블이 있는 데이터의 입력 변수, 출력 변수, 레이블이 없는 데이터의 출력변수를 저장합니다.

# In[27]:


print(len(unlabeled))


# #### 6. 하이퍼 파라미터 튜닝을 수행합니다.
# 베이지안 최적화를 사용하여 XGBoost 모델의 최적 매개변수를 찾습니다.

# In[28]:


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

#exclude = [0.0, 1.0]
#filtered_data = [~np.isin(labeled['Delay_per'], exclude)]
per_mean = np.mean(labeled['Delay_per'])
labeled['Delay_num'] = labeled['Delay_num'].fillna((labeled['Delay_per'] > per_mean).astype(int))
train = labeled
print(train.shape)


# #### 7. 찾아낸 최적의 parameters를 이용해서, XGBoost 학습을 진행하여 Delay_num의 결측치를 채워 넣습니다.
# `labeled` 데이터프레임에 `['Not_Delay_per']`, `['Delay_per']` 두 열을 만들어주는 이유는, `bst.predict(dtest)`의 return 값이 이 형태로 return 되기 때문입니다.

# In[29]:


# # 최적 매개변수 저장
# best_params = xgb_bo.max['params']

# # 최적 매개변수 중 max_depth와 n_estimators를 정수로 변환
# best_params['max_depth'] = int(best_params['max_depth'])
# best_params['n_estimators'] = int(best_params['n_estimators'])
# best_params['objective'] = 'multi:softprob'
# best_params['num_class'] = len(sample_submission.columns)

# # 최적 매개변수로 XGBoost 모델 훈련
# bst = xgb.train(best_params, dtrain)


# 레이블이 없는 데이터에 대한 예측값 저장




# labeled와 unlabeled 데이터 프레임 연결하여 train 데이터 프레임 생성

print("Delay_num.fill_na Done.")
print("Dataset pre-processing Complete.")


# #### 생성된 `unlabeled['Delay_per']` 열의 평균값을 기준으로 `['Delay_num']` 열의 결측값을 채워넣습니다.

# In[30]:


# import matplotlib.pyplot as plt
# import seaborn as sns

# # 데이터 칼럼을 data_column 변수에 할당합니다.
# data_column = unlabeled['Delay_per']

# # 1000개의 구간으로 나눕니다.
# counts, bins = np.histogram(data_column, bins=1000)

# # bar chart를 그립니다.
# plt.bar(bins[:-1], counts, width=np.diff(bins))

# # 데이터들의 평균 값을 계산합니다.
# top_20_percent = np.percentile(data_column, 80)

# # 평균 값의 위치에 세로 선을 그립니다.
# plt.axvline(top_20_percent, color='r', linestyle='dashed', linewidth=2)

# plt.show()


# In[31]:


# # 평균값보다 크면 Delayed, 작으면 Not_Delayed로 판단해서 ['Delay_num']을 채워 넣음
# train['Delay_num'] = train['Delay_num'].fillna((train['Delay_per'] > top_20_percent).astype(int))


# #### 전처리가 완료된 데이터를 csv, parquet 형태로 저장합니다.

# In[32]:


# 전처리 코드를 여러 번 실행 하지 않기 위해서 csv, parquet 형식으로 한번 분리하였음
train_pre = pd.DataFrame(data=train, columns=train.columns, index=train.index)
train_pre.to_csv('train_pre.csv', index=False)

test_pre = pd.DataFrame(data=test, columns=test.columns, index=test.index)
test_pre.to_csv('test_pre.csv', index=False)

csv_to_parquet('./train_pre.csv', 'train_pre')
csv_to_parquet('./test_pre.csv', 'test_pre')


# In[33]:


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
# train의 출력 변수인 train_y의 경우, 결측치를 채워 넣은 Delay_per의 데이터는 기존 데이터와 달리 int형이 아닌 Delay됐을 확률(float)입니다. 따라서, 이를 반올림해 0.5 미만이면 Not_Delayed, 이상이면 Delayed로 간주합니다.

# In[34]:


train_x = train.drop(columns=['ID', 'Delay', 'Delay_num', 'Not_Delay_per', 'Delay_per'])
train_y = train['Delay_num']
test_x = test.drop(columns=['ID'])


# 모든 열이 int, float type인 것을 확인합니다.

# In[35]:


train_x.info()


# In[36]:


train_y.sample(10)


# In[37]:


test_x.info()


# #### 2. 하이퍼 파라미터 튜닝을 수행합니다.¶
# 베이지안 최적화를 사용하여 XGBoost 모델의 최적 매개변수를 찾습니다.

# In[38]:


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

# In[39]:


# 최적 매개변수 저장
best_params = xgb_bo.max['params']

# 최적 매개변수 중 max_depth와 n_estimators를 정수로 변환
best_params['max_depth'] = int(best_params['max_depth'])
best_params['n_estimators'] = int(best_params['n_estimators'])

# 목적 함수와 클래스 개수 설정
best_params['objective'] = 'multi:softprob'
best_params['num_class'] = len(sample_submission.columns)

# 최적 매개변수로 XGBoost 모델 훈련
bst = xgb.train(best_params, dtrain)
# bst = xgb.train(params, dtrain)

# 레이블이 없는 데이터에 대한 예측값 생성
y_pred = bst.predict(dtest)


# #### 4. 최종 제출 파일을 생성합니다.

# In[40]:


# 예측값을 submission 데이터 프레임으로 저장
submission = pd.DataFrame(data=y_pred, columns=sample_submission.columns, index=sample_submission.index)

# submission 데이터 프레임을 CSV 파일로 출력
submission.to_csv('FlightDelayPrediction_submission_pre_tune.csv', index=True)


# 해당 제출 파일으로, 2023년 04월 19일 21시 기준으로 0.635점으로 2등을 달성하였습니다.
# ![image.png](attachment:image.png)

# In[41]:


time_end = time.time()
sec = (time_end - time_start)
print("수행 시간:", datetime.timedelta(seconds=sec))
print('종료 시각:', time.strftime('%Y.%m.%d - %H:%M:%S'))

