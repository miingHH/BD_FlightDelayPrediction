#%%
from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd
from collections import defaultdict

app = Flask(__name__, static_folder='static')

# XGBoost 모델 로드
with open("xgboost_model2.pkl", "rb") as f:
    model = pickle.load(f)

def preprocess_distance(data):
    for i in range(51):
        data.loc[data['Distance'].between(i*100, (i+1)*100, 'left'), 'Distance'] = i
    data = data.astype({'Distance': int})
    return data

def to_minutes(x):
    x = int(x)
    x = str(x)
    if len(x) > 2:
        hours, mins = int(x[:-2]), int(x[-2:])
    else:
        hours, mins = 0, int(x[-2:])
    return hours*60+mins

def preprocess_estimated_times(data):
    estimated_times = ['EDT', 'EAT']
    
    for ET in estimated_times:
        cond = ~data[ET].isnull()
        data.loc[cond, ET] = data.loc[cond, ET].apply(lambda x: to_minutes(x))
    
    time_flying = defaultdict(int)
    time_number = defaultdict(int)
    
    cond_arr2 = ~data['EAT'].isnull()
    cond_dep2 = ~data['EDT'].isnull()
    
    for _, row in data.loc[cond_arr2 & cond_dep2, :].iterrows():
        OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
        time_flying[(OAID,DAID)] += (row['EAT'] - row['EDT'])%1440
        time_number[(OAID,DAID)] += 1
    
    for key in time_flying.keys():
        time_flying[key] /= time_number[key]
    
    for index, row in data.loc[data['EDT'].isnull(),].iterrows():
        OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
        data.loc[index,'EDT'] = (data.loc[index]['EAT'] - time_flying[(OAID, DAID)])%1440
    
    for index, row in data.loc[data['EAT'].isnull(),].iterrows():
        OAID, DAID = row['Origin_Airport_ID'], row['Destination_Airport_ID']
        data.loc[index,'EAT'] = (data.loc[index]['EDT'] + time_flying[(OAID, DAID)])%1440
    
    return data

def preprocess_day(data):
    def to_days(x):
        month_to_days = {1: 0, 2: 31, 3: 60, 4: 91, 5: 121, 6: 152, 7: 182, 8: 213, 9: 244, 10: 274, 11: 305, 12: 335}
        return month_to_days[x]

    data['Day'] = data['Month'].apply(to_days) + data['Day_of_Month']
    data = pd.get_dummies(data, columns=['Day'], prefix='Day')
    data.drop(['Day_of_Month', 'Month'], axis=1, inplace=True)
    return data

@app.route("/")
def cover():
    return render_template("cover.html")

@app.route("/index")
def index():
    return render_template("index.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        data = {
            'Origin_Airport_ID': [int(request.form.get('Origin_Airport_ID'))],
            'Destination_Airport_ID': [int(request.form.get('Destination_Airport_ID'))],
            'Distance': [int(request.form.get('Distance'))],
            'Carrier_ID': [int(request.form.get('Carrier_ID'))],
            'Tail_Number': [int(request.form.get('Tail_Number'))],
            'EDT': [int(request.form.get('EDTtime') + request.form.get('EDTmin'))],
            'EAT': [int(request.form.get('EATtime') + request.form.get('EATmin'))],
            'Day_of_Month': [int(request.form.get('Day_of_Month'))],
            'Month': [int(request.form.get('Month'))]
        }
        predict_request = pd.DataFrame(data)
        predict_request.set_index(pd.Index([0]), inplace=True)
        
        predict_request = preprocess_distance(predict_request)
        predict_request = preprocess_estimated_times(predict_request)
        predict_request = preprocess_day(predict_request)

        # 피처 이름 일치시키기
        predict_request.columns = ['Origin_Airport_ID',
                                   'Destination_Airport_ID',
                                   'Distance',
                                   'Carrier_ID(DOT)',
                                   'Tail_Number',
                                   'EDT', 
                                   'EAT',
                                   'Day']

        dtest = xgb.DMatrix(predict_request)
        predictions = model.predict(dtest)
        last_prediction = predictions[-1]  # 마지막 예측값 선택
        prediction = last_prediction[1] * 100  # 두 번째 클래스의 예측값 선택
        
        return render_template('predict.html', prediction=f'{prediction:.2f}%', predict_request=predict_request)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True, threaded=True, use_reloader=False)

# %%
