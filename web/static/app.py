from flask import Flask, render_template, request
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

app = Flask(__name__, static_folder='static')

# XGBoost 모델 로드
with open("xgboost_model.pkl", "rb") as f:
    model = pickle.load(f)

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
            'Month': [float(request.form['Month'])],
            'Day_of_Month': [float(request.form['Day_of_Month'])],
            'Estimated_Departure_Time': [float(request.form['Estimated_Departure_Time'])],
            'Estimated_Arrival_Time': [float(request.form['Estimated_Arrival_Time'])],
            'Cancelled': [float(request.form['Cancelled'])],
            'Diverted': [float(request.form['Diverted'])],
            'Origin_Airport': [float(request.form['Origin_Airport'])],
            'Origin_Airport_ID': [float(request.form['Origin_Airport_ID'])],
            'Origin_State': [float(request.form['Origin_State'])],
            'Destination_Airport': [float(request.form['Destination_Airport'])],
            'Destination_Airport_ID': [float(request.form['Destination_Airport_ID'])],
            'Destination_State': [float(request.form['Destination_State'])],
            'Distance': [float(request.form['Distance'])],
            'Airline': [float(request.form['Airline'])],
            'Carrier_Code': [float(request.form['Carrier_Code'])],
            'Carrier_ID': [float(request.form['Carrier_ID'])],
            'Tail_Number': [float(request.form['Tail_Number'])]
        }
        predict_request = pd.DataFrame(data)
        predict_request.set_index(pd.Index([0]), inplace=True)

        # 피처 이름 일치시키기
        predict_request.columns = ['Month', 'Day_of_Month', 'Estimated_Departure_Time', 'Estimated_Arrival_Time',
                                   'Cancelled', 'Diverted', 'Origin_Airport', 'Origin_Airport_ID', 'Origin_State',
                                   'Destination_Airport', 'Destination_Airport_ID', 'Destination_State', 'Distance',
                                   'Airline', 'Carrier_Code(IATA)', 'Carrier_ID(DOT)', 'Tail_Number']

        dtest = xgb.DMatrix(predict_request)
        predictions = model.predict(dtest)
        last_prediction = predictions[-1]  # 마지막 예측값 선택
        prediction = last_prediction[1] * 100  # 두 번째 클래스의 예측값 선택
        return render_template('predict.html', prediction=f'{prediction:.2f}%')

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=8080, debug=True, threaded=True, use_reloader=False)
