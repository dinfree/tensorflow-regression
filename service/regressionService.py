from flask import Flask, render_template

import pandas as pd

import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# 가중치와 옵티마이저를 포함하여 정확히 동일한 모델을 다시 생성합니다
model = keras.models.load_model('models/mpg_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def info():
    # 모델로 부터 예측 - 테스트데이터셋 사용
    new_data = pd.DataFrame([[1.48388718, 1.86598835, 2.23462028, 1.01878165, -2.53089071, -1.60464169, 0.77467638,
                              -0.46514837, -0.49522541]])
    test_predictions = model.predict(new_data).flatten()

    return render_template('info.html', result=test_predictions)
