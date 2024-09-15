from flask import Flask, request, render_template, jsonify
from app_ml.models import predict_KNN, predict_ML, predict_DL, Get_KNN_Accuracy, Get_NN_Accuracy, GET_LG_Accuracy
import pandas as pd
from app_ml import app

@app.route("/")
@app.route("/index")
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Prepare input data
        age = request.form['age']
        sex = request.form['sex']
        chest_pain_type = request.form['chestPainType']
        resting_bp = request.form['restingBP']
        cholesterol = request.form['cholesterol']
        fasting_bs = request.form['fastingBS']
        resting_ecg = request.form['restingECG']
        max_hr = request.form['maxHR']
        exercise_angina = request.form['exerciseAngina']
        oldpeak = request.form['oldpeak']
        st_slope = request.form['stSlope']
        algorithm = request.form['algorithm']

        input_data = pd.DataFrame({
            'Age': [age],
            'Sex': [sex],
            'ChestPainType': [chest_pain_type],
            'RestingBP': [resting_bp],
            'Cholesterol': [cholesterol],
            'FastingBS': [fasting_bs],
            'RestingECG': [resting_ecg],
            'MaxHR': [max_hr],
            'ExerciseAngina': [exercise_angina],
            'Oldpeak': [oldpeak],
            'ST_Slope': [st_slope]
        })
        acc = 0.0
        if algorithm == "KNN":
            prediction = predict_KNN(input_data)
            acc = Get_KNN_Accuracy()
        elif algorithm == "ML":
            prediction = predict_ML(input_data)
            acc = GET_LG_Accuracy()
        else:
            prediction = predict_DL(input_data)
            acc = Get_NN_Accuracy()
        acc *= 100.0
        if prediction is None:
            return render_template('error.html', message="Model not trained yet")
        pred = ''
        if prediction[0] == 1:
            pred = 'diseased'
        else:
            pred = 'not diseased'
        return render_template('result.html', prediction=pred, accuracy = acc)
    
    except Exception as e:
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
