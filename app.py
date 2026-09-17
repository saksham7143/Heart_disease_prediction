import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Base directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'heart.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'heartprediction.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

# Global variables for model, scaler, and initial data scaler
model = None
scaler = None
scaler_num = None
ch_mean = 244.63
resting_bp_mean = 132.54

def initialize_pipeline():
    global model, scaler, scaler_num, ch_mean, resting_bp_mean
    
    # 1. Load ML Artifacts
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Missing heartprediction.pkl or scaler.pkl model files!")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # 2. Re-create the initial numeric scaler using heart.csv exactly as in the notebook
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        # Handle zero replacements for Cholesterol and RestingBP as done in notebook
        ch_mean = df.loc[df['Cholesterol'] != 0, 'Cholesterol'].mean()
        resting_bp_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean().round(2)
        
        df_clean = df.copy()
        df_clean['Cholesterol'] = df_clean['Cholesterol'].replace(0, ch_mean)
        df_clean['RestingBP'] = df_clean['RestingBP'].replace(0, resting_bp_mean)
        
        num_col = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        from sklearn.preprocessing import StandardScaler
        scaler_num = StandardScaler()
        scaler_num.fit(df_clean[num_col])
    else:
        raise FileNotFoundError("heart.csv is required to compute initial dataset scale parameters.")

# Initialize pipeline on startup
initialize_pipeline()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Support both JSON payload and Form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
            
        # Parse and sanitize user parameters
        age = float(data.get('age', 50))
        sex = str(data.get('sex', 'M')).upper()
        chest_pain = str(data.get('chest_pain', 'ASY')).upper()
        resting_bp = float(data.get('resting_bp', 130))
        cholesterol = float(data.get('cholesterol', 220))
        fasting_bs = int(data.get('fasting_bs', 0))
        resting_ecg = str(data.get('resting_ecg', 'Normal'))
        max_hr = float(data.get('max_hr', 150))
        exercise_angina = str(data.get('exercise_angina', 'N')).upper()
        oldpeak = float(data.get('oldpeak', 0.0))
        st_slope = str(data.get('st_slope', 'Up'))

        # Handle zeros according to model training logic
        if resting_bp == 0:
            resting_bp = float(resting_bp_mean)
        if cholesterol == 0:
            cholesterol = float(ch_mean)

        # 1. One-hot encoding logic matching notebook drop_first=True
        # Sex: M vs F -> Sex_M
        sex_M = 1 if sex == 'M' else 0
        
        # ChestPainType: ATA, NAP, TA (Baseline: ASY)
        chest_pain_ATA = 1 if chest_pain == 'ATA' else 0
        chest_pain_NAP = 1 if chest_pain == 'NAP' else 0
        chest_pain_TA = 1 if chest_pain == 'TA' else 0
        
        # RestingECG: Normal, ST (Baseline: LVH)
        resting_ecg_Normal = 1 if resting_ecg == 'Normal' else 0
        resting_ecg_ST = 1 if resting_ecg == 'ST' else 0
        
        # ExerciseAngina: Y vs N -> ExerciseAngina_Y
        exercise_angina_Y = 1 if exercise_angina == 'Y' else 0
        
        # ST_Slope: Flat, Up (Baseline: Down)
        st_slope_Flat = 1 if st_slope == 'Flat' else 0
        st_slope_Up = 1 if st_slope == 'Up' else 0

        # 2. Build pandas DataFrame with raw inputs
        input_df = pd.DataFrame([{
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_M': sex_M,
            'ChestPainType_ATA': chest_pain_ATA,
            'ChestPainType_NAP': chest_pain_NAP,
            'ChestPainType_TA': chest_pain_TA,
            'RestingECG_Normal': resting_ecg_Normal,
            'RestingECG_ST': resting_ecg_ST,
            'ExerciseAngina_Y': exercise_angina_Y,
            'ST_Slope_Flat': st_slope_Flat,
            'ST_Slope_Up': st_slope_Up
        }])

        # 3. Apply first-stage numerical scaling (num_col) as done in notebook
        num_col = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        input_df[num_col] = scaler_num.transform(input_df[num_col])

        # 4. Ensure exact 15 feature ordering expected by saved scaler & KNN model
        feature_order = [
            'Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
            'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
            'RestingECG_Normal', 'RestingECG_ST', 'ExerciseAngina_Y',
            'ST_Slope_Flat', 'ST_Slope_Up'
        ]
        X_input = input_df[feature_order]

        # 5. Transform using saved scaler.pkl
        X_scaled = scaler.transform(X_input)

        # 6. Predict using saved heartprediction.pkl
        prediction = int(model.predict(X_scaled)[0])
        probabilities = model.predict_proba(X_scaled)[0].tolist()
        
        risk_percentage = round(probabilities[1] * 100, 1)

        # Identify key clinical risk factors for informative UI feedback
        risk_factors = []
        if age >= 55:
            risk_factors.append("Age 55 or older")
        if chest_pain == 'ASY':
            risk_factors.append("Asymptomatic Chest Pain presentation")
        if exercise_angina == 'Y':
            risk_factors.append("Exercise-Induced Angina positive")
        if st_slope in ['Flat', 'Down']:
            risk_factors.append(f"Abnormal ST Slope ({st_slope})")
        if oldpeak >= 1.0:
            risk_factors.append(f"Elevated ST depression ({oldpeak} mm)")
        if fasting_bs == 1:
            risk_factors.append("Fasting Blood Sugar > 120 mg/dl")
        if resting_bp >= 140:
            risk_factors.append(f"Elevated Resting Blood Pressure ({int(resting_bp)} mm Hg)")
        if cholesterol >= 240:
            risk_factors.append(f"High Serum Cholesterol ({int(cholesterol)} mg/dl)")

        return jsonify({
            'success': True,
            'prediction': prediction,
            'risk_label': 'High Risk of Heart Disease' if prediction == 1 else 'Low Risk of Heart Disease',
            'risk_percentage': risk_percentage,
            'confidence': round(max(probabilities) * 100, 1),
            'probabilities': {
                'low_risk': round(probabilities[0] * 100, 1),
                'high_risk': round(probabilities[1] * 100, 1)
            },
            'risk_factors': risk_factors,
            'inputs_processed': {
                'Age': age,
                'Sex': 'Male' if sex == 'M' else 'Female',
                'ChestPainType': chest_pain,
                'RestingBP': resting_bp,
                'Cholesterol': cholesterol,
                'FastingBS': 'Yes (>120 mg/dl)' if fasting_bs == 1 else 'No (<=120 mg/dl)',
                'RestingECG': resting_ecg,
                'MaxHR': max_hr,
                'ExerciseAngina': 'Yes' if exercise_angina == 'Y' else 'No',
                'Oldpeak': oldpeak,
                'ST_Slope': st_slope
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
