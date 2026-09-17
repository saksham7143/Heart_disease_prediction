# CardioShield AI - Heart Disease Prediction Web Application

An end-to-end, medical AI web application powered by **Flask** (backend) and a modern **Glassmorphism Web Dashboard** (frontend), utilizing a trained **K-Nearest Neighbors (KNN)** classification model and **StandardScaler** to assess real-time heart disease risk.

---

## 🌟 Key Features

- **Intuitive Clinical Input**: Patients or healthcare providers enter 11 clear medical parameters rather than raw encoded numbers.
- **Automated ML Pipeline**: The backend handles zero-imputation (`Cholesterol` & `RestingBP`), categorical one-hot encoding (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`), initial numerical scaling, and transformation via `scaler.pkl`.
- **Pre-trained ML Model**: Uses the existing `heartprediction.pkl` model without retraining or modifying model files.
- **Real-time Visual Analytics**: Features an interactive SVG risk gauge, confidence percentage split, and primary clinical risk factor callouts.
- **Quick Test Profiles**: Built-in sample presets (*Low Risk*, *High Risk*, *Moderate Risk*) for one-click demonstration.

---

## 📁 Repository Structure

```text
Heart_disease_prediction/
├── app.py                         # Flask web application & REST API
├── heartprediction.pkl            # Pre-trained KNN model file
├── scaler.pkl                     # Pre-trained StandardScaler file
├── heart.csv                      # Source clinical dataset (used for baseline parameters)
├── Heart_disease_prediction.ipynb # Data analysis & model training notebook
├── requirements.txt               # Required Python packages
├── README.md                      # Documentation & instructions
├── static/
│   └── style.css                  # Custom Glassmorphism UI stylesheet
└── templates/
    └── index.html                 # Main web application dashboard template
```

---

## 🚀 Installation & Setup Guide

### 1. Prerequisites
Ensure you have **Python 3.8 or higher** installed on your system.

### 2. Clone / Navigate to Project Directory
```bash
cd "Heart_disease_prediction"
```

### 3. Install Dependencies
Install all required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 4. Run the Flask Web Application
Start the application server by running `app.py`:
```bash
python app.py
```

### 5. Access the Web Application
Open your web browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🩺 Patient Parameter Input Mapping

The web interface accepts 11 intuitive parameters and converts them into the 15 features expected by `scaler.pkl` & `heartprediction.pkl`:

| User Input Field | Description | Encoding / Preprocessing Logic |
| :--- | :--- | :--- |
| **Age** | Patient age in years | Scaled using numeric StandardScaler |
| **Biological Sex** | Male / Female | `Sex_M` (1 = Male, 0 = Female) |
| **Chest Pain Type** | ASY, ATA, NAP, TA | `ChestPainType_ATA`, `ChestPainType_NAP`, `ChestPainType_TA` |
| **Resting BP** | Blood pressure in mm Hg | 0s imputed with ~132.54 mm Hg; scaled |
| **Cholesterol** | Serum cholesterol in mg/dl | 0s imputed with ~244.63 mg/dl; scaled |
| **Fasting Blood Sugar** | &le; 120 vs &gt; 120 mg/dl | `FastingBS` (1 = &gt;120 mg/dl, 0 = &le;120 mg/dl) |
| **Resting ECG** | Normal, ST, LVH | `RestingECG_Normal`, `RestingECG_ST` |
| **Max Heart Rate** | Peak HR in bpm | Scaled using numeric StandardScaler |
| **Exercise Angina** | Induced by exercise (Y/N) | `ExerciseAngina_Y` (1 = Yes, 0 = No) |
| **ST Depression (Oldpeak)**| Exercise relative to rest | Scaled using numeric StandardScaler |
| **ST Slope** | Peak exercise ST slope | `ST_Slope_Flat`, `ST_Slope_Up` |

---

## 📡 API Reference

### `POST /predict`
Submits patient medical parameters to receive heart disease prediction and probability scores.

#### Request Header:
`Content-Type: application/json`

#### Request Payload Sample:
```json
{
  "age": 58,
  "sex": "M",
  "chest_pain": "ASY",
  "resting_bp": 160,
  "cholesterol": 286,
  "fasting_bs": 1,
  "resting_ecg": "LVH",
  "max_hr": 115,
  "exercise_angina": "Y",
  "oldpeak": 2.6,
  "st_slope": "Flat"
}
```

#### Response Sample:
```json
{
  "success": true,
  "prediction": 1,
  "risk_label": "High Risk of Heart Disease",
  "risk_percentage": 100.0,
  "confidence": 100.0,
  "probabilities": {
    "low_risk": 0.0,
    "high_risk": 100.0
  },
  "risk_factors": [
    "Age 55 or older",
    "Asymptomatic Chest Pain presentation",
    "Exercise-Induced Angina positive",
    "Abnormal ST Slope (Flat)",
    "Elevated ST depression (2.6 mm)",
    "Fasting Blood Sugar > 120 mg/dl",
    "Elevated Resting Blood Pressure (160 mm Hg)",
    "High Serum Cholesterol (286 mg/dl)"
  ]
}
```

---

## 🛡️ License & Disclaimer

This application is designed for educational and clinical decision-support research. It uses machine learning predictions and should be interpreted in conjunction with professional medical evaluation.