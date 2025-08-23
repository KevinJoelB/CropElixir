# Install dependencies if running in Colab
# !pip install xgboost imblearn shap gradio openpyxl

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import gradio as gr

# -------------------------------
# 1️⃣ Train Model
# -------------------------------
def train_and_save_model(file_path):
    # Load data
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)

    data = data.dropna()

    # Features & Target
    X = data[['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type',
              'Nitrogen', 'Potassium', 'Phosphorous', 'pH_Level', 'EC']]
    y = data['Fertilizer']

    # Encode categorical
    label_encoders = {}
    for col in ['Soil_Type', 'Crop_Type']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    y_le = LabelEncoder()
    y = y_le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Handle imbalance
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X_train_resampled, y_train_resampled)

    # Save preprocessing
    joblib.dump(model, 'xgb_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(y_le, 'y_label_encoder.pkl')

    print("✅ Model trained and saved.")


# -------------------------------
# 2️⃣ Prediction Function
# -------------------------------
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
y_le = joblib.load('y_label_encoder.pkl')

expected_columns = ['Temparature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type',
                    'Nitrogen', 'Potassium', 'Phosphorous', 'pH_Level', 'EC']


def predict_fertilizer(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file.name)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file.name)
        else:
            return "❌ Unsupported file format!"
    except Exception as e:
        return f"❌ Error: {e}"

    # Ensure columns
    if not all(col in df.columns for col in expected_columns):
        return "❌ Missing required columns in the uploaded file."

    # Preprocess
    X_new = df[expected_columns].copy()
    for col in ['Soil_Type', 'Crop_Type']:
        X_new[col] = label_encoders[col].transform(X_new[col])

    X_new_scaled = scaler.transform(X_new)
    preds = model.predict(X_new_scaled)
    preds = y_le.inverse_transform(preds)

    df['Predicted_Fertilizer'] = preds

    # Save output
    output_file = "fertilizer_predictions.xlsx"
    df.to_excel(output_file, index=False)

    return output_file


# -------------------------------
# 3️⃣ Gradio App
# -------------------------------
interface = gr.Interface(
    fn=predict_fertilizer,
    inputs=gr.File(label="Upload Soil Report (.csv or .xlsx)"),
    outputs=gr.File(label="Download Predictions"),
    title="🌱 CropElixir",
    description="Upload a soil analysis report to get recommended fertilizer."
)

if __name__ == "__main__":
    interface.launch()
