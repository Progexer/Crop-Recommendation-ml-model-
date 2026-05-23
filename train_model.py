import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
# 1. Load Dataset
data = pd.read_csv("jharkhand.csv")
# ✅ Match columns exactly as in CSV
expected_columns = [
    "Temparature",   # spelling in CSV
    "Humidity",
    "Moisture",
    "Nitrogen",
    "Phosphorous",
    "Potassium",
    "Ph",            # matches CSV
    "Zn",
    "S",
    "Rainfall",
    "Wind Speed",
    "CLOUD_AMT",
    "PS",
    "Crop"
]
# Keep only required columns
data = data[expected_columns]
# 2. Handle Missing Values - FIXED
# Remove rows with missing Crop labels (we need target labels)
data = data.dropna(subset=['Crop'])

# Fill missing values in feature columns with their respective means
feature_columns = [col for col in expected_columns if col != 'Crop']
for col in feature_columns:
    if data[col].isna().any():
        data[col] = data[col].fillna(data[col].mean())

# 3. Split Features & Target
X = data.drop(columns=["Crop"])   # features
y = data["Crop"]                  # target label
# Encode crop labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)
# 5. Train Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
# 6. Save Model & Encoder
joblib.dump(model, "crop_recommendation_model.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("✅ Model training complete! Model, encoder, and feature list saved.")
# 7. Function to Suggest Top Crops
def suggest_crop(input_values, top_n=3):
    """
    input_values: list of sensor + weather values in the same order as dataset columns (except Crop).
                  If some values are missing, just put None.
    top_n: how many crop suggestions to return (default = 3).
    """
    # Load saved model and encoder
    model = joblib.load("crop_recommendation_model.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    # Convert to DataFrame
    input_df = pd.DataFrame([input_values], columns=feature_columns)
    # Fill missing values with mean
    input_df = input_df.fillna(X.mean(numeric_only=True))
    # Predict probabilities
    probabilities = model.predict_proba(input_df)[0]
    # Get top_n crops
    top_indices = probabilities.argsort()[-top_n:][::-1]
    # Map back to crop names
    top_crops = [
        (label_encoder.inverse_transform([i])[0], round(probabilities[i] * 100, 2))
        for i in top_indices
    ]
    return top_crops