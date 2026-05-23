# Jharkhand Crop Recommendation (SIH)

This project was built for Smart India Hackathon (SIH) as part of our team work. We secured 21st rank out of 200 teams. My contribution focused on data research and training the machine learning model.

The goal is to recommend suitable crops for Jharkhand based on soil and weather features such as moisture, nutrients, rainfall, and wind speed.

## Dataset
The training dataset is stored in `jharkhand.csv` and uses the following columns (exact spellings from the CSV):

- Temparature
- Humidity
- Moisture
- Nitrogen
- Phosphorous
- Potassium
- Ph
- Zn
- S
- Rainfall
- Wind Speed
- CLOUD_AMT
- PS
- Crop (target label)

Note: The column name `Temparature` is intentionally spelled to match the CSV.

## Approach
- Load and filter the expected columns.
- Drop rows where the target label `Crop` is missing.
- Fill missing feature values with column means.
- Encode the crop labels using `LabelEncoder`.
- Train a `RandomForestClassifier` with 200 trees.
- Save the model, label encoder, and feature list as `.pkl` artifacts.

## How to Run
1. Install dependencies:

```bash
pip install pandas scikit-learn joblib
```

2. Train the model:

```bash
python train_model.py
```

This will create:
- `crop_recommendation_model.pkl`
- `label_encoder.pkl`
- `feature_columns.pkl`

## Example Usage
You can call `suggest_crop` after training. The input must follow the exact feature order from the dataset (all columns except `Crop`).

```python
from train_model import suggest_crop

sample_input = [
    28.5,  # Temparature
    65,    # Humidity
    40,    # Moisture
    80,    # Nitrogen
    45,    # Phosphorous
    35,    # Potassium
    6.5,   # Ph
    0.9,   # Zn
    10,    # S
    120,   # Rainfall
    5.5,   # Wind Speed
    50,    # CLOUD_AMT
    1010   # PS
]

print(suggest_crop(sample_input, top_n=3))
```

## Plagiarism Check
No external plagiarism check was run from this workspace. If your institute requires a report, run a plagiarism check using your approved tool and add the report or status here.

## Project Structure
- `train_model.py` - data cleaning, training, and crop suggestion logic
- `jharkhand.csv` - training dataset
- `*.pkl` - trained model and related artifacts

## Notes and Limitations
- Mean imputation is used for missing feature values.
- The model expects input features in the exact column order.
- Feature names must match the CSV header exactly.

## Acknowledgements
Built as part of SIH with the contributions of our team members.
