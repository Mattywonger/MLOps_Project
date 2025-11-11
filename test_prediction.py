"""
Simple script to test model predictions without FastAPI
"""

import joblib
import pandas as pd

# Load the models
print("Loading models...")
model = joblib.load('models/best_model_RandomForest.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
print("✓ Models loaded successfully!\n")

# Sample student data (from the first row of your dataset)
sample_data = {
    "Marital status": 1,
    "Application mode": 17,
    "Application order": 5,
    "Course": 171,
    "Daytime/evening attendance\t": 1,
    "Previous qualification": 1,
    "Previous qualification (grade)": 122.0,
    "Nacionality": 1,
    "Mother's qualification": 19,
    "Father's qualification": 12,
    "Mother's occupation": 5,
    "Father's occupation": 9,
    "Admission grade": 127.3,
    "Displaced": 1,
    "Educational special needs": 0,
    "Debtor": 0,
    "Tuition fees up to date": 1,
    "Gender": 1,
    "Scholarship holder": 0,
    "Age at enrollment": 20,
    "International": 0,
    "Curricular units 1st sem (credited)": 0,
    "Curricular units 1st sem (enrolled)": 5,
    "Curricular units 1st sem (evaluations)": 6,
    "Curricular units 1st sem (approved)": 5,
    "Curricular units 1st sem (grade)": 14.5,
    "Curricular units 1st sem (without evaluations)": 0,
    "Curricular units 2nd sem (credited)": 0,
    "Curricular units 2nd sem (enrolled)": 6,
    "Curricular units 2nd sem (evaluations)": 6,
    "Curricular units 2nd sem (approved)": 6,
    "Curricular units 2nd sem (grade)": 13.67,
    "Curricular units 2nd sem (without evaluations)": 0,
    "Unemployment rate": 10.8,
    "Inflation rate": 1.4,
    "GDP": 1.74
}

print("="*70)
print("STUDENT DATA SAMPLE")
print("="*70)
print(f"Course: {sample_data['Course']}")
print(f"Age: {sample_data['Age at enrollment']}")
print(f"1st Semester Grade: {sample_data['Curricular units 1st sem (grade)']}")
print(f"2nd Semester Grade: {sample_data['Curricular units 2nd sem (grade)']}")
print(f"Scholarship holder: {'Yes' if sample_data['Scholarship holder'] else 'No'}")
print(f"Debtor: {'Yes' if sample_data['Debtor'] else 'No'}")
print()

# Convert to DataFrame
X = pd.DataFrame([sample_data])

# Preprocess
print("Preprocessing data...")
X_preprocessed = preprocessor.transform(X)
print(f"✓ Preprocessed shape: {X_preprocessed.shape}")
print()

# Make prediction
print("Making prediction...")
prediction_encoded = model.predict(X_preprocessed)[0]
prediction_proba = model.predict_proba(X_preprocessed)[0]

# Decode prediction
prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

# Get probabilities for all classes
probabilities = {
    label: float(prob)
    for label, prob in zip(label_encoder.classes_, prediction_proba)
}

# Display results
print("="*70)
print("PREDICTION RESULTS")
print("="*70)
print(f"🎯 Prediction: {prediction_label}")
print(f"📊 Confidence: {max(prediction_proba):.2%}")
print()
print("Probabilities for each class:")
for label, prob in probabilities.items():
    bar = "█" * int(prob * 50)
    print(f"  {label:12s}: {prob:.2%} {bar}")
print("="*70)

# Let's test with a few different scenarios
print("\n\n" + "="*70)
print("TESTING DIFFERENT SCENARIOS")
print("="*70)

scenarios = [
    {
        "name": "High-Performing Student",
        "data": {**sample_data,
                 "Curricular units 1st sem (grade)": 18.0,
                 "Curricular units 2nd sem (grade)": 17.5,
                 "Debtor": 0,
                 "Tuition fees up to date": 1}
    },
    {
        "name": "At-Risk Student",
        "data": {**sample_data,
                 "Curricular units 1st sem (grade)": 10.0,
                 "Curricular units 2nd sem (grade)": 9.5,
                 "Curricular units 1st sem (approved)": 2,
                 "Curricular units 2nd sem (approved)": 2,
                 "Debtor": 1,
                 "Tuition fees up to date": 0}
    },
    {
        "name": "Average Student",
        "data": {**sample_data,
                 "Curricular units 1st sem (grade)": 13.0,
                 "Curricular units 2nd sem (grade)": 13.5}
    }
]

for scenario in scenarios:
    print(f"\n📋 Scenario: {scenario['name']}")
    print("-" * 70)

    X_test = pd.DataFrame([scenario['data']])
    X_test_preprocessed = preprocessor.transform(X_test)

    pred_encoded = model.predict(X_test_preprocessed)[0]
    pred_proba = model.predict_proba(X_test_preprocessed)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    print(f"   Prediction: {pred_label} (Confidence: {max(pred_proba):.1%})")

    # Show probabilities
    probs = {label: float(prob) for label, prob in zip(label_encoder.classes_, pred_proba)}
    for label, prob in probs.items():
        print(f"   - {label}: {prob:.1%}")

print("\n" + "="*70)
print("✓ All predictions completed successfully!")
print("="*70)
