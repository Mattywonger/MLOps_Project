# End-to-End MLOps Pipeline: Student Dropout Prediction

A comprehensive machine learning operations (MLOps) project demonstrating the complete lifecycle of an ML model from development to deployment, including automated CI/CD pipelines, experiment tracking, and model deployment.

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Project Architecture](#project-architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [MLOps Pipeline](#mlops-pipeline)
- [Model Performance](#model-performance)
- [CI/CD with CML](#cicd-with-cml)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete MLOps pipeline for predicting student dropout risk in higher education institutions. It showcases industry best practices including:

- Automated data preprocessing
- Multiple model training and comparison
- Hyperparameter tuning with MLflow tracking
- Continuous Integration/Continuous Deployment (CI/CD) with CML
- Model versioning and registry
- FastAPI-based model serving
- Docker containerization

## Problem Statement

The goal is to predict the academic risk of students in higher education using a three-category classification:

- **Dropout** - Students who left the program
- **Enrolled** - Students still actively enrolled
- **Graduate** - Students who successfully completed the program

Early prediction of at-risk students enables institutions to provide timely intervention and support.

## Dataset

The dataset originates from a higher education institution and contains information about students enrolled in various undergraduate programs including:

- Agronomy, Design, Education, Nursing, Journalism, Management, Social Service, and Technologies

**Dataset Attributes:**
- **76,518 rows** and **38 columns**
- Enrollment information (demographics, academic path, socio-economic factors)
- Academic performance (first and second semester results)
- Curricular units (enrolled, approved, graded)
- Economic indicators (unemployment rate, inflation rate, GDP)

**Data Source:** [Kaggle - Predict Students' Dropout and Academic Success](https://www.kaggle.com/competitions/playground-series-s3e7)

## Project Architecture

```
┌─────────────────┐
│   Data Layer    │
│   (data.csv)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocessing  │
│  - StandardScaler
│  - OneHotEncoder │
│  - LabelEncoder  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Model Training  │
│  8 ML Models    │
│  - RandomForest │
│  - GradBoosting │
│  - Logistic Reg │
│  - SVC, etc.    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Hyperparameter  │
│    Tuning       │
│  (MLflow)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   CI/CD with    │
│   GitHub Actions│
│      (CML)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Deployment    │
│   (FastAPI +    │
│     Docker)     │
└─────────────────┘
```

## Features

### Data Processing
- ✅ Automated missing value handling
- ✅ Feature scaling with StandardScaler
- ✅ One-hot encoding for categorical features
- ✅ Label encoding for target variable
- ✅ Stratified train-test split

### Model Training
- ✅ 8 different algorithms trained and compared
- ✅ Cross-validation for robust evaluation
- ✅ Multiple performance metrics (accuracy, precision, recall, F1-score)
- ✅ Model serialization with joblib

### Experiment Tracking
- ✅ MLflow integration for hyperparameter tuning
- ✅ Experiment logging and comparison
- ✅ Model registry and versioning

### CI/CD Pipeline
- ✅ Automated testing on every push
- ✅ Visual performance reports with CML
- ✅ Automated model evaluation plots
- ✅ GitHub Actions workflow

### Deployment
- ✅ FastAPI REST API for model serving
- ✅ Docker containerization
- ✅ Health check endpoints
- ✅ Model inference API

## Installation

### Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerized deployment)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/MLOps_Project.git
cd MLOps_Project
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Prepare data**
```bash
# Ensure data.csv is in the project root directory
ls data.csv
```

## Usage

### 1. Interactive Model Development (Jupyter Notebook)

```bash
jupyter notebook model.ipynb
```

The notebook includes:
- Data exploration and visualization
- Preprocessing pipeline setup
- Model training and comparison
- MLflow experiment tracking
- Hyperparameter tuning

### 2. Automated Training Script

```bash
python train_and_evaluate.py
```

This script:
- Loads and preprocesses data
- Trains multiple models
- Generates evaluation metrics
- Creates visualization plots
- Saves the best model

**Outputs:**
- `models/` - Saved models and preprocessors
- `plots/` - Performance visualizations
- `metrics.txt` - Performance metrics table

### 3. MLflow Experiment Tracking

Start the MLflow UI to view experiments:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Access at: `http://localhost:5000`

### 4. CI/CD Pipeline

The pipeline runs automatically on push to GitHub. To trigger manually:

1. Make changes to your code
2. Commit and push:
```bash
git add .
git commit -m "Update model"
git push origin main
```

3. View results in GitHub Actions tab
4. CML report appears as a comment on your commit/PR

## MLOps Pipeline

### Preprocessing Pipeline

```python
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('course', OneHotEncoder(handle_unknown='ignore'), course_column)
    ],
    remainder='passthrough'
)
```

- **StandardScaler**: Normalizes numerical features to mean=0, std=1
- **OneHotEncoder**: Converts 'Course' categorical feature to binary columns
- **LabelEncoder**: Encodes target classes (Dropout, Enrolled, Graduate)

### Models Trained

| Model | Type | Key Parameters |
|-------|------|----------------|
| RandomForest | Ensemble | n_estimators=500, max_depth=10 |
| GradientBoosting | Ensemble | n_estimators=400, learning_rate=0.1 |
| LogisticRegression | Linear | max_iter=1000, multi_class |
| DecisionTree | Tree | max_depth=10 |
| SVC | Kernel | kernel='rbf' |
| AdaBoost | Ensemble | n_estimators=50 |
| KNN | Instance-based | n_neighbors=5 |
| GaussianNB | Probabilistic | default |

### Hyperparameter Tuning

Using `RandomizedSearchCV` with:
- **Scoring**: F1-weighted (optimal for imbalanced multi-class)
- **Cross-validation**: 5-fold CV
- **MLflow tracking**: All experiments logged automatically

```python
hyperparameter_tuning(
    model_name='RandomForest',
    model=RandomForestClassifier(),
    param_dist={...},
    n_iter=10,
    cv=5
)
```

## Model Performance

### Evaluation Metrics

All models are evaluated using:
- **Accuracy**: Overall correctness
- **Precision** (weighted): Quality of positive predictions
- **Recall** (weighted): Coverage of actual positives
- **F1-Score** (weighted): Harmonic mean of precision/recall
- **Cross-Validation Score**: Robustness measure

### Why F1-Weighted?

Given the **imbalanced dataset**:
- Graduate: ~36,000 samples
- Dropout: ~25,000 samples
- Enrolled: ~15,000 samples

**F1-weighted** accounts for class imbalance by weighting each class's F1-score by its frequency, providing a more representative overall metric.

### Performance Comparison

Example results:

| Model | Accuracy | Precision | Recall | F1-Score | CV Score |
|-------|----------|-----------|--------|----------|----------|
| GradientBoosting | 0.8923 | 0.8901 | 0.8923 | 0.8910 | 0.8850 ± 0.0045 |
| RandomForest | 0.8845 | 0.8830 | 0.8845 | 0.8835 | 0.8790 ± 0.0052 |
| LogisticRegression | 0.8456 | 0.8423 | 0.8456 | 0.8438 | 0.8401 ± 0.0067 |

*Note: Actual results may vary based on your data and random seed*

## CI/CD with CML

### GitHub Actions Workflow

**File**: [.github/workflows/cml.yml](.github/workflows/cml.yml)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`

**Steps:**
1. Checkout repository
2. Setup Python 3.11
3. Install dependencies
4. Run training script
5. Generate CML report with visualizations
6. Post report as GitHub comment

### CML Reports Include:
- 📊 Model performance comparison table
- 📈 Bar chart comparing all metrics
- 🔍 Confusion matrix for best model
- 🎯 Feature importance visualization

### Viewing Reports

Reports automatically appear as comments on:
- Commits (when pushed to main/develop)
- Pull requests (for code review)

## Deployment

### FastAPI Application

Create `app.py` for model serving:

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model and preprocessor
model = joblib.load('models/best_model_RandomForest.pkl')
preprocessor = joblib.load('models/preprocessor.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

@app.get("/")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(data: dict):
    # Preprocess input
    df = pd.DataFrame([data])
    X = preprocessor.transform(df)

    # Predict
    prediction = model.predict(X)
    proba = model.predict_proba(X)

    # Decode prediction
    result = label_encoder.inverse_transform(prediction)[0]

    return {
        "prediction": result,
        "confidence": float(max(proba[0]))
    }
```

### Running the API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Access at: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t student-dropout-predictor .
docker run -p 8000:8000 student-dropout-predictor
```

## Project Structure

```
MLOps_Project/
├── .github/
│   └── workflows/
│       └── cml.yml                 # CI/CD workflow
├── data.csv                        # Dataset
├── model.ipynb                     # Interactive development notebook
├── train_and_evaluate.py           # Automated training script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── mlruns/                         # MLflow experiment tracking
├── models/                         # Saved models (generated)
│   ├── best_model_*.pkl
│   ├── preprocessor.pkl
│   └── label_encoder.pkl
├── plots/                          # Visualization outputs (generated)
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   └── feature_importance.png
└── metrics.txt                     # Performance metrics (generated)
```

## Technologies Used

### Core ML Stack
- **Python 3.11** - Programming language
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning algorithms
- **NumPy** - Numerical computing

### MLOps Tools
- **MLflow** - Experiment tracking and model registry
- **DVC** (optional) - Data version control
- **CML** - Continuous Machine Learning reports

### Visualization
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations

### Deployment
- **FastAPI** - REST API framework
- **Uvicorn** - ASGI server
- **Docker** - Containerization

### CI/CD
- **GitHub Actions** - Workflow automation
- **CML (Continuous Machine Learning)** - ML-specific CI/CD

### Model Serialization
- **joblib** - Efficient model persistence

## Key Concepts Explained

### What is StandardScaler?

StandardScaler transforms features to have:
- **Mean = 0**
- **Standard deviation = 1**

**Formula**: `z = (x - μ) / σ`

**Why use it?**
- Puts all features on the same scale
- Prevents features with large ranges from dominating
- Improves model convergence and performance

### What are .pkl Files?

`.pkl` (pickle) files store:
- Trained model weights and parameters
- Model architecture and configuration
- Preprocessing transformers

**Why save models?**
- Reuse without retraining
- Deploy to production
- Version control for model artifacts

### Why Multiple Models?

Training multiple algorithms helps:
- Find the best performer for your data
- Understand model tradeoffs (speed vs accuracy)
- Ensemble methods (combine predictions)
- Robustness testing

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

The CI/CD pipeline will automatically run on your PR and provide performance feedback.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from Kaggle competition: [Predict Students' Dropout and Academic Success](https://www.kaggle.com/competitions/playground-series-s3e7)
- MLOps best practices inspired by industry standards
- CML by Iterative AI for continuous ML workflows

## Contact

For questions or feedback, please open an issue in the GitHub repository.

---

**Built with focus on MLOps best practices** | **End-to-End Machine Learning Pipeline** | **Production-Ready Deployment**
