"""
FastAPI Application for Student Dropout Prediction
Serves the trained ML model via REST API
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import logging
import os
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Student Dropout Prediction API",
    description="API for predicting student academic success (Dropout, Enrolled, or Graduate)",
    version="1.0.0"
)

# Mount static files directory if it exists
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model, preprocessor, and label encoder
try:
    # Find the best model file
    model_files = [f for f in os.listdir('models') if f.startswith('best_model_') and f.endswith('.pkl')]
    if not model_files:
        raise FileNotFoundError("No trained model found in models/ directory")

    model_path = os.path.join('models', model_files[0])
    model = joblib.load(model_path)
    preprocessor = joblib.load('models/preprocessor.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')

    logger.info(f"✓ Loaded model: {model_files[0]}")
    logger.info(f"✓ Model classes: {label_encoder.classes_}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    preprocessor = None
    label_encoder = None


# Define Pydantic model for input validation
class PredictionInput(BaseModel):
    """
    Input schema for student data
    All fields match the columns in the training dataset
    """
    marital_status: int = Field(..., alias="Marital status", ge=1, le=6)
    application_mode: int = Field(..., alias="Application mode", ge=1)
    application_order: int = Field(..., alias="Application order", ge=0, le=9)
    course: int = Field(..., alias="Course", ge=1)
    daytime_evening_attendance: int = Field(..., alias="Daytime/evening attendance\t", ge=0, le=1)
    previous_qualification: int = Field(..., alias="Previous qualification", ge=1)
    previous_qualification_grade: float = Field(..., alias="Previous qualification (grade)", ge=0, le=200)
    nationality: int = Field(..., alias="Nacionality", ge=1)
    mothers_qualification: int = Field(..., alias="Mother's qualification", ge=1)
    fathers_qualification: int = Field(..., alias="Father's qualification", ge=1)
    mothers_occupation: int = Field(..., alias="Mother's occupation", ge=0)
    fathers_occupation: int = Field(..., alias="Father's occupation", ge=0)
    admission_grade: float = Field(..., alias="Admission grade", ge=0, le=200)
    displaced: int = Field(..., alias="Displaced", ge=0, le=1)
    educational_special_needs: int = Field(..., alias="Educational special needs", ge=0, le=1)
    debtor: int = Field(..., alias="Debtor", ge=0, le=1)
    tuition_fees_up_to_date: int = Field(..., alias="Tuition fees up to date", ge=0, le=1)
    gender: int = Field(..., alias="Gender", ge=0, le=1)
    scholarship_holder: int = Field(..., alias="Scholarship holder", ge=0, le=1)
    age_at_enrollment: int = Field(..., alias="Age at enrollment", ge=17, le=70)
    international: int = Field(..., alias="International", ge=0, le=1)
    curricular_units_1st_sem_credited: int = Field(..., alias="Curricular units 1st sem (credited)", ge=0)
    curricular_units_1st_sem_enrolled: int = Field(..., alias="Curricular units 1st sem (enrolled)", ge=0)
    curricular_units_1st_sem_evaluations: int = Field(..., alias="Curricular units 1st sem (evaluations)", ge=0)
    curricular_units_1st_sem_approved: int = Field(..., alias="Curricular units 1st sem (approved)", ge=0)
    curricular_units_1st_sem_grade: float = Field(..., alias="Curricular units 1st sem (grade)", ge=0, le=20)
    curricular_units_1st_sem_without_evaluations: int = Field(..., alias="Curricular units 1st sem (without evaluations)", ge=0)
    curricular_units_2nd_sem_credited: int = Field(..., alias="Curricular units 2nd sem (credited)", ge=0)
    curricular_units_2nd_sem_enrolled: int = Field(..., alias="Curricular units 2nd sem (enrolled)", ge=0)
    curricular_units_2nd_sem_evaluations: int = Field(..., alias="Curricular units 2nd sem (evaluations)", ge=0)
    curricular_units_2nd_sem_approved: int = Field(..., alias="Curricular units 2nd sem (approved)", ge=0)
    curricular_units_2nd_sem_grade: float = Field(..., alias="Curricular units 2nd sem (grade)", ge=0, le=20)
    curricular_units_2nd_sem_without_evaluations: int = Field(..., alias="Curricular units 2nd sem (without evaluations)", ge=0)
    unemployment_rate: float = Field(..., alias="Unemployment rate", ge=0, le=100)
    inflation_rate: float = Field(..., alias="Inflation rate", ge=-10, le=10)
    gdp: float = Field(..., alias="GDP", ge=-10, le=10)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
        }


class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    prediction: str
    confidence: float
    probabilities: dict


# Root endpoint
@app.get("/", response_class=HTMLResponse)
async def root():
    """
    Root endpoint - Returns API information
    """
    html_content = """
    <html>
        <head>
            <title>Student Dropout Prediction API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                .endpoint { background: #f4f4f4; padding: 10px; margin: 10px 0; border-radius: 5px; }
                code { background: #e8e8e8; padding: 2px 5px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>🎓 Student Dropout Prediction API</h1>
            <p>Welcome to the Student Dropout Prediction API. This API predicts whether a student will Dropout, remain Enrolled, or Graduate.</p>

            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <strong>GET /</strong> - This page
            </div>
            <div class="endpoint">
                <strong>GET /health</strong> - Health check endpoint
            </div>
            <div class="endpoint">
                <strong>POST /predict</strong> - Make a prediction (requires student data)
            </div>
            <div class="endpoint">
                <strong>GET /docs</strong> - Interactive API documentation (Swagger UI)
            </div>
            <div class="endpoint">
                <strong>GET /redoc</strong> - Alternative API documentation
            </div>

            <h2>Quick Start:</h2>
            <p>Visit <a href="/docs">/docs</a> for interactive API documentation and to test the prediction endpoint.</p>
        </body>
    </html>
    """
    return html_content


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    Returns the status of the API and whether the model is loaded
    """
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    """
    Prediction endpoint

    Takes student data as input and returns a prediction:
    - Dropout: Student is at risk of dropping out
    - Enrolled: Student will remain enrolled
    - Graduate: Student will successfully graduate

    Returns prediction with confidence scores for each class
    """
    # Check if model is loaded
    if model is None or preprocessor is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure models are trained and saved in models/ directory"
        )

    try:
        # Convert Pydantic model to dictionary with original field names
        data_dict = data.dict(by_alias=True)

        # Create DataFrame with single row
        X = pd.DataFrame([data_dict])

        logger.info(f"Received prediction request with {len(data_dict)} features")

        # Preprocess the data
        X_preprocessed = preprocessor.transform(X)

        # Make prediction
        prediction_encoded = model.predict(X_preprocessed)[0]
        prediction_proba = model.predict_proba(X_preprocessed)[0]

        # Decode prediction
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        # Get confidence (probability of predicted class)
        confidence = float(max(prediction_proba))

        # Create probability dictionary for all classes
        probabilities = {
            label: float(prob)
            for label, prob in zip(label_encoder.classes_, prediction_proba)
        }

        logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.4f})")

        return PredictionOutput(
            prediction=prediction_label,
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """
    Returns information about the loaded model
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "classes": label_encoder.classes_.tolist() if label_encoder else [],
        "n_features": model.n_features_in_ if hasattr(model, 'n_features_in_') else "unknown"
    }


# Run with: uvicorn fast:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
