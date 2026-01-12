from src.config import PROJECT_ROOT
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import fastapi
from src.components.preprocessing import preprocess_data
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)

    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(..., description="0 or 1")
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", 
        "Mailed check", 
        "Bank transfer (automatic)", 
        "Credit card (automatic)"
    ]

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.85,
                "TotalCharges": 29.85
            }
        }
    }

MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
CHAMPION_URI = "models:/Churn Prediction API@champion"
MODEL_NAME = "Churn Prediction API"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = None

app = fastapi.FastAPI()

@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.pyfunc.load_model(CHAMPION_URI)
        print("Champion model loaded on startup")
    except Exception as e:
        print(f"No champion found on startup: {e}")

def champion_model_promotion(run_id: str):
    client = MlflowClient()
    
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)

    global model
    model = mlflow.pyfunc.load_model(CHAMPION_URI)
    print(f"Sucessful! Model version: {mv.version} from run {run_id} is now CHAMPION")
    return mv.version

@app.post("/manage/promote")
def promote_endpoint(run_id: str):
    try:
        version = champion_model_promotion(run_id)
        return {"status": "success", "new_champion_version": version}
    except Exception as e:
        return {"status": "error", "message": str(e)}   
    
@app.post('/predict')
def predict(data: CustomerData):
    if model is None:
        return {"error": "Model not loaded. Use /manage/promote first."}

    full_df = pd.DataFrame([data.model_dump()])
    full_df = preprocess_data(full_df)

    try:
        required_columns = model.metadata.get_input_schema().input_names()
        filtered_df = full_df[required_columns]
    except Exception as e:
        filtered_df = full_df

    for col in filtered_df.select_dtypes(['category']).columns:
        filtered_df[col] = filtered_df[col].astype(str)

    prediction = model.predict(filtered_df)
    return {"churn_prediction": int(prediction[0])}