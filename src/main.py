from src.config import PROJECT_ROOT
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc
import fastapi
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd

class CustomerData(BaseModel):
    tenure: int = Field(..., ge=0, description="Months the customer has stayed")
    TotalCharges: float = Field(..., ge=0)
    TotalServices: int = Field(..., ge=0, description="Count of active services")
    Monthly_Per_Service: float = Field(..., ge=0)

    SeniorCitizen: Literal["Yes", "No"] 
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank Transfer (automatic)", "Credit card (automatic)"]

    model_config = {
        "extra": "ignore",
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "TotalCharges": 800.5,
                "TotalServices": 3,
                "Monthly_Per_Service": 25.0,
                "SeniorCitizen": "No",
                "Dependents": "Yes",
                "PhoneService": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check"
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

    input_df = pd.DataFrame([data.model_dump()])

    for col in input_df.columns:
        if input_df[col].dtype == 'int64':
            input_df[col] = input_df[col].astype('int32')
        elif input_df[col].dtype == 'float64':
            input_df[col] = input_df[col].astype('float32')

    prediction = model.predict(input_df)
    return {"churn_prediction": int(prediction[0])}