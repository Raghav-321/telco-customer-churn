from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # our single source of truth for inference

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# Request schema (same fields you collect in the UI)
class CustomerData(BaseModel):
    gender: str
    Partner: str
    PhoneService: str
    InternetService: str
    OnlineSecurity: str
    DeviceProtection: str
    TechSupport: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    tenure: int
    MonthlyCharges: float

@app.post("/predict")
def api_predict(data: CustomerData):
    try:
        out = predict(data.dict())
        return {"prediction": out}
    except Exception as e:
        return {"error": str(e)}

# --- Gradio UI wrappers the same predict() ---
def gradio_interface(
    gender, Partner, PhoneService,
    InternetService, OnlineSecurity, DeviceProtection,
    TechSupport, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges
):
    payload = {
        "gender": gender,
        "Partner": Partner,
        "PhoneService": PhoneService,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
    }
    out = predict(payload)
    return str(out)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender"),
        gr.Dropdown(["Yes", "No"], label="Partner"),
        gr.Dropdown(["Yes", "No"], label="Phone Service"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support"),
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing"),
        gr.Dropdown(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            label="Payment Method"
        ),
        gr.Number(label="Tenure (months)"),
        gr.Number(label="Monthly Charges"),
    ],
    outputs="text",
    title="Telco Churn Predictor",
    description="Fill in the customer details to get a churn prediction.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")
