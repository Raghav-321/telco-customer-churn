"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================

This application provides a complete serving solution for the Telco Customer Churn model
with both programmatic API access and a user-friendly web interface.

Architecture:
- FastAPI: High-performance REST API with automatic OpenAPI documentation
- Gradio: User-friendly web UI for manual testing and demonstrations
- Pydantic: Data validation and automatic API documentation
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
from src.serving.inference import predict  # Core ML inference logic

# Initialize FastAPI application
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="ML API for predicting customer churn in telecom industry",
    version="1.0.0"
)

# === HEALTH CHECK ENDPOINT ===
# CRITICAL: Required for AWS Application Load Balancer health checks
@app.get("/")
def root():
    """
    Health check endpoint for monitoring and load balancer health checks.
    """
    return {"status": "ok"}

# === REQUEST DATA SCHEMA ===
# === REQUEST DATA SCHEMA ===
# Pydantic model for automatic validation and API documentation
class CustomerData(BaseModel):
    """
    Customer data schema for churn prediction.
    """
    # Demographics
    gender: str                # "Male" or "Female"
    Partner: str               # "Yes" or "No" - has partner
    
    # Phone & Internet services
    PhoneService: str          # "Yes" or "No"
    InternetService: str       # "DSL", "Fiber optic", or "No"
    OnlineSecurity: str        # "Yes", "No", or "No internet service"
    DeviceProtection: str      # "Yes", "No", or "No internet service"
    TechSupport: str           # "Yes", "No", or "No internet service"
    
    # Account information
    Contract: str              # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str      # "Yes" or "No"
    PaymentMethod: str         # "Electronic check", "Mailed check", etc.
    
    # Numeric features
    tenure: int                # Number of months with company
    MonthlyCharges: float      # Monthly charges in dollars

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main prediction endpoint for customer churn prediction.
    """
    try:
        # Convert Pydantic model to dict and call inference pipeline
        result = predict(data.dict())
        return {"prediction": result}
    except Exception as e:
        return {"error": str(e)}


# =================================================== # 


# === GRADIO WEB INTERFACE ===
def gradio_interface(
    gender, Partner, PhoneService, InternetService, OnlineSecurity,
    DeviceProtection, TechSupport, Contract, PaperlessBilling,
    PaymentMethod, tenure, MonthlyCharges
):
    # Construct data dictionary matching streamlined schema
    data = {
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
    
    # Call inference pipeline
    result = predict(data)
    return str(result)

# === GRADIO UI CONFIGURATION ===
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        # Demographics section
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        
        # Phone & Internet services section
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        
        # Contract and billing section
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        
        # Numeric features
        gr.Number(label="Tenure (months)", value=1, minimum=0, maximum=100),
        gr.Number(label="Monthly Charges ($)", value=85.0, minimum=0, maximum=200),
    ],
    outputs=gr.Textbox(label="Churn Prediction", lines=2),
    title="🔮 Telco Customer Churn Predictor",
    description="""
    **Predict customer churn probability using machine learning**
    
    Fill in the customer details below to get a churn prediction. The model uses XGBoost trained on 
    historical telecom customer data to identify customers at risk of churning.
    """,
    examples=[
        # High churn risk example
        ["Female", "No", "Yes", "Fiber optic", "No", "No", "No", "Month-to-month", "Yes", "Electronic check", 1, 85.0],
        # Low churn risk example  
        ["Male", "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Two year", "No", "Credit card (automatic)", 60, 45.0]
    ],
    theme=gr.themes.Soft()
)

# === MOUNT GRADIO UI INTO FASTAPI ===
# This creates the /ui endpoint that serves the Gradio interface
# IMPORTANT: This must be the final line to properly integrate Gradio with FastAPI
app = gr.mount_gradio_app(
    app,           # FastAPI application instance
    demo,          # Gradio interface
    path="/ui"     # URL path where Gradio will be accessible
)
