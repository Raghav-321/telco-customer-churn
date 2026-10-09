import os
import sys
import streamlit as st
import pandas as pd

# Add src to Python path so imports resolve cleanly
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.serving.inference import predict

# Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 1rem;
    }
    .risk-high {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .risk-low {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title & Subtitle
st.markdown('<div class="main-header">🔮 Telco Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time Machine Learning prediction to identify customers at risk of cancellation.</div>', unsafe_allow_html=True)

# Sidebar with Presets & Info
st.sidebar.header("⚡ Quick Presets")
preset = st.sidebar.selectbox(
    "Load sample customer profile:",
    ["Custom / Default", "High Churn Risk Customer", "Low Churn Risk Customer"]
)

if preset == "High Churn Risk Customer":
    default_gender = "Female"
    default_partner = "No"
    default_dependents = "No"
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_device = "No"
    default_tech = "No"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Month-to-month"
    default_billing = "Yes"
    default_payment = "Electronic check"
    default_tenure = 1
    default_monthly = 85.0
    default_total = 85.0
elif preset == "Low Churn Risk Customer":
    default_gender = "Male"
    default_partner = "Yes"
    default_dependents = "Yes"
    default_phone = "Yes"
    default_multiple = "Yes"
    default_internet = "DSL"
    default_security = "Yes"
    default_backup = "Yes"
    default_device = "Yes"
    default_tech = "Yes"
    default_tv = "No"
    default_movies = "No"
    default_contract = "Two year"
    default_billing = "No"
    default_payment = "Credit card (automatic)"
    default_tenure = 60
    default_monthly = 45.0
    default_total = 2700.0
else:
    default_gender = "Male"
    default_partner = "No"
    default_dependents = "No"
    default_phone = "Yes"
    default_multiple = "No"
    default_internet = "Fiber optic"
    default_security = "No"
    default_backup = "No"
    default_device = "No"
    default_tech = "No"
    default_tv = "Yes"
    default_movies = "Yes"
    default_contract = "Month-to-month"
    default_billing = "Yes"
    default_payment = "Electronic check"
    default_tenure = 12
    default_monthly = 70.0
    default_total = 840.0

st.sidebar.markdown("---")
st.sidebar.info("""
**Model Details:**
- **Engine:** XGBoost Classifier
- **Features:** 18 Customer Attributes
- **Framework:** MLflow & Scikit-learn
""")

# Input Form
with st.form("churn_prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"], index=0 if default_gender == "Male" else 1)
        partner = st.selectbox("Partner", ["Yes", "No"], index=0 if default_partner == "Yes" else 1)
        dependents = st.selectbox("Dependents", ["Yes", "No"], index=0 if default_dependents == "Yes" else 1)
        tenure = st.number_input("Tenure (Months with company)", min_value=0, max_value=100, value=default_tenure, step=1)

    with col2:
        st.subheader("📞 Phone & Internet Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0 if default_phone == "Yes" else 1)
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
            index=["Yes", "No", "No phone service"].index(default_multiple)
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
            index=["DSL", "Fiber optic", "No"].index(default_internet)
        )
        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_security)
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_backup)
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_device)
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_tech)
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_tv)
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"],
            index=["Yes", "No", "No internet service"].index(default_movies)
        )

    with col3:
        st.subheader("💳 Account & Billing")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(default_contract)
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if default_billing == "Yes" else 1)
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ].index(default_payment)
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=300.0,
            value=float(default_monthly),
            step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=15000.0,
            value=float(default_total),
            step=1.0
        )

    submit_button = st.form_submit_button("🚀 Run Churn Prediction", use_container_width=True)

# Prediction Result Section
if submit_button:
    customer_dict = {
        "gender": gender,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "tenure": int(tenure),
        "MonthlyCharges": float(monthly_charges),
        "TotalCharges": float(total_charges),
    }

    try:
        prediction_result = predict(customer_dict)

        st.markdown("### 📊 Prediction Results")

        if prediction_result == "Likely to churn":
            st.markdown("""
            <div class="risk-high">
                <h3 style="color: #991B1B; margin-top:0;">⚠️ High Churn Risk Detected</h3>
                <p style="color: #7F1D1D; margin-bottom: 0;"><strong>Status:</strong> Likely to churn</p>
                <p style="color: #7F1D1D;">This customer shows behavioral indicators strongly correlated with churn (e.g., month-to-month contract, electronic check payment, short tenure).</p>
            </div>
            """, unsafe_allow_html=True)

            st.warning("💡 **Recommended Retention Actions:**\n"
                       "- Offer long-term contract discounts (1-year or 2-year upgrade).\n"
                       "- Encourage enrollment in automatic payment (Credit card or Bank transfer).\n"
                       "- Provide complimentary Tech Support or Online Security bundle.")

        else:
            st.markdown("""
            <div class="risk-low">
                <h3 style="color: #065F46; margin-top:0;">✅ Customer is Loyal / Low Risk</h3>
                <p style="color: #047857; margin-bottom: 0;"><strong>Status:</strong> Not likely to churn</p>
                <p style="color: #047857;">Customer profile indicates strong retention stability.</p>
            </div>
            """, unsafe_allow_html=True)

            st.success("🎉 **Action:** Maintain regular customer engagement and satisfaction monitoring.")

        # Show raw feature dictionary
        with st.expander("🔍 View Processed Feature Payload"):
            st.json(customer_dict)

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
