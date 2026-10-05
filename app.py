import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Customer Churn Prediction", page_icon="📊", layout="wide")

# Load model, encoders, and scaler (Cached so they load only once)
@st.cache_resource
def load_data():
    with open('best_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('encoder.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler_data = pickle.load(scaler_file)
    return loaded_model, encoders, scaler_data

loaded_model, encoders, scaler_data = load_data()

st.title("📊 Customer Churn Prediction App")
st.write("Customer ki details neeche bharein aur check karein ki woh churn karega ya nahi.")

# Create a form using Streamlit columns
with st.form("churn_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    with col3:
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=800.0)

    # Submit Button
    submit_button = st.form_submit_button(label="Predict Churn", use_container_width=True)

# Prediction Logic when button is clicked
if submit_button:
    input_data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # DataFrame conversion and transformation
    input_df = pd.DataFrame([input_data])

    for col, encoder in encoders.items():
        input_df[col] = encoder.transform(input_df[col])

    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[numerical_cols] = scaler_data.transform(input_df[numerical_cols])

    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0, 1]

    result_text = "Churn" if prediction == 1 else "No Churn"

    st.markdown("---")
    st.subheader("Prediction Results:")
    if result_text == "Churn":
        st.error(f"⚠️ **Prediction:** {result_text}")
    else:
        st.success(f"✅ **Prediction:** {result_text}")
    
    st.info(f"📈 **Probability of Churn:** {probability:.2f}")