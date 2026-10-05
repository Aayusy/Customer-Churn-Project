# 📉 Customer Churn Prediction App

Ek end-to-end Machine Learning web application jo customers ke data ko analyze karke yeh predict karti hai ki kaun sa customer service chod kar (churn) ja sakta hai. Is project mein machine learning models (`.pkl`) aur Streamlit ka use karke ek interactive user interface banaya gaya hai.

---

## 🚀 Key Features

* **Real-time Predictions:** User inputs ke basis par instant customer churn risk prediction provide karna.
* **Trained ML Models:** Optimized classification models (`best_model.pkl`) ka use accuracy enhance karne ke liye.
* **Data Preprocessing Pipeline:** Features ko properly scale aur encode karne ke liye saved `scaler.pkl` aur `encoder.pkl` ka integration.
* **Interactive Dashboard:** Clean aur user-friendly web interface powered by Streamlit.

---

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── app.py             # Main Streamlit application file
├── best_model.pkl     # Trained machine learning model
├── encoder.pkl        # Categorical data encoder
├── scaler.pkl         # Feature scaler object
└── README.md          # Project documentation
