# Traffic-Congestion-Predictor-using-ML
MS AI AZURE INTERNSHIP  MAY 2025 BATCH : PROJECT
# 🚦 Traffic Congestion Predictor

This project presents a machine learning-based system to classify traffic congestion levels (Low, Medium, High) using zone-wise and environmental data. The model was developed as part of an AICTE internship project and aims to support smarter urban traffic planning.

---

## 📌 Project Overview

- **Objective:** Predict traffic congestion levels using machine learning
- **Model Used:** Random Forest Classifier
- **Accuracy Achieved:** 100% (after target binning and preprocessing)
- **Tools:** Python, Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 📊 Dataset

- **Total Records:** 1440
- **Features Used:**
  - Zone number
  - Temperature
  - Weather (one-hot encoded)
  - Derived Hour (Zone % 24)
- **Target:** Traffic (binned into 3 levels: Low, Medium, High)

---

## 🔧 System Pipeline

1. **Data Cleaning** – Dropped unnecessary columns, handled encoding
2. **Feature Engineering** – Created new Hour feature
3. **Target Binning** – Original 1–5 levels binned to 3 categories
4. **Model Training** – Random Forest with 80/20 split
5. **Evaluation** – Accuracy, classification report, confusion matrix
6. **Visualization** – Feature importance, zone-wise traffic plot

---

## 🧠 Model Performance

- **Accuracy:** `100%` on the test dataset
- **Visual Outputs:**
  - Confusion Matrix (Perfect classification)
  - Feature Importance (Zone and Temperature ranked highest)
  - Boxplots and bar charts showing traffic distribution

---



## 🚀 Future Scope

- Integrate live traffic and weather APIs
- Expand dataset to include multiple cities or dynamic zones
- Use advanced models (e.g., XGBoost, LightGBM)
- Deploy as a web app or dashboard (Streamlit/Flask)
- Explore edge deployment for IoT-based traffic sensors

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org)
- [Pandas & NumPy](https://pandas.pydata.org/)
- [Matplotlib & Seaborn](https://seaborn.pydata.org/)
- [AICTE Internship Portal](https://internship.aicte-india.org)
- [Edunet Foundation](https://edunetfoundation.org/)

---

> 🚦 Built with Python for AICTE Internship, 2025 – by ATHAR SHAIKH
