# 🚀 Customer Churn Prediction & Analytics Dashboard

## 📌 Project Overview

This project presents a **complete end-to-end Customer Churn Analysis and Prediction system** built using **Python and Streamlit**.

It combines:

* 📊 Advanced data analysis
* 🤖 Machine learning modeling
* 📈 Interactive dashboarding
* 💡 Business-driven insights

The goal is to help companies **understand why customers churn and take actionable decisions to retain them**.

---

## 🎯 Business Problem

Customer churn is one of the most critical challenges for subscription-based businesses.

Losing customers means:

* Revenue loss 💸
* Increased acquisition cost 📈
* Reduced customer lifetime value

This project answers key questions:

* Who is likely to churn?
* Why are customers leaving?
* How much revenue is at risk?
* What actions should be taken?

---

## 🧠 Key Features

### 🔹 1. Executive Overview Dashboard

* Total customers
* Churned customers
* Churn rate %
* Predicted churn customers
* Average Monthly Charges
* Average tenure
* Revenue at risk

---

### 🔹 2. Churn Analysis

Compare customers who churned vs those who stayed:

* Gender
* Senior Citizen
* Partner / Dependents
* Contract Type
* Internet Service
* Payment Method
* Paperless Billing

📊 Helps answer:

* Do monthly contracts churn more?
* Are Fiber optic users more likely to leave?
* Does payment method affect churn?

---

### 🔹 3. Financial Insights 💰

* Average Monthly Charges (Churn vs Non-Churn)
* Estimated total revenue
* **At-Risk Revenue**
* Monthly revenue loss from predicted churn

---

### 🔹 4. Risk Drivers Analysis 🔍

Identifies the most important factors behind churn:

* Tenure (new vs loyal customers)
* Contract type
* Internet service
* Tech support & security
* Payment method
* Monthly charges

📌 Built using:

* Feature importance
* Correlation analysis

---

### 🔹 5. Customer Segmentation

Customers are grouped into:

* 🔴 High Risk
* 🟠 Medium Risk
* 🟢 Low Risk

Also segmented by:

* New vs Loyal customers
* High-value vs Low-value customers

---

### 🔹 6. Individual Prediction 🔮

Interactive form to predict churn for a single customer:

Inputs:

* Demographics
* Services
* Contract details
* Payment method
* Monthly charges

Outputs:

* Churn prediction (Yes / No)
* Probability score
* Risk level
* Explanation
* Action recommendation

---

### 🔹 7. Batch Prediction (Advanced)

Upload a CSV file to:

* Predict churn for multiple customers
* Rank customers by risk
* Identify high-risk segments
* Export results

---

### 🔹 8. Actionable Customers Table 📋

A business-ready table showing:

* Customer ID
* Churn probability
* Risk level
* Monthly charges
* Contract type
* Key risk drivers
* Recommended action

Example:

| Customer   | Risk | Probability | Action                |
| ---------- | ---- | ----------- | --------------------- |
| 7590-VHVEG | High | 0.87        | Offer annual discount |
| 9237-HQITU | High | 0.81        | Add tech support      |

---

### 🔹 9. Interactive Filters 🎛️

Users can filter by:

* Contract
* Internet Service
* Payment Method
* Gender
* Senior Citizen
* Partner / Dependents
* Churn status
* Risk level

---

## 🤖 Machine Learning Model

### Models Used:

* Logistic Regression
* Random Forest (optional)
* XGBoost (optional)

### Techniques Applied:

* Feature Engineering
* One-Hot Encoding
* Class Imbalance Handling
* Hyperparameter tuning
* Threshold optimization

### Evaluation Metrics:

* Recall (focus on catching churn)
* Precision
* F1-score
* ROC-AUC

---

## 📊 Key Insights

* Customers on **month-to-month contracts** have the highest churn rate
* **New customers** are significantly more likely to churn
* Customers without **Tech Support** or **Online Security** are at higher risk
* **Electronic check users** show higher churn probability
* Long-term contracts significantly reduce churn

---

## 🛠️ Tech Stack

* Python 🐍
* Streamlit 📊
* Pandas & NumPy
* Scikit-learn
* XGBoost
* Matplotlib / Seaborn

---

## ▶️ How to Run the Project

### 1. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn xgboost
```

### 2. Run the app

```bash
streamlit run Churn_Traning1.py
```

### 3. Open in browser

```
http://localhost:8501
```

---

## 📁 Project Structure

```
├── Churn_Traning1.py      # Main Streamlit app
├── churn_model.pkl        # Trained model
├── data.csv               # Dataset
├── README.md              # Project documentation
```

---

## 💡 Future Improvements

* SHAP Explainability 🔍
* Real-time API integration
* Deployment on cloud (Streamlit Cloud / AWS)
* Automated retraining pipeline
* Customer retention recommendation engine

---

## 🏆 Why This Project Matters

This is not just a dashboard.

It is a **decision-support system** that:

* Predicts churn
* Explains customer behavior
* Quantifies financial risk
* Recommends business actions

---

## 👨‍💻 Author

AI Engineering Student passionate about:

* Machine Learning
* Data Science
* Business Intelligence

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and feel free to connect!
