# 📊 Telco Customer Churn Prediction & Insights

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Machine Learning](https://img.shields.io/badge/Task-Classification-green.svg)

## 🎯 Project Overview
Customer churn is one of the most critical metrics for telecommunications companies. This project aims to build a machine learning pipeline to **predict which customers are likely to leave** and identify the **underlying factors** driving churn.

By leveraging historical data, we provide actionable insights to help marketing and retention teams reduce customer attrition.

## 📂 Dataset Description
The dataset contains **7,043 rows** and **21 features**, including:
- **Demographics:** Gender, seniority, partners, and dependents.
- **Service Details:** Tenure, phone service, internet service (DSL, Fiber optic), and security add-ons.
- **Billing Information:** Contract type, payment method, monthly charges, and total charges.
- **Target Variable:** `Churn` (Yes/No).

## 🛠️ Tech Stack & Methods
- **Preprocessing:** `StandardScaler`, `OneHotEncoder`, `LabelEncoder`.
- **Handling Imbalance:** `class_weight='balanced'` and Threshold tuning.
- **Models Benchmarked:** Logistic Regression, Random Forest, XGBoost, SVM, KNN, and Decision Trees.
- **Evaluation Metrics:** ROC-AUC, Precision-Recall, F1-Score, and Confusion Matrix.

## 🚀 Key Achievements
- **Recall Optimization:** Achieved **~79% Recall** for the Churn class using Logistic Regression, ensuring the company catches the majority of potential leavers.
- **Strategic Thresholding:** Fine-tuned the decision threshold to **0.6** to balance the cost of false alarms with the benefit of retention.
- **Feature Engineering:** Effectively scaled numerical data and encoded categorical variables to maximize model stability.

## 📈 Top Business Insights
Through **Feature Importance** analysis, we identified the primary drivers of churn:
1. **Contract Type:** "Month-to-month" contracts are the strongest predictors of churn.
2. **Tenure:** Customers in their first 6 months are at the highest risk.
3. **Internet Service:** Users with **Fiber Optic** internet have higher churn rates, suggesting potential pricing or service quality issues.



## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/churn-prediction.git](https://github.com/your-username/churn-prediction.git)