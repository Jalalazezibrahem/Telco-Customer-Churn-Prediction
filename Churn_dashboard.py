import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# =========================================================
# PAGE CONFIG
# =========================================================
df = pd.read_csv(r"C:\Users\Dell\Downloads\archive (28)\WA_Fn-UseC_-Telco-Customer-Churn.csv")

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], .main, .block-container {
        background: #08112a !important;
        color: #e5e9f0 !important;
    }
    .reportview-container .main .block-container {
        background: rgba(10, 20, 40, 0.98) !important;
        color: #e5e9f0 !important;
    }
    .block-container {
        padding-top: 1.75rem;
        padding-bottom: 1.25rem;
        padding-left: 2.25rem;
        padding-right: 2.25rem;
        max-width: 1600px;
        margin: 0 auto;
    }
    .css-1oe6wy4, .css-1d391kg {
        background: transparent !important;
    }
    .hero {
        background: linear-gradient(135deg, #0f2c72, #003f86, #9f1c4a);
        padding: 1.6rem 1.8rem;
        border-radius: 26px;
        color: white;
        margin-bottom: 1.4rem;
        box-shadow: 0 18px 50px rgba(0,0,0,0.24);
    }
    .hero h2 {
        margin-bottom: 0.35rem;
        font-size: 2.05rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.75rem 0 0.25rem;
        line-height: 1.55;
        opacity: 0.95;
    }
    .hero .badge {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-size: 0.88rem;
        margin-top: 0.65rem;
    }
    .stMetric {
        border-radius: 20px;
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.25);
        padding: 1rem 1.1rem;
        transition: transform 0.2s ease;
    }
    .stMetric:hover {
        transform: translateY(-2px);
    }
    .stMetricValue, .css-1v0mbdj.ejae7h70 {
        color: #e5e9f0 !important;
        font-weight: 700;
    }
    .stMetricLabel, .css-1j8bk8l.e1fqkh3o4 {
        color: #cbd5e1 !important;
        font-size: 0.95rem;
    }
    .section-heading {
        font-size: 1.45rem;
        font-weight: 700;
        color: #a8d0ff;
        margin-top: 1.75rem;
        margin-bottom: 0.85rem;
    }
    .stSidebar .css-1d391kg, .stSidebar .css-1lcbmhc.e1fqkh3o3 {
        background: rgba(5, 15, 35, 0.92) !important;
        border-radius: 18px;
        color: #e5e9f0 !important;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .stSidebar .stMarkdown {
        margin-bottom: 1rem;
        color: #e5e9f0 !important;
    }
    .stButton button, .stDownloadButton button {
        background: #1f4b8b !important;
        color: #ffffff !important;
        border: none !important;
    }
    .stDataFrame table {
        background: rgba(255,255,255,0.05) !important;
        color: #e5e9f0 !important;
    }
    .stDataFrame th {
        color: #cbd5e1 !important;
    }
    .stDataFrame td {
        color: #e5e9f0 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(10, 20, 40, 0.98) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05) !important;
        color: #e5e9f0 !important;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: #1f4b8b !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# MODEL FILE
# =========================================================
MODEL_FILE = "churn_model.pkl"

# =========================================================
# DATA CLEANING
# IMPORTANT:
# This assumes df already exists in your file BEFORE this dashboard code.
# =========================================================
if "df" not in globals():
    st.error("The dataframe 'df' was not found. Please make sure df is created before the dashboard code.")
    st.stop()

df = df.copy()

# Convert numeric SeniorCitizen values to Yes/No labels for better user input handling
if "SeniorCitizen" in df.columns:
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({1: "Yes", 0: "No", "1": "Yes", "0": "No", True: "Yes", False: "No"}).astype(str)

for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "Churn" in df.columns:
    df["Churn"] = df["Churn"].astype(str).str.strip()

# Ensure customerID exists
if "customerID" not in df.columns:
    df["customerID"] = [f"CUST-{i+1}" for i in range(len(df))]

# =========================================================
# REQUIRED COLUMNS CHECK
# =========================================================
required_cols = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "InternetService", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required columns in df: {missing}")
    st.stop()

# Optional columns that may or may not exist
optional_cols = [
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Separate Contract for Ordinal Encoding
    contract_col = ["Contract"] if "Contract" in categorical_cols else []
    other_cat_cols = [c for c in categorical_cols if c != "Contract"]

    transformers = []
    if contract_col:
        contract_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(categories=[["Month-to-month", "One year", "Two year"]]))
        ])
        transformers.append(("contract", contract_pipeline, contract_col))

    if other_cat_cols:
        other_cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", other_cat_pipeline, other_cat_cols))

    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    return preprocessor


def align_input_features(X_input, expected_features):
    X_temp = X_input.copy()
    missing = [col for col in expected_features if col not in X_temp.columns]
    for col in missing:
        X_temp[col] = np.nan
    return X_temp.reindex(columns=expected_features)


def prepare_features(df_input, reference_features=None):
    X = df_input.drop(columns=["Churn", "customerID"], errors="ignore")
    if reference_features is not None:
        X = align_input_features(X, reference_features)
    return X


@st.cache_resource
def load_or_train_model(df_input, model_path):
    """
    Load saved model if possible.
    If loading fails, train a fallback Pipeline model.
    Also return metadata about how to prepare features.
    """
    if Path(model_path).exists():
        try:
            model = joblib.load(model_path)

            model_info = {
                "source": "Loaded from churn_model.pkl",
                "mode": "auto"
            }

            if hasattr(model, "feature_names_in_"):
                model_info["expected_features"] = list(model.feature_names_in_)
            else:
                model_info["expected_features"] = None

            return model, model_info

        except Exception as e:
            st.warning("Saved model could not be loaded. A fallback model will be trained instead.")
            st.code(str(e))

    temp = df_input.copy()
    temp = temp.dropna(subset=["Churn"]).copy()

    y = temp["Churn"].map({"Yes": 1, "No": 0})
    X = temp.drop(columns=["Churn", "customerID"], errors="ignore")

    preprocessor = build_preprocessor(X)

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model.fit(X_train, y_train)

    model_info = {
        "source": "Fallback Logistic Regression model trained automatically",
        "mode": "pipeline",
        "expected_features": X.columns.tolist()
    }

    return model, model_info


def get_prediction_outputs(model, X, model_info=None):
    """
    Handle prediction for:
    1) sklearn Pipeline models
    2) models trained on one-hot encoded data
    3) fallback models
    """
    X_input = X.copy()

    expected_features = None
    if model_info is not None:
        expected_features = model_info.get("expected_features")

    if expected_features is None and hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

    if expected_features is not None:
        X_input = align_input_features(X_input, expected_features)

    # Case 1: Pipeline model
    if isinstance(model, Pipeline):
        pred = model.predict(X_input)

        pred = pd.Series(pred).replace({
            "Yes": 1, "No": 0, "yes": 1, "no": 0,
            True: 1, False: 0
        })
        pred = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_input)[:, 1]
        else:
            prob = pred.astype(float).values

        return pred, prob

    # Case 2: Non-pipeline model with encoded features
    if expected_features is not None:
        X_encoded = pd.get_dummies(X_input)

        for col in expected_features:
            if col not in X_encoded.columns:
                X_encoded[col] = 0

        X_encoded = X_encoded.reindex(columns=expected_features, fill_value=0)

        pred = model.predict(X_encoded)

        pred = pd.Series(pred).replace({
            "Yes": 1, "No": 0, "yes": 1, "no": 0,
            True: 1, False: 0
        })
        pred = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_encoded)[:, 1]
        else:
            prob = pred.astype(float).values

        return pred, prob

    # Case 3: final fallback
    pred = model.predict(X_input)

    pred = pd.Series(pred).replace({
        "Yes": 1, "No": 0, "yes": 1, "no": 0,
        True: 1, False: 0
    })
    pred = pd.to_numeric(pred, errors="coerce").fillna(0).astype(int)

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_input)[:, 1]
    else:
        prob = pred.astype(float).values

    return pred, prob


def risk_band(prob):
    if prob >= 0.70:
        return "High Risk"
    elif prob >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def tenure_segment(v):
    if pd.isna(v):
        return "Unknown"
    if v <= 12:
        return "New Customers"
    elif v <= 36:
        return "Mid-Term Customers"
    return "Loyal Customers"


def value_segment(v):
    if pd.isna(v):
        return "Unknown"
    if v >= 80:
        return "High Value"
    elif v >= 40:
        return "Medium Value"
    return "Low Value"


def churn_rate_by_feature(df_input, feature):
    temp = df_input.copy()
    temp["ChurnFlag"] = temp["Churn"].map({"Yes": 1, "No": 0})

    out = (
        temp.groupby(feature, dropna=False)
        .agg(
            Customers=("customerID", "count"),
            ChurnRate=("ChurnFlag", "mean")
        )
        .reset_index()
    )
    out["ChurnRate"] = (out["ChurnRate"] * 100).round(2)
    out = out.sort_values("ChurnRate", ascending=False)
    return out


def infer_reasons(row):
    reasons = []

    if row.get("Contract") == "Month-to-month":
        reasons.append("Month-to-month contract")
    if row.get("InternetService") == "Fiber optic":
        reasons.append("Fiber optic service")
    if row.get("PaymentMethod") == "Electronic check":
        reasons.append("Electronic check payment")
    if row.get("TechSupport") == "No":
        reasons.append("No tech support")
    if row.get("OnlineSecurity") == "No":
        reasons.append("No online security")
    if pd.notna(row.get("tenure")) and row.get("tenure") <= 12:
        reasons.append("Short customer tenure")
    if pd.notna(row.get("MonthlyCharges")) and row.get("MonthlyCharges") >= 80:
        reasons.append("High monthly charges")

    if not reasons:
        reasons.append("General churn pattern detected")

    return ", ".join(reasons[:2])


def recommend_action(row):
    actions = []

    if row.get("Contract") == "Month-to-month":
        actions.append("Offer annual contract discount")
    if row.get("TechSupport") == "No":
        actions.append("Promote tech support add-on")
    if row.get("OnlineSecurity") == "No":
        actions.append("Offer security bundle")
    if row.get("PaymentMethod") == "Electronic check":
        actions.append("Encourage auto-pay migration")
    if pd.notna(row.get("MonthlyCharges")) and row.get("MonthlyCharges") >= 80:
        actions.append("Offer pricing retention package")

    if not actions:
        actions.append("Run proactive retention outreach")

    return " | ".join(actions[:2])


def feature_importance_from_pipeline(model):
    try:
        if not isinstance(model, Pipeline):
            return None

        preprocessor = model.named_steps.get("preprocessor")
        classifier = model.named_steps.get("classifier")

        if preprocessor is None or classifier is None:
            return None

        feature_names = preprocessor.get_feature_names_out()

        if hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        elif hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        else:
            return None

        fi = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values("Importance", ascending=False)

        fi["Feature"] = fi["Feature"].str.replace("cat__", "", regex=False)
        fi["Feature"] = fi["Feature"].str.replace("contract__", "", regex=False)
        fi["Feature"] = fi["Feature"].str.replace("num__", "", regex=False)

        return fi.head(15)

    except Exception:
        return None

# =========================================================
# MODEL LOADING / TRAINING
# =========================================================
model, model_info = load_or_train_model(df, MODEL_FILE)
model_source = model_info["source"]

X_all = prepare_features(df)
feature_columns = X_all.columns.tolist()
if model_info.get("expected_features") is None:
    model_info["expected_features"] = feature_columns

pred_all, prob_all = get_prediction_outputs(model, X_all, model_info)

df_model = df.copy()
df_model["PredictedChurn"] = pred_all
df_model["ChurnProbability"] = prob_all
df_model["RiskLevel"] = df_model["ChurnProbability"].apply(risk_band)
df_model["TenureSegment"] = df_model["tenure"].apply(tenure_segment)
df_model["ValueSegment"] = df_model["MonthlyCharges"].apply(value_segment)
df_model["RiskReason"] = df_model.apply(infer_reasons, axis=1)
df_model["RecommendedAction"] = df_model.apply(recommend_action, axis=1)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.header("Dashboard Filters")
st.sidebar.markdown("Use these filters to highlight customer segments and compare churn behavior.")

def safe_options(dataframe, col):
    if col in dataframe.columns:
        return sorted(dataframe[col].dropna().astype(str).unique().tolist())
    return []

contract_options = safe_options(df_model, "Contract")
internet_options = safe_options(df_model, "InternetService")
payment_options = safe_options(df_model, "PaymentMethod")
gender_options = safe_options(df_model, "gender")
senior_options = safe_options(df_model, "SeniorCitizen")
partner_options = safe_options(df_model, "Partner")
dependents_options = safe_options(df_model, "Dependents")
churn_options = safe_options(df_model, "Churn")
risk_options = safe_options(df_model, "RiskLevel")

with st.sidebar.expander("Customer profile filters", expanded=True):
    selected_contract = st.multiselect("Contract", contract_options, default=contract_options)
    selected_internet = st.multiselect("Internet Service", internet_options, default=internet_options)
    selected_payment = st.multiselect("Payment Method", payment_options, default=payment_options)
    selected_gender = st.multiselect("Gender", gender_options, default=gender_options)
    selected_senior = st.multiselect("Senior Citizen", senior_options, default=senior_options)
    selected_partner = st.multiselect("Partner", partner_options, default=partner_options)
    selected_dependents = st.multiselect("Dependents", dependents_options, default=dependents_options)

with st.sidebar.expander("Outcome & risk filters", expanded=False):
    selected_churn = st.multiselect("Actual Churn", churn_options, default=churn_options)
    selected_risk = st.multiselect("Risk Level", risk_options, default=risk_options)

filtered = df_model[
    df_model["Contract"].astype(str).isin(selected_contract) &
    df_model["InternetService"].astype(str).isin(selected_internet) &
    df_model["PaymentMethod"].astype(str).isin(selected_payment) &
    df_model["gender"].astype(str).isin(selected_gender) &
    df_model["SeniorCitizen"].astype(str).isin(selected_senior) &
    df_model["Partner"].astype(str).isin(selected_partner) &
    df_model["Dependents"].astype(str).isin(selected_dependents) &
    df_model["Churn"].astype(str).isin(selected_churn) &
    df_model["RiskLevel"].astype(str).isin(selected_risk)
].copy()

if filtered.empty:
    st.warning("No rows match the selected filters.")
    st.stop()

# =========================================================
# HERO
# =========================================================
st.markdown(f"""
<div class="hero">
    <div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:1rem;align-items:flex-start;">
        <div style="max-width:800px;">
            <h2>Customer Churn Analytics & Prediction Dashboard</h2>
            <p>Portfolio-ready dashboard for churn monitoring, customer analysis, financial exposure, predictive scoring, and retention actions.</p>
            <span class="badge">Live model insights</span>
        </div>
        <div style="text-align:right;min-width:220px;">
            <div style="font-size:0.95rem;opacity:0.9;margin-bottom:0.6rem;">Model status</div>
            <div style="font-size:1.05rem;font-weight:700;">{model_source}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI SECTION
# =========================================================
st.markdown("<div class='section-heading'>Key Performance Indicators</div>", unsafe_allow_html=True)

total_customers = len(filtered)
actual_churned = int((filtered["Churn"] == "Yes").sum())
churn_rate = round(actual_churned / total_customers * 100, 2) if total_customers else 0
predicted_churned = int(filtered["PredictedChurn"].sum())
avg_monthly = round(filtered["MonthlyCharges"].mean(), 2)
avg_tenure = round(filtered["tenure"].mean(), 2)
current_revenue = round(filtered["MonthlyCharges"].sum(), 2)
at_risk_revenue = round(filtered.loc[filtered["RiskLevel"] == "High Risk", "MonthlyCharges"].sum(), 2)
predicted_monthly_loss = round(filtered.loc[filtered["PredictedChurn"] == 1, "MonthlyCharges"].sum(), 2)

k1, k2, k3, k4 = st.columns(4)
k5, k6, k7 = st.columns(3)

k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("Customers Churned", f"{actual_churned:,}")
k3.metric("Churn Rate", f"{churn_rate}%")
k4.metric("Predicted to Churn", f"{predicted_churned:,}")
k5.metric("Avg Monthly Charges", f"${avg_monthly:,.2f}")
k6.metric("Avg Tenure", f"{avg_tenure} months")
k7.metric("At-Risk Monthly Revenue", f"${at_risk_revenue:,.2f}")

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Churn Analysis",
    "Model Insights",
    "Predict a Customer",
    "High-Risk Customers"
])

# =========================================================
# TAB 1 - OVERVIEW
# =========================================================
with tab1:
    st.markdown("### Executive Overview")

    c1, c2 = st.columns([1.2, 1])

    with c1:
        contract_dist = (
            filtered["Contract"].value_counts(normalize=True).mul(100).round(2).reset_index()
        )
        contract_dist.columns = ["Contract", "Percentage"]

        fig_contract = px.bar(
            contract_dist,
            x="Contract",
            y="Percentage",
            text="Percentage",
            title="Customer Distribution by Contract Type"
        )
        fig_contract.update_traces(textposition="outside")
        fig_contract.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_contract, use_container_width=True)

    with c2:
        internet_churn = churn_rate_by_feature(filtered, "InternetService")
        fig_internet = px.bar(
            internet_churn,
            x="InternetService",
            y="ChurnRate",
            color="ChurnRate",
            text="ChurnRate",
            title="Churn Rate by Internet Service"
        )
        fig_internet.update_traces(textposition="outside")
        fig_internet.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_internet, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        revenue_df = pd.DataFrame({
            "Metric": ["Current Monthly Revenue", "At-Risk Revenue", "Predicted Monthly Loss"],
            "Amount": [current_revenue, at_risk_revenue, predicted_monthly_loss]
        })
        fig_revenue = px.bar(
            revenue_df,
            x="Metric",
            y="Amount",
            text="Amount",
            title="Revenue Exposure Snapshot"
        )
        fig_revenue.update_traces(texttemplate="$%{y:,.2f}", textposition="outside")
        fig_revenue.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_revenue, use_container_width=True)

    with c4:
        risk_dist = filtered["RiskLevel"].value_counts().reset_index()
        risk_dist.columns = ["RiskLevel", "Customers"]
        fig_risk = px.pie(
            risk_dist,
            names="RiskLevel",
            values="Customers",
            title="Customer Risk Segmentation"
        )
        fig_risk.update_layout(height=400, template="plotly_dark")
        st.plotly_chart(fig_risk, use_container_width=True)

# =========================================================
# TAB 2 - CHURN ANALYSIS
# =========================================================
with tab2:
    st.markdown("### Churn Comparison: Churn = Yes vs Churn = No")
    st.markdown("Compare customer behavior across the key business dimensions.")

    compare_candidates = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "PaperlessBilling"
    ]
    compare_available = [c for c in compare_candidates if c in filtered.columns]

    compare_feature = st.selectbox(
        "Select a dimension to compare churn behavior",
        compare_available
    )

    analysis_df = churn_rate_by_feature(filtered, compare_feature)
    fig_compare = px.bar(
        analysis_df,
        x=compare_feature,
        y="ChurnRate",
        color="ChurnRate",
        text="ChurnRate",
        hover_data=["Customers"],
        title=f"Churn Rate by {compare_feature}"
    )
    fig_compare.update_traces(textposition="outside")
    fig_compare.update_layout(height=460, template="plotly_dark")
    st.plotly_chart(fig_compare, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Tenure Distribution by Churn")
        fig_tenure = px.histogram(
            filtered,
            x="tenure",
            color="Churn",
            barmode="overlay",
            nbins=30,
            title="Tenure Distribution"
        )
        fig_tenure.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_tenure, use_container_width=True)

    with col_b:
        st.markdown("#### Monthly Charges by Churn")
        fig_monthly = px.box(
            filtered,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            title="Monthly Charges: Churned vs Retained"
        )
        fig_monthly.update_layout(height=420, template="plotly_dark")
        st.plotly_chart(fig_monthly, use_container_width=True)

    st.markdown("### Financial Comparison")

    financial_summary = (
        filtered.groupby("Churn", dropna=False)
        .agg(
            Customers=("customerID", "count"),
            AvgMonthlyCharges=("MonthlyCharges", "mean"),
            AvgTotalCharges=("TotalCharges", "mean"),
            TotalCurrentRevenue=("MonthlyCharges", "sum")
        )
        .reset_index()
    )

    financial_summary["AvgMonthlyCharges"] = financial_summary["AvgMonthlyCharges"].round(2)
    financial_summary["AvgTotalCharges"] = financial_summary["AvgTotalCharges"].round(2)
    financial_summary["TotalCurrentRevenue"] = financial_summary["TotalCurrentRevenue"].round(2)

    st.dataframe(financial_summary, use_container_width=True, hide_index=True)

    st.markdown("### Quick Business Insights")

    top_contract = churn_rate_by_feature(filtered, "Contract").head(1)
    top_internet = churn_rate_by_feature(filtered, "InternetService").head(1)
    top_payment = churn_rate_by_feature(filtered, "PaymentMethod").head(1)
    top_senior = churn_rate_by_feature(filtered, "SeniorCitizen").head(1)

    insight_col1, insight_col2 = st.columns(2)

    with insight_col1:
        st.info(
            f"Highest-risk contract segment: **{top_contract.iloc[0, 0]}** "
            f"with churn rate **{top_contract.iloc[0]['ChurnRate']}%**."
        )
        st.info(
            f"Highest-risk internet segment: **{top_internet.iloc[0, 0]}** "
            f"with churn rate **{top_internet.iloc[0]['ChurnRate']}%**."
        )

    with insight_col2:
        st.info(
            f"Highest-risk payment segment: **{top_payment.iloc[0, 0]}** "
            f"with churn rate **{top_payment.iloc[0]['ChurnRate']}%**."
        )
        st.info(
            f"Highest-risk senior segment: **{top_senior.iloc[0, 0]}** "
            f"with churn rate **{top_senior.iloc[0]['ChurnRate']}%**."
        )

# =========================================================
# TAB 3 - MODEL INSIGHTS
# =========================================================
with tab3:
    st.markdown("### Risk Drivers & Segmentation")

    left, right = st.columns([1.05, 0.95])

    with left:
        fi = feature_importance_from_pipeline(model)
        if fi is not None:
            fig_fi = px.bar(
                fi.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Top Model Features"
            )
            fig_fi.update_layout(height=520, template="plotly_dark")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.warning("Feature importance is not directly available for this model.")
            st.caption("Business-driven churn drivers are still shown below.")

    with right:
        driver_candidates = [
            "tenure", "Contract", "InternetService", "OnlineSecurity",
            "TechSupport", "PaymentMethod", "MonthlyCharges"
        ]
        driver_available = [c for c in driver_candidates if c in filtered.columns]

        st.markdown("#### Business Risk Drivers")
        driver_choice = st.selectbox("Choose a churn driver", driver_available)

        if driver_choice in ["tenure", "MonthlyCharges"]:
            temp = filtered.copy()

            if driver_choice == "tenure":
                temp["Band"] = pd.cut(
                    temp["tenure"],
                    bins=[-1, 12, 24, 48, 72, 1000],
                    labels=["0-12", "13-24", "25-48", "49-72", "72+"]
                )
                driver_plot = churn_rate_by_feature(temp.rename(columns={"Band": "DriverBand"}), "DriverBand")
                fig_driver = px.bar(
                    driver_plot,
                    x="DriverBand",
                    y="ChurnRate",
                    text="ChurnRate",
                    title="Churn Rate by Tenure Band"
                )
            else:
                temp["Band"] = pd.cut(
                    temp["MonthlyCharges"],
                    bins=[-1, 30, 60, 90, 120, 1000],
                    labels=["0-30", "31-60", "61-90", "91-120", "120+"]
                )
                driver_plot = churn_rate_by_feature(temp.rename(columns={"Band": "DriverBand"}), "DriverBand")
                fig_driver = px.bar(
                    driver_plot,
                    x="DriverBand",
                    y="ChurnRate",
                    text="ChurnRate",
                    title="Churn Rate by Monthly Charges Band"
                )
        else:
            driver_plot = churn_rate_by_feature(filtered, driver_choice)
            fig_driver = px.bar(
                driver_plot,
                x=driver_choice,
                y="ChurnRate",
                text="ChurnRate",
                title=f"Churn Rate by {driver_choice}"
            )

        fig_driver.update_traces(textposition="outside")
        fig_driver.update_layout(height=520, template="plotly_dark")
        st.plotly_chart(fig_driver, use_container_width=True)

    st.markdown("### Customer Segmentation")

    s1, s2, s3 = st.columns(3)

    with s1:
        seg1 = filtered["RiskLevel"].value_counts().reset_index()
        seg1.columns = ["Segment", "Customers"]
        fig_seg1 = px.pie(seg1, names="Segment", values="Customers", title="Risk Segments")
        fig_seg1.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig_seg1, use_container_width=True)

    with s2:
        seg2 = filtered["TenureSegment"].value_counts().reset_index()
        seg2.columns = ["Segment", "Customers"]
        fig_seg2 = px.pie(seg2, names="Segment", values="Customers", title="Tenure Segments")
        fig_seg2.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig_seg2, use_container_width=True)

    with s3:
        seg3 = filtered["ValueSegment"].value_counts().reset_index()
        seg3.columns = ["Segment", "Customers"]
        fig_seg3 = px.pie(seg3, names="Segment", values="Customers", title="Value Segments")
        fig_seg3.update_layout(height=360, template="plotly_dark")
        st.plotly_chart(fig_seg3, use_container_width=True)

# =========================================================
# TAB 4 - PREDICT A CUSTOMER
# =========================================================
with tab4:
    st.markdown("### Individual Customer Prediction")
    st.write("Enter a customer profile and generate a churn prediction with business recommendations.")

    input_defaults = {}
    for col in X_all.columns:
        if pd.api.types.is_numeric_dtype(X_all[col]):
            input_defaults[col] = float(X_all[col].median())
        else:
            mode_series = X_all[col].dropna().astype(str)
            input_defaults[col] = mode_series.mode().iloc[0] if not mode_series.empty else ""

    ordered_fields = [
        "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    ordered_fields = [f for f in ordered_fields if f in X_all.columns]

    with st.form("single_prediction_form"):
        col1, col2, col3 = st.columns(3)
        user_input = {}

        for idx, field in enumerate(ordered_fields):
            target_col = [col1, col2, col3][idx % 3]

            with target_col:
                if pd.api.types.is_numeric_dtype(X_all[field]):
                    min_val = float(np.nanmin(X_all[field]))
                    max_val = float(np.nanmax(X_all[field]))
                    default_val = float(input_defaults[field])

                    if field == "tenure":
                        user_input[field] = st.slider(field, int(min_val), int(max_val), int(default_val))
                    else:
                        user_input[field] = st.number_input(
                            field,
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val
                        )
                else:
                    if field == "SeniorCitizen":
                        user_input[field] = st.selectbox(field, ["No", "Yes"], index=0)
                    else:
                        options = sorted(X_all[field].dropna().astype(str).unique().tolist())
                        default_value = str(input_defaults[field])
                        default_index = options.index(default_value) if default_value in options else 0
                        user_input[field] = st.selectbox(field, options, index=default_index)

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        single_df = pd.DataFrame([user_input])
        pred, prob = get_prediction_outputs(model, single_df, model_info)

        pred_label = "Likely to Churn" if int(pred.iloc[0]) == 1 else "Likely to Stay"
        pred_prob = float(prob[0])
        pred_risk = risk_band(pred_prob)

        enriched_row = single_df.copy()
        enriched_row["RiskLevel"] = pred_risk
        reason_text = infer_reasons(enriched_row.iloc[0])
        action_text = recommend_action(enriched_row.iloc[0])

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction", pred_label)
        m2.metric("Churn Probability", f"{pred_prob:.2%}")
        m3.metric("Risk Level", pred_risk)

        if pred_risk == "High Risk":
            st.error("This customer is highly exposed to churn.")
        elif pred_risk == "Medium Risk":
            st.warning("This customer shows moderate churn risk.")
        else:
            st.success("This customer appears relatively stable.")

        st.markdown("#### Why is this customer at risk?")
        st.write(reason_text)

        st.markdown("#### Recommended Action")
        st.write(action_text)

# =========================================================
# TAB 5 - HIGH-RISK CUSTOMERS + BATCH PREDICTION
# =========================================================
with tab5:
    st.markdown("### High-Risk Customers")
    st.write("Prioritize retention campaigns by focusing on customers with the highest churn probability.")

    actionable_cols = [
        "customerID", "ChurnProbability", "RiskLevel", "MonthlyCharges",
        "Contract", "tenure", "PaymentMethod", "RiskReason", "RecommendedAction"
    ]
    actionable_cols = [c for c in actionable_cols if c in filtered.columns]

    high_risk_df = (
        filtered.loc[filtered["RiskLevel"] == "High Risk", actionable_cols]
        .sort_values("ChurnProbability", ascending=False)
        .copy()
    )

    if not high_risk_df.empty:
        high_risk_df["ChurnProbability"] = high_risk_df["ChurnProbability"].round(4)
        st.dataframe(high_risk_df, use_container_width=True, hide_index=True)

        csv = high_risk_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download High-Risk Customers CSV",
            data=csv,
            file_name="high_risk_customers.csv",
            mime="text/csv"
        )
    else:
        st.info("No high-risk customers in the current filter selection.")

    st.markdown("---")
    st.markdown("### Batch Prediction")
    st.write("Upload a CSV file with the same input columns used by the model to score customers in bulk.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        batch_features = prepare_features(batch_df, reference_features=feature_columns)

        try:
            batch_pred, batch_prob = get_prediction_outputs(model, batch_features, model_info)

            batch_result = batch_df.copy()
            batch_result["PredictedChurn"] = batch_pred
            batch_result["ChurnProbability"] = batch_prob
            batch_result["RiskLevel"] = batch_result["ChurnProbability"].apply(risk_band)
            batch_result["RiskReason"] = batch_result.apply(infer_reasons, axis=1)
            batch_result["RecommendedAction"] = batch_result.apply(recommend_action, axis=1)

            if "MonthlyCharges" in batch_result.columns:
                batch_at_risk = batch_result.loc[
                    batch_result["RiskLevel"] == "High Risk",
                    "MonthlyCharges"
                ].sum()
            else:
                batch_at_risk = 0

            st.success("Batch prediction completed successfully.")

            b1, b2, b3 = st.columns(3)
            b1.metric("Uploaded Customers", f"{len(batch_result):,}")
            b2.metric("Predicted to Churn", f"{int(batch_result['PredictedChurn'].sum()):,}")
            b3.metric("At-Risk Revenue", f"${batch_at_risk:,.2f}")

            st.dataframe(
                batch_result.sort_values("ChurnProbability", ascending=False),
                use_container_width=True,
                hide_index=True
            )

            batch_csv = batch_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Batch Prediction Results",
                data=batch_csv,
                file_name="batch_churn_predictions.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error("Batch prediction failed. Please make sure the uploaded CSV matches the model input structure.")
            st.code(str(e))
