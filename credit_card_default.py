# credit_card_default_streamlit.py
# Streamlit app: Credit Card Default Prediction
# Uses: streamlit, pandas, numpy, scikit-learn, sqlite3 (cursor), joblib
# Save this file and run: `streamlit run credit_card_default_streamlit.py`

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Constants / DB setup
# -----------------------------
DB_FILENAME = "credit_app.db"
MODEL_FILENAME = "credit_model.joblib"

# Ensure DB exists and a table for predictions is present
def init_db(db_filename=DB_FILENAME):
    conn = sqlite3.connect(db_filename)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            income REAL,
            age REAL,
            loan REAL,
            loan_to_income REAL,
            predicted_default INTEGER,
            proba_default REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    return conn, cur

# -----------------------------
# Synthetic data generator
# -----------------------------
def generate_synthetic_data(n_per_class=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    # Non-default customers (label 0)
    income_good = rng.normal(loc=70000, scale=20000, size=n_per_class)
    age_good = rng.normal(loc=40, scale=10, size=n_per_class)
    loan_good = rng.normal(loc=5000, scale=3000, size=n_per_class)

    # Default customers (label 1) — lower income, younger, higher relative loan
    income_bad = rng.normal(loc=30000, scale=15000, size=n_per_class)
    age_bad = rng.normal(loc=30, scale=8, size=n_per_class)
    loan_bad = rng.normal(loc=9000, scale=4000, size=n_per_class)

    income = np.concatenate([income_good, income_bad])
    age = np.concatenate([age_good, age_bad])
    loan = np.concatenate([loan_good, loan_bad])

    # Clean and bound values
    income = np.clip(income, 1000, None)
    age = np.clip(age, 18, 90)
    loan = np.clip(loan, 0, None)

    loan_to_income = loan / income

    default = np.array([0] * n_per_class + [1] * n_per_class)

    df = pd.DataFrame({
        "Income": income,
        "Age": age,
        "Loan": loan,
        "Loan_to_Income": loan_to_income,
        "Default": default,
    })
    # Shuffle
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df

# -----------------------------
# Model training / helpers
# -----------------------------
def train_model(df, features, target="Default"):
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion": confusion_matrix(y_test, y_pred)
    }

    return pipeline, metrics

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Credit Card Default Prediction", layout="wide")
st.title("💳 Credit Card Default Prediction")
st.markdown(
    "This app lets you generate or upload a dataset (2000 samples per class by default), train a logistic regression classifier, make single-sample predictions, and save results to a local SQLite DB using a cursor."
)

# Initialize DB connection and cursor
conn, cur = init_db()

# Sidebar: Data options
st.sidebar.header("Data / Model Options")
use_sample = st.sidebar.checkbox("Generate synthetic sample data", value=True)
uploaded_file = st.sidebar.file_uploader("Or upload CSV file (must contain columns Income, Age, Loan, Loan_to_Income, Default)")
if use_sample:
    df = generate_synthetic_data(n_per_class=2000)
else:
    df = None

if uploaded_file is not None:
    try:
        df_uploaded = pd.read_csv(uploaded_file)
        # Basic checks
        required_cols = {"Income", "Age", "Loan", "Loan_to_Income", "Default"}
        if not required_cols.issubset(set(df_uploaded.columns)):
            st.sidebar.error(f"Uploaded file is missing required columns: {required_cols - set(df_uploaded.columns)}")
        else:
            df = df_uploaded.copy()
            st.sidebar.success("CSV loaded — using uploaded data")
    except Exception as e:
        st.sidebar.error(f"Failed to read CSV: {e}")

if df is None:
    st.warning("No data available. Generate sample data or upload a CSV with required columns.")
    st.stop()

# Show data preview and basic EDA
st.subheader("Dataset preview & EDA")
col1, col2 = st.columns([2, 1])
with col1:
    st.dataframe(df.head(100))
with col2:
    st.write("Shape:")
    st.write(df.shape)
    st.write("Class balance:")
    st.write(df["Default"].value_counts())

st.write("Summary statistics:")
st.write(df.describe())

# Simple plots
st.subheader("Feature distributions")
fig, axes = plt.subplots(2, 2, figsize=(10, 6))
axes = axes.flatten()
for i, col in enumerate(["Income", "Age", "Loan", "Loan_to_Income"]):
    axes[i].hist(df[col], bins=30)
    axes[i].set_title(col)
plt.tight_layout()
st.pyplot(fig)

# Select features and target
st.sidebar.header("Training options")
feature_cols = st.sidebar.multiselect(
    "Select features to use", options=["Income", "Age", "Loan", "Loan_to_Income"],
    default=["Income", "Age", "Loan_to_Income"]
)
train_button = st.sidebar.button("Train model")

model = None
metrics = None

if train_button:
    if len(feature_cols) == 0:
        st.sidebar.error("Select at least one feature")
    else:
        with st.spinner("Training model..."):
            model, metrics = train_model(df, features=feature_cols)
            joblib.dump(model, MODEL_FILENAME)
        st.success("Model trained and saved to disk")

# If model file exists, load it
if model is None and os.path.exists(MODEL_FILENAME):
    try:
        model = joblib.load(MODEL_FILENAME)
        st.info("Loaded existing model from disk")
    except Exception:
        model = None

# Show metrics if available
if metrics is not None:
    st.subheader("Evaluation metrics (test set)")
    st.write(f"Accuracy: {metrics['accuracy']:.4f}")
    st.write(f"ROC AUC: {metrics['roc_auc']:.4f}")
    st.write("Confusion matrix:")
    st.write(metrics["confusion"])
    st.write("Classification report:")
    st.write(pd.DataFrame(metrics["report"]).transpose())

# Single-sample prediction UI
st.subheader("Make a single prediction")
with st.form(key="single_predict"):
    income_val = st.number_input("Income", value=float(df["Income"].median()))
    age_val = st.number_input("Age", value=float(df["Age"].median()))
    loan_val = st.number_input("Loan", value=float(df["Loan"].median()))
    loan_to_income_val = st.number_input("Loan_to_Income", value=float(loan_val / max(income_val, 1)))

    submit_pred = st.form_submit_button("Predict")

if submit_pred:
    if model is None:
        st.error("No trained model available. Train the model first.")
    else:
        X_new = pd.DataFrame([{"Income": income_val, "Age": age_val, "Loan": loan_val, "Loan_to_Income": loan_to_income_val}])
        # Ensure the model pipeline understands the columns we used
        try:
            proba = model.predict_proba(X_new[model.named_steps['scaler'].__class__.__name__ == model.named_steps['scaler'].__class__.__name__ if False else X_new.columns])
        except Exception:
            # Simpler: select feature columns known to the pipeline input
            try:
                features_for_model = feature_cols if feature_cols else [c for c in ["Income","Age","Loan","Loan_to_Income"] if c in X_new.columns]
                proba = model.predict_proba(X_new[features_for_model])
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                proba = None

        if proba is not None:
            pred_proba = float(proba[0, 1])
            pred_label = int(pred_proba >= 0.5)
            st.write(f"Predicted default: **{pred_label}** (probability {pred_proba:.3f})")
            # Save to DB using cursor
            try:
                insert_sql = "INSERT INTO predictions (income, age, loan, loan_to_income, predicted_default, proba_default) VALUES (?, ?, ?, ?, ?, ?)"
                cur.execute(insert_sql, (income_val, age_val, loan_val, loan_to_income_val, pred_label, pred_proba))
                conn.commit()
                st.success("Prediction saved to local SQLite DB (using cursor)")
            except Exception as e:
                st.error(f"Failed to save prediction to DB: {e}")

# Show saved predictions
st.subheader("Saved predictions (from SQLite DB)")
if st.button("Load saved predictions"):
    try:
        saved_df = pd.read_sql_query("SELECT * FROM predictions ORDER BY created_at DESC LIMIT 200", conn)
        st.dataframe(saved_df)
        csv = saved_df.to_csv(index=False)
        st.download_button("Download saved predictions CSV", csv, file_name="saved_predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Failed to load saved predictions: {e}")

# Optional: show raw DB file path
st.sidebar.markdown(f"**DB file:** `{os.path.abspath(DB_FILENAME)}`")

# Footer / cleanup
st.markdown("---")
st.caption("This is a demo. For real production usage: secure the DB, validate inputs, and use more robust models and feature engineering.")

# Close DB connection when Streamlit shuts down
# (Streamlit will keep the process running, explicit close not strictly necessary here)
