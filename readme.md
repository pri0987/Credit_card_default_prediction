# Credit_card_default_prediction
A Streamlit web application that predicts whether a credit-card customer will default on payments based on income, age, loan amount, and loan-to-income ratio.
It supports synthetic data generation, custom CSV uploads, model training, interactive predictions, and stores results in a local SQLite database.

✨ Features:

Interactive UI built with Streamlit.

Synthetic dataset generator (2,000 samples per class by default).

Upload your own CSV (columns: Income, Age, Loan, Loan_to_Income, Default).

Logistic Regression model with a StandardScaler preprocessing pipeline.

Displays:
     
     ->Accuracy, ROC-AUC, confusion matrix, classification report
     ->Histograms of feature distributions

Single-sample prediction form with real-time probability output.

SQLite database to store all predictions (using sqlite3 cursor).

One-click CSV download of saved predictions.
Beautiful finance-themed gradient background.

🗂️ Project Structure

credit-card-default/
├─ credit_card_default_streamlit.py   # Main Streamlit app
├─ credit_card_default_dataset.csv    # (Optional) Example dataset
├─ credit_app.db                      # SQLite DB (created after first run)
├─ credit_model.joblib                # Saved model (after training)

🛠️ How to Use

1.Data

    ->Leave “Generate synthetic sample data” checked for auto-generated data, or
    ->Upload your own CSV with required columns:
      Income, Age, Loan, Loan_to_Income, Default.

2.Train Model

     ->Pick features in the sidebar.
     ->Click Train model to build and save a logistic regression model.

3.Predict

     ->Enter a single customer’s details.
     ->Click Predict to see the probability of default and save to the DB.

4.View Saved Predictions

     ->Press Load saved predictions to view or download past predictions.

🧩 Key Files

1.credit_card_default_streamlit.py – Full Streamlit app with:

     ->Data generation / EDA
     ->Model training & evaluation
     ->Single prediction form
     ->SQLite persistence

2.credit_card_default_dataset.csv – Optional pre-generated dataset of 4,000 rows (2,000 default / 2,000 non-default).
