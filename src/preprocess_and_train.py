import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, auc

# Database Configuration
DB_URL = "postgresql+psycopg2://postgres:password@localhost:5432/loan_db"

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local MLflow tracking database
EXPERIMENT_NAME = "Loan_Approval_Model"

# File Paths
MODEL_PATH = "loan_approval_model.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLD_PATH = "optimal_threshold.txt"

def fetch_data():
    """Fetches loan data from PostgreSQL and loads it into a Pandas DataFrame."""
    engine = create_engine(DB_URL)
    return pd.read_sql("SELECT * FROM loan_data", con=engine)

def preprocess_data(df):
    """Preprocesses the dataset: handling missing values and encoding categorical variables."""
    print("\nFirst few rows of the dataset:")
    print(df.head())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values per column:\n", missing_values[missing_values > 0])

    # Identify categorical columns and apply Label Encoding
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    print("\nCategorical Columns:", categorical_columns)

    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert to string before encoding
        label_encoders[col] = le

    # Separate last 1000 rows for API testing
    api_test_data = df.iloc[-1000:].copy()
    df = df.iloc[:-1000]  # Remove from training data

    # Save API test data
    api_test_data.to_csv("api_test_data.csv", index=False)
    print("\nAPI test data saved to api_test_data.csv")

    # Save processed training data
    df.to_csv("train_data.csv", index=False)
    print("Training data saved to train_data.csv")

    return df, api_test_data

def prepare_training_data(df):
    """Splits the dataset into training/testing sets and scales numerical features."""
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize numerical features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Trains a Random Forest model with hyperparameter tuning using GridSearchCV."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3]
    }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print("\nBest Model Parameters:", grid_search.best_params_)
    return grid_search.best_estimator_

def compute_optimal_threshold(model, X_test, y_test):
    """Computes the optimal decision threshold using the ROC curve (Youden’s J statistic)."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)
        roc_auc = auc(fpr, tpr)

        # Compute Youden's J statistic
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5

    print(f"\n✅ Optimal Decision Threshold: {optimal_threshold:.4f}")
    print(f"📉 AUC Score: {roc_auc:.4f}")

    # Save the optimal threshold
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(optimal_threshold))

    print(f"📁 Optimal Decision Threshold saved to '{THRESHOLD_PATH}'.")
    return optimal_threshold

def log_model_mlflow(model, scaler, X_test, y_test, optimal_threshold):
    """Logs model parameters, metrics, and artifacts to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))

        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="LoanApprovalModel")
        joblib.dump(scaler, SCALER_PATH)
        mlflow.log_artifact(SCALER_PATH)

        mlflow.log_param("optimal_threshold", optimal_threshold)

    print("✅ Model logged successfully to MLflow.")

def save_model(model, scaler):
    """Saves the trained model and scaler."""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("\nModel and scaler saved successfully.")

def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model on the test set and prints performance metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("\n📊 Model Performance Metrics:")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Precision: {precision:.4f}")
    print(f"✅ Recall: {recall:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Main Execution
if __name__ == "__main__":
    df = fetch_data()  # Load data from PostgreSQL
    df, api_test_data = preprocess_data(df)  # Preprocess data

    X_train, X_test, y_train, y_test, scaler = prepare_training_data(df)  # Prepare training data
    model = train_model(X_train, y_train)  # Train model

    optimal_threshold = compute_optimal_threshold(model, X_test, y_test)  # Compute threshold
    evaluate_model(model, X_test, y_test)  # Evaluate model performance
    log_model_mlflow(model, scaler, X_test, y_test, optimal_threshold)  # Log to MLflow
    save_model(model, scaler)  # Save trained model and scaler