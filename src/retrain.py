import pandas as pd
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Database Configuration
DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/loan_db"
engine = create_engine(DATABASE_URL)

# MLflow Configuration
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"  # Local MLflow tracking
EXPERIMENT_NAME = "Loan_Approval_Model"

# File Path
THRESHOLD_PATH = "optimal_threshold.txt"

def fetch_data():
    """
    Fetches training data and new monitoring data from PostgreSQL.

    Returns:
        original_train (pd.DataFrame): The previous training dataset.
        monitoring_data (pd.DataFrame): The new data from monitoring.
    """
    try:
        # Load previous training data
        original_train = pd.read_csv("train_data.csv")
        print(f"✅ Loaded original training data: {original_train.shape[0]} samples.")

        # Fetch new monitoring data from PostgreSQL
        monitoring_data = pd.read_sql("SELECT * FROM monitoring", con=engine)
        print(f"✅ Fetched new monitoring data: {monitoring_data.shape[0]} samples.")

        # Drop old prediction column if it exists
        monitoring_data = monitoring_data.drop(columns=["prediction"], errors="ignore")

        return original_train, monitoring_data
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None, None

def limit_monitoring_data(original_train, monitoring_data, max_ratio=0.2):
    """
    Limits the new monitoring data to avoid overfitting.

    Args:
        original_train (pd.DataFrame): The old training dataset.
        monitoring_data (pd.DataFrame): The new data from monitoring.
        max_ratio (float): The maximum fraction of new data relative to the old dataset.

    Returns:
        pd.DataFrame: Limited monitoring data.
    """
    max_new_data_samples = int(len(original_train) * max_ratio)
    limited_data = monitoring_data.sample(n=min(len(monitoring_data), max_new_data_samples), random_state=42)

    print(f"✅ Limited monitoring data to {len(limited_data)} samples to prevent overfitting.")
    return limited_data

def preprocess_data(original_train, monitoring_data_limited):
    """
    Merges old and new data, removes unnecessary columns, and splits features/target.

    Args:
        original_train (pd.DataFrame): The original training dataset.
        monitoring_data_limited (pd.DataFrame): The new limited monitoring dataset.

    Returns:
        X_train, X_test, y_train, y_test: Processed and split training/testing sets.
        scaler (StandardScaler): Fitted scaler object.
    """
    # Merge datasets
    new_train_data = pd.concat([original_train, monitoring_data_limited], ignore_index=True)

    # Define features (X) and target (y)
    X = new_train_data.drop(columns=["id", "loan_status"], errors="ignore")
    y = new_train_data["loan_status"]

    # Remove datetime columns (if any)
    X = X.select_dtypes(exclude=["datetime64"])
    print("✅ Final features used for training:", X.columns.tolist())

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """
    Trains a Random Forest model.

    Args:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.

    Returns:
        RandomForestClassifier: Trained model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("✅ Model trained successfully.")
    return model

def compute_optimal_threshold(model, X_test, y_test):
    """
    Computes the optimal decision threshold using ROC Curve.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): True test labels.

    Returns:
        float: Optimal threshold for classification.
    """
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, probabilities)

        # Compute Youden's J statistic
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
    else:
        optimal_threshold = 0.5

    print(f"\n✅ Optimal Decision Threshold: {optimal_threshold:.4f}")

    # Save the optimal threshold
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(optimal_threshold))

    print(f"📁 Optimal Decision Threshold saved to '{THRESHOLD_PATH}'.")
    return optimal_threshold

def log_model_mlflow(model, X_test, y_test, scaler, optimal_threshold):
    """
    Logs the trained model and parameters to MLflow.

    Args:
        model (RandomForestClassifier): Trained model.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): True test labels.
        scaler (StandardScaler): Standardization object.
        optimal_threshold (float): Optimal decision threshold.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))

        mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name="LoanApprovalModel")

        # Save scaler as an artifact
        joblib.dump(scaler, "scaler.pkl")
        mlflow.log_artifact("scaler.pkl")

        # Log optimal threshold
        mlflow.log_param("optimal_threshold", optimal_threshold)

    print("✅ Model logged successfully to MLflow.")

def save_model_files(model, scaler, optimal_threshold):
    """
    Saves the trained model, scaler, and optimal threshold.

    Args:
        model (RandomForestClassifier): Trained model.
        scaler (StandardScaler): Standardization object.
        optimal_threshold (float): Optimal decision threshold.
    """
    joblib.dump(model, "loan_approval_model.pkl")
    joblib.dump(scaler, "scaler.pkl")

    with open("optimal_threshold.txt", "w") as f:
        f.write(str(optimal_threshold))

    print("✅ Model, scaler, and threshold saved successfully.")

def main():
    """
    Main execution pipeline:
    1. Fetch old and new data
    2. Limit monitoring data
    3. Preprocess and split datasets
    4. Train the model
    5. Compute the optimal threshold
    6. Log the model to MLflow
    7. Save the model locally
    """
    original_train, monitoring_data = fetch_data()
    if original_train is None or monitoring_data is None:
        return

    monitoring_data_limited = limit_monitoring_data(original_train, monitoring_data)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(original_train, monitoring_data_limited)

    model = train_model(X_train, y_train)
    optimal_threshold = compute_optimal_threshold(model, X_test, y_test)

    log_model_mlflow(model, X_test, y_test, scaler, optimal_threshold)
    save_model_files(model, scaler, optimal_threshold)

if __name__ == "__main__":
    main()