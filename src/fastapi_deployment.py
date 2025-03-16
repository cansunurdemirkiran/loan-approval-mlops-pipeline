import os
import time
import logging
import subprocess
import datetime
import joblib
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sqlalchemy import create_engine, Column, Integer, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import Response
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from pydantic import BaseModel

# ✅ Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Database Configuration
DATABASE_URL = "postgresql+psycopg2://postgres:password@localhost:5432/loan_db"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Monitoring Table in PostgreSQL
class PredictionLog(Base):
    __tablename__ = "monitoring"
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    person_age = Column(Integer)
    person_gender = Column(Integer)
    person_education = Column(Integer)
    person_income = Column(Float)
    person_emp_exp = Column(Float)
    person_home_ownership = Column(Integer)
    loan_amnt = Column(Float)
    loan_intent = Column(Integer)
    loan_int_rate = Column(Float)
    loan_percent_income = Column(Float)
    cb_person_cred_hist_length = Column(Integer)
    credit_score = Column(Float)
    previous_loan_defaults_on_file = Column(Integer)
    loan_status = Column(Integer)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

# ✅ Create the database table if not exists
Base.metadata.create_all(bind=engine)

# ✅ Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Define API Request Body Schema
class LoanApplication(BaseModel):
    person_age: int
    person_gender: int
    person_education: int
    person_income: float
    person_emp_exp: float
    person_home_ownership: int
    loan_amnt: float
    loan_intent: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    credit_score: float
    previous_loan_defaults_on_file: int 

# ✅ Model Paths
MODEL_PATH = "loan_approval_model.pkl"
SCALER_PATH = "scaler.pkl"
THRESHOLD_PATH = "optimal_threshold.txt"

# ✅ Drift Threshold (Jensen-Shannon Divergence)
DRIFT_THRESHOLD = 0.1  

# ✅ Load Model and Threshold
model = None
scaler = None
APPROVAL_THRESHOLD = None

def load_latest_model():
    """Loads the latest model, scaler, and decision threshold."""
    global model, scaler, APPROVAL_THRESHOLD

    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)

        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH, "r") as f:
                APPROVAL_THRESHOLD = float(f.read().strip())
        else:
            APPROVAL_THRESHOLD = 0.5  # Default threshold if file is missing

        logging.info(f"📌 Model Loaded Successfully with Threshold: {APPROVAL_THRESHOLD:.4f}")

    except Exception as e:
        logging.error(f"❌ Model Loading Failed: {str(e)}")

# ✅ Load model at startup
load_latest_model()

# ✅ Watch for Model Changes Using Watchdog
class ModelFileChangeHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.src_path in [MODEL_PATH, SCALER_PATH, THRESHOLD_PATH]:
            logging.info(f"🔄 Detected Model Change in {event.src_path}, Reloading...")
            load_latest_model()

observer = Observer()
observer.schedule(ModelFileChangeHandler(), path=".", recursive=False)
observer.start()

# ✅ Initialize FastAPI App
app = FastAPI()

# ✅ Prometheus Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total number of requests")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds")

@app.get("/metrics")
def get_metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health")
def health_check():
    """Health Check API."""
    return {"status": "API is up and running"}

@app.post("/predict")
def predict_loan_approval(application: LoanApplication, db: Session = Depends(get_db)):
    """Predict loan approval status and log request to database."""
    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([application.dict().values()], columns=application.dict().keys())

        # Scale Input Data
        scaled_data = scaler.transform(input_data)

        # Get Prediction Probability
        approval_prob = model.predict_proba(scaled_data)[0][1] if hasattr(model, "predict_proba") else model.predict(scaled_data)[0]

        # Apply Decision Threshold
        prediction = int(approval_prob > APPROVAL_THRESHOLD)

        # Log Prediction to Database
        db.add(PredictionLog(**application.dict(), loan_status=prediction))
        db.commit()

        latency = time.time() - start_time
        REQUEST_LATENCY.observe(latency)

        return {"loan_approval_status": prediction, "approval_probability": approval_prob}

    except Exception as e:
        logging.error(f"❌ Prediction Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/data_drift")
def check_data_drift(db: Session = Depends(get_db)):
    """Detects data drift using Jensen-Shannon Divergence."""
    try:
        monitoring_df = pd.read_sql("SELECT * FROM monitoring", con=engine)
        training_df = pd.read_csv("train_data.csv")

        if monitoring_df.empty:
            return {"status": "No monitoring data available."}

        drift_results = {}
        retraining_required = False

        for col in training_df.columns:
            if col in monitoring_df.columns:
                # Compute Probability Distributions
                p = training_df[col].value_counts(normalize=True).sort_index()
                q = monitoring_df[col].value_counts(normalize=True).sort_index()

                # Align Distributions
                common_index = p.index.union(q.index)
                p, q = p.reindex(common_index, fill_value=0), q.reindex(common_index, fill_value=0)

                # Compute Jensen-Shannon Distance
                js_divergence = jensenshannon(p.to_numpy(), q.to_numpy())
                drift_results[col] = js_divergence

                # Trigger Retraining if Threshold Exceeded
                if js_divergence > DRIFT_THRESHOLD:
                    logging.warning(f"⚠️ Data Drift in {col}: {js_divergence:.4f}")
                    retraining_required = True

        if retraining_required:
            logging.warning("🚀 Significant Data Drift Detected! Initiating Retraining...")
            subprocess.Popen(["python3", "retrain.py"])

        return {"drift_analysis": drift_results, "retraining_triggered": retraining_required}

    except Exception as e:
        logging.error(f"❌ Data Drift Analysis Failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
