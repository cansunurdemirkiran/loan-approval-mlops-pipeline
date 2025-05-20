# Loan Approval Prediction System

This project is an **end-to-end MLOps pipeline** designed for **loan approval classification** using state-of-the-art machine learning and automation techniques. It integrates:
- **Data Migration** with **PostgreSQL & Docker Compose**
- **Preprocessing & Model Training** with **scikit-learn**
- **Model Development** for **FastAPI**
- **Model Versioning** using **MLflow**
- **Monitoring & Logging** with **Prometheus & FastAPI**
- **Drift Detection** using **FastAPI** and  **Automated Model Retraining** with **Subprocess** when drift is detected.
- **Regular Model Retraining** with **Cron**
- **System Validation** with **API Requests**
  

## Project Structure

```

loan-approval-system

├──  data

│ ├── loan_data.csv # Project data

│ ├── api_test_data.csv # API test data

│ ├── train_data.csv # Processed training data

├──  models

│ ├── loan_approval_model.pkl # Trained model

│ ├── scaler.pkl # Scaler for preprocessing

│ ├── optimal_threshold.txt # Decision threshold

├──  src

│ ├── data_migration.py # Migrate data from CSV to PostgreSQL

│ ├── preprocess_and_train.py # Preprocessing & Model training script

│ ├── fastapi_deployment.py # FastAPI server for predictions

│ ├── retrain.py # Model retraining script

├──  test

│ ├── test_api.py # API testing script

├──  requirements.txt # Dependencies

├──  docker-compose.yml # Docker Compose configuration

├──  README.md # Documentation

├──  .gitignore # Files to ignore in Git

```
  

## Setup Instructions


### 1. Install Dependencies

Make sure you have Python installed, then run:

```bash

pip  install  -r  requirements.txt

```

  

### 2. Run Services with Docker Compose

Ensure Docker and Docker Compose are installed, then run:

```bash

docker-compose  up  -d

```

This will start PostgreSQL and pgAdmin services.

  

### 3. Migrate Data

Move data from CSV to PostgreSQL:

```bash

python3  src/data_migration.py

```

- Creates database connection.

- Inserts the CSV file into the `loan_data` table in PostgreSQL.

  

### 4. Preprocess & Train the Model

Apply preproccessing and train the RandomForest model and save the artifacts:

```bash

python3  src/preprocess_and_train.py

```

- Loads the loan dataset from PostgreSQL (loan_data.csv).

- Preprocesses the dataset:

- Apply Exploratory Data Analysis.

- Handles missing values.

- Encodes categorical features.

- Applies feature scaling.

- Splits data into training & test sets (train_test_split).

- Trains a RandomForestClassifier using Scikit-learn.

- Evaluates model performance (e.g., accuracy, precision, recall).

- Determines the optimal threshold from the ROC curve.

- Logs the model, metrics, and artifacts in MLflow

- Saves trained model & scaler locally (loan_approval_model.pkl, scaler.pkl).

  

### 5. Run the FastAPI Server

Deploy the API for loan prediction:

```bash

uvicorn  fastapi_deployment:app  --reload

```

- GET /health → Checks if the API is running.

- POST /predict → Accepts user input, scales data, and returns loan approval prediction.

- GET /monitoring/data_drift → Checks for data drift using Jensen-Shannon Divergence and triggers the retrain.py.

- GET /metrics → Exposes Prometheus monitoring metrics.

  

### 6. Test the API

Run API tests:

```bash

python3  src/test_api.py

```

- Loads 1,000 test samples (api_test_data.csv which created from the last 1000 line of the loan_data.csv).

- Sends API requests to /predict using http.

- Checks that responses are valid (e.g., correct format, status codes).

- Verifies that predictions are logged in PostgreSQL (monitoring table).

- Evaluates API performance:

- Response time

- Accuracy vs. ground truth

- Generates a test report (pass/fail for each request).

  

### 7. Model Retraining (Triggered from '/monitoring/data_drift' endpoint)

Trigger retraining if data drift is detected:

```bash

python3  src/retrain.py

```

- Connects to PostgreSQL and fetches new monitoring data.

- Compares feature distributions between new & old data using Jensen-Shannon Divergence.

- If drift is detected (divergence > threshold), it:

- Fetches new data from the monitoring table.

- Merges new + old data for retraining.

- Runs data preprocessing pipeline.

- Retrains the model and registers a new version in MLflow.

- Overwrites the API model (loan_approval_model.pkl).

- Triggers API update to use the new model.

  

## API Endpoints

| Endpoint | Description |
|--|--|
| `/health` | Health check with **GET** Method |
| `/metrics` | Prometheus metrics with **GET** Method |
|  `/predict` | Predict loan approval with **POST** Method|
| `/monitoring/data_drift` | Check data drift with **GET** Method|



## Automatically Overwrite the Model in FastAPI

After retraining, **FastAPI automatically reloads the latest model** from the filesystem. 

1. When `retrain.py` detects drift, it **saves the new model** (`loan_approval_model.pkl`).
2. **FastAPI's monitoring script** detects the file update and reloads it.
3. The **API continues serving predictions** using the updated model **without downtime**.

 **This ensures the latest version of the model is always in production.**

## Schedule Auto-Retraining (Cron)

To schedule automatic model retraining, **use a cron job**.

1. Open the cron scheduler:
   ```bash
   crontab -e
   ```

2. Add the following line to **run retraining every day at 2 AM**:
   ```bash
   0 2 * * * /usr/bin/python3 /path/to/retrain.py
   ```


##  Web-Based Tools for API, Model Tracking & Database Management

### **1. FastAPI - API Documentation & Testing**

FastAPI provides an **interactive Swagger UI** for testing the **loan approval API**.
1. Open [http://localhost:8000/docs](http://localhost:8000/docs) in your browser.
2. You will see a list of API endpoints.
3. Click on **`/predict`** → Enter loan applicant details.
4. Click **Execute** → See the prediction result.

**Runs when `fastapi_deployment.py` is executed.**

---
### **2. MLflow UI - Model Tracking & Versioning**

MLflow is used for **tracking machine learning experiments** and **model versioning**.

1. Start MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
2. Open [http://127.0.0.1:5000/#/experiments/](http://127.0.0.1:5000/#/experiments/).
3. Click on **"LoanApprovalModel"** to see:
   - Training **metrics & parameters**.
   - Model **artifacts (`loan_approval_model.pkl`)**.
   - Different **model versions**.

**Runs when `train_model.py` logs a model in MLflow.**

---
### **3. pgAdmin - PostgreSQL Database Management**

pgAdmin is a **web-based GUI** for managing **PostgreSQL databases**.

1. Open [http://localhost:5050/browser/](http://localhost:5050/browser/).
2. Login:
   - **Email:** `admin@example.com`
   - **Password:** `admin`
3. Connect to PostgreSQL:
   - **Host:** `loan_postgres`
   - **Database:** `loan_db`
4. Run queries:
   ```sql
   SELECT * FROM loan_data LIMIT 10;
   ```

**Runs when `docker-compose up -d` starts PostgreSQL.**
  

## Technologies Used

-  **Python** (FastAPI, Pandas, Scikit-Learn, MLflow)

-  **PostgreSQL** (Database)

-  **Docker & Docker Compose** (For containerized deployment)

-  **Prometheus** (Monitoring)

- **Cron Jobs** (Automated retraining pipeline)


## License

This project is open-source and available under the MIT License.
