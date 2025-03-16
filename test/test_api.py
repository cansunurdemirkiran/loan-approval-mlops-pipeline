import requests
import psycopg2
import pandas as pd

# API Endpoint Configuration
API_URL = "http://localhost:8000/predict"

# Database Configuration
DB_CONFIG = {
    "dbname": "loan_db",
    "user": "postgres",
    "password": "password",
    "host": "localhost",
    "port": 5432,
}

def load_test_data(file_path):
    """
    Loads test data from a CSV file.

    Args:
        file_path (str): Path to the test dataset CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame containing the test data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded {len(df)} test samples from '{file_path}'.")
        return df
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        return None

def send_api_requests(df):
    """
    Sends test samples to the API endpoint for predictions.

    Args:
        df (pd.DataFrame): Test dataset.

    Returns:
        int: Number of successful API responses.
    """
    success_count = 0
    total_samples = len(df)

    for _, row in df.iterrows():
        data = row.to_dict()
        try:
            response = requests.post(API_URL, json=data)

            if response.status_code == 200:
                success_count += 1
            else:
                print(f"❌ API Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Exception: {e}")

    print(f"✅ {success_count}/{total_samples} API Calls Successful")
    return success_count

def check_db_predictions():
    """
    Checks the number of predictions logged in the PostgreSQL database.

    Returns:
        int: Total number of predictions logged in the `monitoring` table.
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM monitoring;")
        row_count = cursor.fetchone()[0]

        conn.close()
        print(f"📌 Total Predictions Logged in DB: {row_count}")
        return row_count

    except psycopg2.Error as e:
        print(f"❌ Database Error: {e}")
        return None

def main():
    """
    Main function to execute the test pipeline:
    1. Load test data
    2. Send API requests
    3. Verify predictions are logged in the database
    """
    test_data = load_test_data("api_test_data.csv")

    if test_data is not None:
        send_api_requests(test_data)
        check_db_predictions()

if __name__ == "__main__":
    main()