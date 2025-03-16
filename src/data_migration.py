import os
import pandas as pd
import psycopg2
from sqlalchemy import create_engine

def get_db_connection():
    """
    Establishes a connection to the PostgreSQL database.

    The connection parameters (database name, user, password, host, and port) 
    are retrieved from environment variables. If environment variables are not set, 
    default values are used.

    Returns:
        psycopg2.connection: A connection object to interact with the PostgreSQL database.
    """
    return psycopg2.connect(
        dbname=os.getenv("POSTGRES_DB", "loan_db"),  # Database name
        user=os.getenv("POSTGRES_USER", "postgres"),  # Database username
        password=os.getenv("POSTGRES_PASSWORD", "password"),  # Database password
        host=os.getenv("POSTGRES_HOST", "localhost"),  # Database host (default: localhost)
        port=os.getenv("POSTGRES_PORT", "5432")  # Database port (default: 5432)
    )

def create_table():
    """
    Creates the `loan_data` table in the PostgreSQL database if it does not already exist.

    The table consists of the following columns:

    - `id`: Auto-incrementing primary key.
    - `person_age`: Age of the person (Float).
    - `person_gender`: Gender of the person (Categorical: e.g., Male, Female, Other).
    - `person_education`: Highest education level attained (Categorical).
    - `person_income`: Annual income of the person (Float).
    - `person_emp_exp`: Number of years of employment experience (Integer).
    - `person_home_ownership`: Home ownership status (Categorical: Rent, Own, Mortgage).
    - `loan_amnt`: Loan amount requested (Float).
    - `loan_intent`: Purpose of the loan (Categorical: e.g., Education, Medical, Personal).
    - `loan_int_rate`: Interest rate on the loan (Float).
    - `loan_percent_income`: Loan amount as a percentage of annual income (Float).
    - `cb_person_cred_hist_length`: Length of credit history in years (Float).
    - `credit_score`: Credit score of the applicant (Integer).
    - `previous_loan_defaults_on_file`: Indicator of previous loan defaults (Categorical: Yes/No).
    - `loan_status`: Target variable - Loan approval status (Integer: 1 = Approved, 0 = Rejected).

    This function ensures that the table is created before inserting data.
    """
    create_table_query = '''
    CREATE TABLE IF NOT EXISTS loan_data (
        id SERIAL PRIMARY KEY,  -- Unique ID for each record (auto-incrementing)
        person_age FLOAT,  -- Age of the applicant
        person_gender TEXT,  -- Gender of the applicant (Male/Female/Other)
        person_education TEXT,  -- Highest education level attained
        person_income FLOAT,  -- Annual income of the applicant
        person_emp_exp INT,  -- Years of employment experience
        person_home_ownership TEXT,  -- Home ownership status (Rent/Own/Mortgage)
        loan_amnt FLOAT,  -- Requested loan amount
        loan_intent TEXT,  -- Purpose of the loan (Education/Medical/Personal/etc.)
        loan_int_rate FLOAT,  -- Loan interest rate
        loan_percent_income FLOAT,  -- Loan amount as a percentage of annual income
        cb_person_cred_hist_length FLOAT,  -- Length of credit history in years
        credit_score INT,  -- Credit score of the applicant
        previous_loan_defaults_on_file TEXT,  -- Indicator of previous loan defaults (Yes/No)
        loan_status INT  -- Target variable: Loan approval status (1 = Approved, 0 = Rejected)
    );
    '''
    conn = get_db_connection()  # Establish database connection
    cur = conn.cursor()  # Create a cursor to execute SQL queries
    cur.execute(create_table_query)  # Execute the table creation query
    conn.commit()  # Commit the changes to the database
    cur.close()  # Close the cursor
    conn.close()  # Close the database connection

def migrate_data(csv_path):
    """
    Reads data from a CSV file and inserts it into the `loan_data` table in PostgreSQL.

    The function performs the following steps:
    1. Reads the CSV file into a Pandas DataFrame.
    2. Connects to the PostgreSQL database using SQLAlchemy.
    3. Uses `to_sql` to insert the data into the `loan_data` table.
    4. Replaces the existing data if the table already contains records.

    Args:
        csv_path (str): The file path of the CSV containing loan data.
    
    Note:
        - This function assumes that the CSV file's column names match the database table's columns.
        - `if_exists='replace'` ensures that the previous table data is replaced with new data.
    """
    df = pd.read_csv(csv_path)  # Load the CSV file into a DataFrame
    engine = create_engine("postgresql+psycopg2://postgres:password@localhost:5432/loan_db")  
    df.to_sql("loan_data", con=engine, if_exists='replace', index=False)  # Insert data into table
    print("Data migrated successfully!")  # Confirm successful migration

if __name__ == "__main__":
    create_table()  # Ensure the table exists before inserting data
    migrate_data("loan_data.csv")  # Load data from CSV and insert it into the table