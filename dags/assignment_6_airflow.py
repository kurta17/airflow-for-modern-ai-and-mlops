from datetime import datetime
from airflow import DAG
from airflow.decorators import task
import sys
import os

# Add the dags directory to Python path so we can import from src
sys.path.append(os.path.dirname(__file__))

from src.load_data import load_and_save_data
from src.split_dataset import split_dataset

with DAG(
    dag_id="assignment-6-airflow",
    start_date=datetime(2025, 1, 1),
    schedule="* * * * *",  # every minute
    catchup=False,
    tags=["assignment", "mlops", "iris"]
) as dag:

    @task()
    def task_load_data():
        features = load_and_save_data()
        return f"Loaded and saved {features.shape[0]} rows with {features.shape[1]} columns"

    @task()
    def task_split_data():
        train_df, test_df = split_dataset(test_size=0.2)
        return f"Split into train ({train_df.shape[0]} rows) and test ({test_df.shape[0]} rows)"

    # Set dependencies
    task_load_data() >> task_split_data()
