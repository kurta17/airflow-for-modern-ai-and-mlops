import pandas as pd
from sklearn.datasets import load_iris


def get_features(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Get the features from the dataset.
    """
    features = dataset.copy()
    # Rename columns: replace (cm) and spaces
    features.rename(
        columns=lambda s: s.replace("(cm)", "").strip().replace(" ", "_"), inplace=True
    )

    # Uncomment to add features
    features['sepal_length_to_sepal_width'] = (
        features['sepal_length'] / features['sepal_width']
    )
    features['petal_length_to_petal_width'] = (
        features['petal_length'] / features['petal_width']
    )

    return features


def load_and_save_data():
    """Load the iris dataset and save as CSV."""
    # Load the iris dataset
    iris_data = load_iris(as_frame=True)
    print(f"Loaded iris dataset with target names: {list(iris_data.target_names)}")

    # Get the feature DataFrame from the iris dataset
    dataset = iris_data.frame
    features = get_features(dataset)
    
    # Create data directory if it doesn't exist
    import os
    data_dir = "/opt/airflow/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    csv_path = os.path.join(data_dir, "features_iris.csv")
    features.to_csv(csv_path, index=False)
    print(f"Saved features to {csv_path} with shape: {features.shape}")
    
    return features


if __name__ == "__main__":
    load_and_save_data()
