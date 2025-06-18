import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(test_size: float = 0.2):
    import os
    
    data_dir = "/opt/airflow/data"
    features_path = os.path.join(data_dir, "features_iris.csv")
    dataset = pd.read_csv(features_path)
    print(f"Loaded dataset with shape: {dataset.shape}")

    df_train, df_test = train_test_split(
        dataset, test_size=test_size, random_state=42
    )

    # Save the split datasets
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"Split dataset into:")
    print(f"  - Training set: {df_train.shape} -> {train_path}")
    print(f"  - Test set: {df_test.shape} -> {test_path}")
    
    return df_train, df_test


if __name__ == "__main__":
    split_dataset(test_size=0.2)
