"""Script to split the Indicators of Heart Disease dataset into training and
testing sets."""
import uuid
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from src import logger


def load_data():
    """Load the Indicators of Heart Disease dataset with progress bar."""
    logger.info("Loading the dataset")
    data_path = Path('data/raw/heart_2022_with_nans.csv.zip')

    with tqdm(total=1, desc="Loading Data", unit="file") as pbar:
        data = pd.read_csv(data_path)
        pbar.update(1)

    logger.info("Dataset loaded successfully")
    return data

def split_data(data: pd.DataFrame):
    """Split the data into training and testing sets"""
    logger.info("Splitting the data into training and testing sets")
    df_train, df_test = train_test_split(data, test_size=0.2, random_state=2506)
    logger.info("Data split successfully")
    logger.info(f"Training set shape: {df_train.shape}")
    logger.info(f"Testing set shape: {df_test.shape}")
    return df_train, df_test


def main():
    """Main function to split the data into training and testing sets"""
    logger.info("Starting data split process")
    data = load_data()
    logger.info("Generate UID")
    data["uuid"] = [uuid.uuid4() for x in range(len(data))]
    data["uuid"] = data["uuid"].astype("str")
    # Reorder columns to make `uuid` the first column
    columns_order = (["uuid"] + 
                    [col for col in data.columns if col not in ["uuid", "HadHeartAttack"]] + 
                    ["HadHeartAttack"])
    data = data[columns_order]
    df_train, df_test = split_data(data)
    df_train.to_parquet('data/interim/heart_train.parquet', index=False)
    df_test.to_parquet('data/interim/heart_test.parquet', index=False)
    logger.info("Data saved to parquet files successfully")


if __name__ == '__main__':
    main()
