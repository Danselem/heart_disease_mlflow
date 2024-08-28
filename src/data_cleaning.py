"""Script to clean the dataset"""
import pandas as pd
from src import logger


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataset by removing missing values and duplicates

    Args:
        data (pd.DataFrame): The dataset to be cleaned

    Returns:
        pd.DataFrame: The cleaned dataset
    """
    logger.info("Original data shape: %s", data.shape)
    data = data.dropna()
    data = data.drop_duplicates()
    logger.info("Cleaned data shape: %s", data.shape)
    return data


def main():
    """Main function to clean the dataset"""
    logger.info("Cleaning the training dataset")
    data = pd.read_parquet("data/interim/heart_train.parquet")
    cleaned_data = clean_data(data)
    cleaned_data.to_parquet("data/processed/heart_train_cleaned.parquet", index=False)
    logger.info("Cleaned data saved to 'data/processed/heart_train_cleaned.parquet'")


if __name__ == "__main__":
    main()
