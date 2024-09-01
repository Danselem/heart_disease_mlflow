"""Script to clean the dataset"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from pathlib import Path
import pickle
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
    
    # Apply label encoding
    label_encoder = LabelEncoder()
    data['HadHeartAttack'] = label_encoder.fit_transform(data['HadHeartAttack'])
    logger.info("Cleaned data shape: %s", data.shape)
    return data


def transform_and_save_data(data: pd.DataFrame, x_save_path: Path, 
                            y_save_path: Path, vec_save_path: Path) -> None:
    try:
        # Separate features and target variable
        x_train_heart = data.drop(columns=["HadHeartAttack"])
        y_train_heart = data["HadHeartAttack"]

        # Convert x_train_heart to list of dictionaries
        x_train_dict = x_train_heart.to_dict(orient='records')

        # Initialize and fit DictVectorizer
        vec = DictVectorizer(sparse=False)
        x_train_transformed = vec.fit_transform(x_train_dict)

        # Save transformed x and y variables
        
        np.save(x_save_path, x_train_transformed)
        
        logger.info(f"Saving y_train to {y_save_path}")
        np.save(y_save_path, y_train_heart.to_numpy())

        # Save DictVectorizer
        logger.info(f"Saving DictVectorizer to {vec_save_path}")
        with open(vec_save_path, 'wb') as f:
            pickle.dump(vec, f)
        
        logger.info("Data transformation and saving completed successfully.")

    except Exception as e:
        logger.error(f"An error occurred: {e}")


def main():
    """Main function to clean the dataset"""
    logger.info("Cleaning the training dataset")
    read_data_path = Path("data/interim/heart_train.parquet")
    data = pd.read_parquet(read_data_path)
    cleaned_data = clean_data(data)
    save_data_path = Path("data/processed/heart_train_cleaned.parquet")
    cleaned_data.to_parquet(save_data_path, index=False)
    logger.info("Cleaned data saved to 'data/processed/heart_train_cleaned.parquet'")
    transform_and_save_data(
    data=cleaned_data,
    x_save_path=Path("data/processed/x_train_transformed.npy"),
    y_save_path=Path("data/processed/y_train.npy"),
    vec_save_path=Path("data/processed/dict_vectorizer.pkl")
)


if __name__ == "__main__":
    main()
