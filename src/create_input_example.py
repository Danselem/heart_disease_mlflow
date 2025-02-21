"""Get a sample from train inputs and save as a dict to test the API."""
import json
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import pickle


def main():
    """Function to get a sample from train inputs and save as a dict."""
    # Sample data
    data_path = Path("data/processed/heart_train_cleaned.parquet")
    df_train = pd.read_parquet(data_path)
    df_train.drop(columns=['HadHeartAttack'], inplace=True)
    sample = df_train.sample(1).to_dict(orient="records")[0]
    
    # Load the DictVectorizer from the pickle file
    dictvect_path = Path('data/processed/dict_vectorizer.pkl')
    with open(dictvect_path, 'rb') as f_in:
        dict_vectorizer = pickle.load(f_in)
        
    # Transform the sample data using DictVectorizer
    sample_transformed = dict_vectorizer.transform([sample])

    # Convert to a list of feature names and their transformed values
    feature_names = dict_vectorizer.get_feature_names_out()
    transformed_dict = dict(zip(feature_names, sample_transformed[0]))

    # Save sample as a dict
    with open("sample_input.json", "w", encoding="utf-8") as file:
        json.dump(sample, file)
    
    # Save the transformed data as a JSON file
    with open("sample_input_transformed.json", "w", encoding="utf-8") as file:
        json.dump(transformed_dict, file)


if __name__ == "__main__":
    main()
