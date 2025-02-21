import json
import pickle
import numpy as np
from pathlib import Path

# Load the model (assuming it's saved as 'model.pkl')
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the DictVectorizer (assuming it's saved as 'dict_vectorizer.pkl')
with open('data/processed/dict_vectorizer.pkl', 'rb') as vec_file:
    dict_vectorizer = pickle.load(vec_file)

# Load the JSON data from file (make sure your file is in the correct format)
sample_path = Path("sample_input.json")
with open(sample_path, "r", encoding="utf-8") as file:
    sample_data = json.load(file)

# Check if the sample data is loaded correctly
print("Loaded input data:", sample_data)

# Transform the input data using DictVectorizer
# Convert the data into a list of dictionaries (the DictVectorizer expects a list of dicts)
transformed_data = dict_vectorizer.transform([sample_data])

# Make a prediction with the loaded model
prediction = model.predict(transformed_data)[0]


# Print the prediction result
print("Prediction result:", prediction)
