"""This module contains the code for the Flask app that serves the model."""
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI


import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)


with open('/code/app/dict_vectorizer.pkl', 'rb') as f_in:
    dict_vectorizer = pickle.load(f_in)

app = FastAPI(title="Indicators of Heart Disease",
              openapi_tags=[
                  {
                      "name": "Health",
                      "description": "Get API health"
                  },
                  {
                      "name": "Prediction",
                      "description": "Model prediction"
                  }
              ])


@app.get(path='/', tags=['Health'])
def api_health():
    """
    A function that represents the health endpoint of the API.

    Returns:
        dict: A dictionary containing the status of the API, with the key
        "status" and the value "healthy".
    """
    return {"status": "healthy"}


@app.post('/predict', tags=['Prediction'])
def predict(request: dict):
    """Receives a POST request with a JSON payload, runs the model, and returns
    the prediction."""
    try:
        # Log the incoming request
        logging.debug(f"Received input data: {request}")
        
        # Check if request is a dictionary and has the expected keys
        if not isinstance(request, dict):
            raise HTTPException(status_code=400, detail="Request payload should be a JSON object")

        print(f"Received request: {request}")  # Debugging line
        
        input_data = request
        # df_input = pd.DataFrame.from_dict(input_data, orient='index').T
        df_input = pd.DataFrame([input_data])
        print(df_input)

        # For sample inputs, the target column is included in the input data.
        if "HadHeartAttack" in df_input.columns:
            df_input = df_input.drop(columns=["HadHeartAttack"])
            
        # Log the DataFrame before transformation
        logging.debug(f"Data before transformation: {df_input}")
            
        # Apply DictVectorizer to transform the input data
        transformed_data = dict_vectorizer.transform([input_data])

        # Log the transformed input
        logging.debug(f"Transformed input: {transformed_data}")

        # pred = model.predict(df_input)[0]
        pred = model.predict(transformed_data)[0]
        
        # Convert numpy types to native Python types (e.g., int or float)
        if isinstance(pred, np.int64):  # If prediction is a numpy int, convert it
            pred = int(pred)
        result = {'HadHeartAttack': pred}

        return result
    
    except Exception as e:
        # Log the error
        logging.error(f"Error during prediction: {e}")
        return {"error": f"Prediction failed: {str(e)}"}
