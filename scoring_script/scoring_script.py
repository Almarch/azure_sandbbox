import json
import os
import pandas as pd
from mlflow.pyfunc import load_model

model = None

def init():
    """
    Called once when the web service starts. Loads the MLflow Pyfunc model.
    """
    global model

    model_path = os.getenv("AZUREML_MODEL_DIR")

    try:
        model = load_model(model_path)
        print("Pyfunc model loaded successfully.")
    except Exception as e:
        print(f"Error loading Pyfunc model: {e}")

def run(raw_data):
    """
    Called for every HTTP request to the web service.
    
    Args:
        raw_data (str): Raw JSON data sent by the user.
    Returns:
        list: Predictions formatted as JSON.
    """
    
    try:
        # Deserialize input data (JSON to Python object)
        data = json.loads(raw_data)
        
        # Convert to Pandas DataFrame (required by Pyfunc convention)
        input_df = pd.DataFrame(data)
        
        # Run inference (handles scaling, prediction, and label mapping)
        predictions = model.predict(input_df)
        
        # Serialize results back to JSON
        return json.dumps(predictions)

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})