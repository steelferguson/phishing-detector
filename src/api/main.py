from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from src.utils import add_ratio_features

# Load model
model = joblib.load('src/model_store/final_model.pkl')

# Define input schema
class InputFeatures(BaseModel):
    num_words: int
    num_unique_words: int
    num_stopwords: int
    num_links: int
    num_unique_domains: int
    num_email_addresses: int
    num_spelling_errors: int
    num_urgent_keywords: int

# Init app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/predict/")
def predict(features: InputFeatures):
    # Convert to DataFrame for consistent preprocessing
    input_df = pd.DataFrame([features.dict()])

    # Add ratio features
    input_df = add_ratio_features(input_df)

    # Reorder columns to match training set if needed
    # X = input_df[TRAINING_COLUMN_ORDER] if you saved it
    X = input_df.values  # assumes columns match training order

    pred_prob = model.predict_proba(X)[0][1]
    pred_class = int(pred_prob > 0.5)

    return {"prediction": pred_class, "probability": round(pred_prob, 4)}