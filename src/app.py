from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import skops.io as sio

from src.data_cleaning import KingCountyCleaner
from src.features import KingCountyFeatureEngineer

app = FastAPI(title="King County Housing Price Predictor", version="0.1.0")

# Load the trained model pipeline on startup
MODEL_PATH = "model/model.bin"
try:
  with open(MODEL_PATH, "rb") as f:
    model = sio.load(f)
except Exception as e:
  model = None


# Define the expected input schema based on raw house features
class HouseInput(BaseModel):
  id: str | int | None = None
  date: str | None = None
  bedrooms: float
  bathrooms: float
  sqft_living: float
  sqft_lot: float
  floors: float
  waterfront: float | int | None = None
  view: float | int | None = None
  condition: int
  grade: int
  sqft_above: float
  sqft_basement: str | float
  yr_built: int
  yr_renovated: float | int | None = None
  zipcode: int
  lat: float
  long: float
  sqft_living15: float
  sqft_lot15: float


@app.get("/")
def read_root():
  return {"message": "Welcome to the King County Housing Price Prediction API!"}


@app.post("/predict")
def predict_price(house: HouseInput):
  if model is None:
    raise HTTPException(status_code=500, detail="Trained model not found.")

  try:
    # 1. Convert incoming JSON to a pandas DataFrame
    input_df = pd.DataFrame([house.model_dump()])

    # 2. Apply cleaning and feature engineering pipelines
    cleaner = KingCountyCleaner()
    engineer = KingCountyFeatureEngineer()

    df_cleaned = cleaner.fit_transform(input_df)
    df_processed = engineer.fit_transform(df_cleaned)

    # 3. Drop columns not used by the model
    drop_lst = ["price", "sqft_price", "date", "delta_lat", "delta_long", "id"]
    X = df_processed.drop(
        columns=[col for col in drop_lst if col in df_processed.columns]
    )

    # 4. Generate prediction
    prediction = model.predict(X)[0]

    return {
        "predicted_price_usd": round(float(prediction), 2),
        "status": "success",
    }

  except Exception as e:
    raise HTTPException(status_code=400, detail=str(e))