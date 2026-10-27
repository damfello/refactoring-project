from fastapi.testclient import TestClient
import pandas as pd
from src.app import app
from src.data_cleaning import KingCountyCleaner
from src.features import KingCountyFeatureEngineer

client = TestClient(app)


def test_cleaner_drops_outlier():
  # Create a dummy dataframe containing the 33-bedroom outlier row
  data = {
      "id": [1, 2],
      "bedrooms": [3, 33],
      "bathrooms": [2.0, 2.0],
      "sqft_living": [1500, 1620],
      "sqft_lot": [5000, 5000],
      "floors": [1.0, 1.0],
      "waterfront": [0, 0],
      "view": [0, 0],
      "condition": [3, 3],
      "grade": [7, 7],
      "sqft_above": [1500, 1620],
      "sqft_basement": [0, 0],
      "yr_built": [1990, 1990],
      "yr_renovated": [0, 0],
      "zipcode": [98001, 98001],
      "lat": [47.5, 47.5],
      "long": [-122.2, -122.2],
      "sqft_living15": [1500, 1500],
      "sqft_lot15": [5000, 5000],
      "price": [300000, 400000],
  }
  df = pd.DataFrame(data)

  cleaner = KingCountyCleaner()
  df_cleaned = cleaner.fit_transform(df)

  # Verify the 33-bedroom row (index 1 / id 2) was dropped
  assert len(df_cleaned) == 1
  assert 33 not in df_cleaned["bedrooms"].values


def test_root_endpoint():
  response = client.get("/")
  assert response.status_code == 200
  assert (
      response.json()["message"]
      == "Welcome to the King County Housing Price Prediction API!"
  )


def test_predict_endpoint():
  payload = {
      "id": 12345,
      "date": "2014515T000000",
      "bedrooms": 3,
      "bathrooms": 2.0,
      "sqft_living": 1800,
      "sqft_lot": 7500,
      "floors": 1.0,
      "waterfront": 0,
      "view": 0,
      "condition": 3,
      "grade": 7,
      "sqft_above": 1800,
      "sqft_basement": 0,
      "yr_built": 1985,
      "yr_renovated": 0,
      "zipcode": 98001,
      "lat": 47.3638,
      "long": -122.193,
      "sqft_living15": 1700,
      "sqft_lot15": 7500,
  }

  response = client.post("/predict", json=payload)
  assert response.status_code == 200
  data = response.json()
  assert data["status"] == "success"
  assert "predicted_price_usd" in data
  assert isinstance(data["predicted_price_usd"], float)