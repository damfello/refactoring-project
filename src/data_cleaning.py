import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class KingCountyCleaner(BaseEstimator, TransformerMixin):
  """Cleans the raw King County housing dataset."""

  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    df = X.copy()

    # 1. Drop the implausible 33-bedroom row if present
    if "bedrooms" in df.columns:
      # Find index or query out the 33-bedroom error safely
      df = df.query("bedrooms != 33")

    # 2. Fix sqft_basement containing '?' by recalculating it
    if "sqft_living" in df.columns and "sqft_above" in df.columns:
      df["sqft_basement"] = df["sqft_living"] - df["sqft_above"]

    # 3. Fill missing categorical/binary values
    if "view" in df.columns:
      df["view"] = df["view"].fillna(0)
    if "waterfront" in df.columns:
      df["waterfront"] = df["waterfront"].fillna(0)

    return df