import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class KingCountyFeatureEngineer(BaseEstimator, TransformerMixin):
  """Engineers new features for the King County dataset."""

  def __init__(self):
    pass

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    df = X.copy()

    # 1. Create last_known_change from yr_built and yr_renovated
    if "yr_built" in df.columns and "yr_renovated" in df.columns:
      last_known_change = []
      for idx, yr_re in df["yr_renovated"].items():
        if pd.isna(yr_re) or yr_re == 0.0:
          last_known_change.append(df.loc[idx, "yr_built"])
        else:
          last_known_change.append(int(yr_re))
      df["last_known_change"] = last_known_change
      df = df.drop(columns=["yr_renovated", "yr_built"], errors="ignore")

    # 2. Distance to the wealth center
    if "lat" in df.columns and "long" in df.columns:
      ref_lat, ref_long = 47.62774, -122.24194
      delta_lat = np.absolute(ref_lat - df["lat"])
      delta_long = np.absolute(ref_long - df["long"])
      df["center_distance"] = (
          (delta_long * np.cos(np.radians(47.6219))) ** 2 + delta_lat ** 2
      ) ** (1 / 2) * 2 * np.pi * 6378 / 360

    # 3. Distance to waterfront houses
    if "waterfront" in df.columns and "lat" in df.columns and "long" in df.columns:
      water_list = df[df["waterfront"] == 1]
      if not water_list.empty:
        water_distances = []
        for idx, row in df.iterrows():
          dlat = row["lat"] - water_list["lat"]
          dlong = (row["long"] - water_list["long"]) * np.cos(
              np.radians(water_list["lat"])
          )
          dists = ((dlong**2 + dlat**2) ** (1 / 2) * 2 * np.pi * 6378 / 360)
          water_distances.append(dists.min())
        df["water_distance"] = water_distances
      else:
        df["water_distance"] = 0.0

    return df