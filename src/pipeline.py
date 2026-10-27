import pandas as pd
from sklearn.pipeline import Pipeline
from src.data_cleaning import KingCountyCleaner
from src.features import KingCountyFeatureEngineer

# 1. Build the pipeline combining cleaning and feature engineering
preprocessing_pipeline = Pipeline([
    ("data_cleaner", KingCountyCleaner()),
    ("feature_engineer", KingCountyFeatureEngineer()),
])

if __name__ == "__main__":
  # 2. Load raw data to test the pipeline
  df_raw = pd.read_csv("data/King_County_House_prices_dataset.csv")

  # 3. Run it through the reusable pipeline!
  df_processed = preprocessing_pipeline.fit_transform(df_raw)

  print("Raw shape:", df_raw.shape)
  print("Processed shape:", df_processed.shape)
  print("Pipeline executed successfully!")