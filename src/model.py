import os
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import numpy as np
import pandas as pd
import skops.io as sio

from src.data_cleaning import KingCountyCleaner
from src.features import KingCountyFeatureEngineer


def train_and_save_model(data_path="data/King_County_House_prices_dataset.csv"):
  print("Loading dataset...")
  df = pd.read_csv(data_path)

  # 1. Run the cleaning and feature engineering pipeline on the raw data first
  print("Running data cleaning and feature engineering pipeline...")
  preprocessing_pipeline = Pipeline([
      ("cleaner", KingCountyCleaner()),
      ("engineer", KingCountyFeatureEngineer()),
  ])
  df_processed = preprocessing_pipeline.fit_transform(df)

  # 2. Separate features and target from the fully processed dataframe
  drop_lst = ["price", "sqft_price", "date", "delta_lat", "delta_long", "id"]
  X = df_processed.drop(
      columns=[col for col in drop_lst if col in df_processed.columns]
  )
  y = df_processed["price"]

  # 3. Train/Test Split
  print("Splitting data into train and test sets...")
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.3, random_state=42
  )

  # 4. Build the modeling pipeline (Polynomials -> Scaler -> ElasticNet)
  print("Building modeling pipeline...")
  modeling_pipeline = Pipeline([
      ("polynomial", PolynomialFeatures(2, include_bias=False)),
      ("scaler", StandardScaler()),
      (
          "model",
          TransformedTargetRegressor(
              regressor=ElasticNet(max_iter=5000, tol=1e-4, precompute=True),
              transformer=StandardScaler(),
          ),
      ),
  ])

  # 5. Hyperparameter tuning with GridSearchCV
  param_grid = {
      "model__regressor__alpha": [0.01, 0.1, 1],
      "model__regressor__l1_ratio": [0.2, 0.5, 0.8],
  }

  print("Running Grid Search (this may take a moment)...")
  grid_search = GridSearchCV(
      modeling_pipeline,
      param_grid,
      cv=5,
      scoring="r2",
      verbose=1,
      n_jobs=-1,
  )

  grid_search.fit(X_train, y_train)

  print(f"Best Parameters found: {grid_search.best_params_}")

  # 6. Evaluate on test set (R^2 and RMSE)
  best_model = grid_search.best_estimator_
  y_pred = best_model.predict(X_test)

  test_r2 = r2_score(y_test, y_pred)
  test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

  print(f"Test R^2 Score: {test_r2:.3f}")
  print(f"Test RMSE: ${test_rmse:,.2f}")

  # 7. Save the trained model pipeline
  os.makedirs("model", exist_ok=True)
  model_path = "model/model.bin"
  with open(model_path, "wb") as f_out:
    sio.dump(best_model, f_out)
  print(f"Model successfully saved to {model_path}")

  return best_model


if __name__ == "__main__":
  train_and_save_model()