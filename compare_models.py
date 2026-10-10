from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split

# Load the data.
project_dir = Path(__file__).resolve().parent
data = pd.read_csv(project_dir / "data" / "lidar.dat", sep=r"\s+")

X = data[["range"]].astype("float64")
y = data["logratio"]

# Recreate exactly the previous split.
# The final test data will not be evaluated here.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use identical validation folds for both models.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train))

models = {
    "mean-baseline": DummyRegressor(strategy="mean"),
    "linear-regression": LinearRegression(),
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lidar Regression")

summary = []

for model_name, estimator in models.items():
    with mlflow.start_run(run_name=f"{model_name}-cv"):
        mlflow.log_params({
            "model_type": model_name,
            "cv_folds": 5,
            "cv_seed": 42,
            "split_seed": 42,
            "test_size": 0.2,
            "n_train": len(X_train),
        })

        fold_results = []

        for fold_number, (fit_idx, val_idx) in enumerate(folds, start=1):
            # A fresh model is trained for each fold.
            model = clone(estimator)

            model.fit(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx],
            )

            predictions = model.predict(X_train.iloc[val_idx])
            actual = y_train.iloc[val_idx]

            mae = mean_absolute_error(actual, predictions)
            rmse = np.sqrt(mean_squared_error(actual, predictions))

            fold_results.append({
                "fold": fold_number,
                "mae": float(mae),
                "rmse": float(rmse),
            })

            mlflow.log_metric("fold_mae", float(mae), step=fold_number)
            mlflow.log_metric("fold_rmse", float(rmse), step=fold_number)

        results = pd.DataFrame(fold_results)

        metrics = {
            "cv_mae_mean": float(results["mae"].mean()),
            "cv_rmse_mean": float(results["rmse"].mean()),
            "cv_rmse_std": float(results["rmse"].std(ddof=1)),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_table(results, artifact_file="validation_folds.json")

        summary.append({"model": model_name, **metrics})

print("\nCross-validation results:")
print(pd.DataFrame(summary).to_string(index=False, float_format="%.6f"))
print(f"\nFinal test observations remain reserved: {len(X_test)}")
