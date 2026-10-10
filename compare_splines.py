from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer


# Load the dataset.
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "lidar.dat"
data = pd.read_csv(data_path, sep=r"\s+")

X = data[["range"]].astype("float64")
y = data["logratio"].astype("float64")

# Recreate the original split. Do not evaluate the test set yet.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Identical validation folds for every candidate.
cv = KFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train))

candidates = [
    (
        "mean-baseline",
        DummyRegressor(strategy="mean"),
        {"model_type": "mean-baseline"},
    ),
    (
        "linear-regression",
        LinearRegression(),
        {"model_type": "linear-regression"},
    ),
]

for n_knots in [4, 6, 10]:
    for alpha in [0.01, 1.0]:
        estimator = make_pipeline(
            SplineTransformer(
                n_knots=n_knots,
                degree=3,
                include_bias=False,
            ),
            Ridge(alpha=alpha),
        )

        candidates.append((
            f"spline-knots{n_knots}-alpha{alpha}",
            estimator,
            {
                "model_type": "spline-ridge",
                "n_knots": n_knots,
                "degree": 3,
                "alpha": alpha,
            },
        ))

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lidar Regression")

trusted_spline_types = [
    "scipy.interpolate._bsplines.BSpline",
    "scipy.interpolate._bsplines._BSpline",
]

summary = []

for model_name, estimator, parameters in candidates:
    print(f"\nEvaluating: {model_name}", flush=True)

    with mlflow.start_run(run_name=f"{model_name}-cv"):
        mlflow.log_params({
            **parameters,
            "cv_folds": 5,
            "cv_seed": 42,
            "split_seed": 42,
            "test_size": 0.2,
            "n_train": len(X_train),
        })

        fold_results = []

        for fold_number, (fit_idx, val_idx) in enumerate(folds, start=1):
            model = clone(estimator)

            model.fit(
                X_train.iloc[fit_idx],
                y_train.iloc[fit_idx],
            )

            predictions = model.predict(X_train.iloc[val_idx])
            actual = y_train.iloc[val_idx]

            mae = float(mean_absolute_error(actual, predictions))
            rmse = float(np.sqrt(
                mean_squared_error(actual, predictions)
            ))

            fold_results.append({
                "fold": fold_number,
                "mae": mae,
                "rmse": rmse,
            })

            mlflow.log_metric("fold_mae", mae, step=fold_number)
            mlflow.log_metric("fold_rmse", rmse, step=fold_number)

        results = pd.DataFrame(fold_results)

        metrics = {
            "cv_mae_mean": float(results["mae"].mean()),
            "cv_rmse_mean": float(results["rmse"].mean()),
            "cv_rmse_std": float(results["rmse"].std(ddof=1)),
        }

        mlflow.log_metrics(metrics)
        mlflow.log_table(
            results,
            artifact_file="validation_folds.json",
        )

        # Fit the complete pipeline on the 176 training observations.
        fitted_model = clone(estimator)
        fitted_model.fit(X_train, y_train)

        training_predictions = fitted_model.predict(X_train)
        mlflow.log_metrics({
            "train_mae": float(
                mean_absolute_error(y_train, training_predictions)
            ),
            "train_rmse": float(np.sqrt(
                mean_squared_error(y_train, training_predictions)
            )),
        })

        # Explicitly select skops and authorize only the spline types.
        model_info = mlflow.sklearn.log_model(
            sk_model=fitted_model,
            name="model",
            input_example=X_train.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_SKOPS,
            skops_trusted_types=(
                trusted_spline_types
                if parameters["model_type"] == "spline-ridge"
                else []
            ),
        )

        # Verify that the saved model can be reloaded.
        restored_model = mlflow.sklearn.load_model(model_info.model_uri)
        restored_predictions = restored_model.predict(X_train.head(3))

        np.testing.assert_allclose(
            restored_predictions,
            fitted_model.predict(X_train.head(3)),
        )
        mlflow.set_tag("reload_check", "passed")

        # Plot only the training observations.
        x_grid = pd.DataFrame({
            "range": np.linspace(
                X_train["range"].min(),
                X_train["range"].max(),
                300,
            )
        })

        fig, ax = plt.subplots()
        ax.scatter(
            X_train["range"],
            y_train,
            s=15,
            alpha=0.6,
            label="Training data",
        )
        ax.plot(
            x_grid["range"],
            fitted_model.predict(x_grid),
            color="red",
            label=model_name,
        )
        ax.set_xlabel("range")
        ax.set_ylabel("logratio")
        ax.set_title(model_name)
        ax.legend()
        fig.tight_layout()

        mlflow.log_figure(fig, "plots/fitted_curve.png")
        plt.close(fig)

        summary.append({
            "model": model_name,
            **metrics,
            "model_uri": model_info.model_uri,
        })

ranking = pd.DataFrame(summary).sort_values("cv_rmse_mean")

print("\nModels ranked by validation RMSE:")
print(
    ranking.drop(columns="model_uri").to_string(
        index=False,
        float_format="%.6f",
    )
)

best = ranking.iloc[0]
print(f"\nBest candidate: {best['model']}")
print(f"Saved model URI: {best['model_uri']}")
print(f"Final test observations remain reserved: {len(X_test)}")
