from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Locate the data relative to this script.
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "lidar.dat"

data = pd.read_csv(data_path, sep=r"\s+")
X = data[["range"]]
y = data["logratio"]

# Reserve 20% of the observations for the final test.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Lidar Regression")

with mlflow.start_run(run_name="linear-baseline"):
    mlflow.log_params({
        "model_type": "LinearRegression",
        "fit_intercept": True,
        "test_size": 0.2,
        "split_seed": 42,
        "n_train": len(X_train),
        "n_test": len(X_test),
    })

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Record training errors first.
    train_predictions = model.predict(X_train)
    mlflow.log_metrics({
        "train_mae": mean_absolute_error(y_train, train_predictions),
        "train_rmse": np.sqrt(
            mean_squared_error(y_train, train_predictions)
        ),
    })

    # Save the model with an example of its expected input.
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="linear_model",
        input_example=X_train.head(3),
    )

    # Plot observations and the fitted line.
    x_grid = pd.DataFrame({
        "range": np.linspace(
            X_train["range"].min(),
            X_train["range"].max(),
            200,
        )
    })

    fig, ax = plt.subplots()
    ax.scatter(X_train["range"], y_train, s=15, label="Training data")
    ax.plot(
        x_grid["range"],
        model.predict(x_grid),
        color="red",
        label="Linear baseline",
    )
    ax.set_xlabel("range")
    ax.set_ylabel("logratio")
    ax.legend()
    fig.tight_layout()

    mlflow.log_figure(fig, "plots/linear_baseline.png")
    plt.close(fig)

    print(f"Training observations: {len(X_train)}")
    print(f"Reserved test observations: {len(X_test)}")
    print(f"Model URI: {model_info.model_uri}")
