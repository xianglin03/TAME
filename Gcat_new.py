import argparse
import json
import os
import copy
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")


class CustomTorchRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_units: List[int] = [128, 64],
        activations: List[str] = ["relu", "relu"],
        dropout_rates: List[float] = [0.2, 0.2],
    ):
        super().__init__()
        if not (len(hidden_units) == len(activations) == len(dropout_rates)):
            raise ValueError("hidden_units, activations, and dropout_rates must have the same length.")

        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim, act_name, drop_rate in zip(hidden_units, activations, dropout_rates):
            layers.append(nn.Linear(current_dim, hidden_dim))

            act_name = act_name.lower()
            if act_name == "relu":
                layers.append(nn.ReLU())
            elif act_name == "tanh":
                layers.append(nn.Tanh())
            elif act_name == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise ValueError(f"Unsupported activation: {act_name}")

            if drop_rate > 0:
                layers.append(nn.Dropout(drop_rate))

            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


DEFAULT_MODEL_CONFIG: Dict[str, Union[List[int], List[str], List[float]]] = {
    "hidden_units": [256, 128, 64],
    "activations": ["relu", "relu", "leaky_relu"],
    "dropout_rates": [0.3, 0.2, 0.1],
}

DEFAULT_TRAINING_CONFIG: Dict[str, Union[int, float]] = {
    "learning_rate": 5e-4,
    "epochs": 100,
    "batch_size": 128,
    "early_stopping_patience": 15,
}


class IdentityTargetScaler:
    def fit(self, y: np.ndarray):
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y



def _build_feature_preprocessor(X_df: pd.DataFrame) -> Tuple[Union[ColumnTransformer, str], List[str], List[str]]:
    X_features = X_df.copy()

    numerical_cols = X_features.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X_features.select_dtypes(include=["object", "category"]).columns.tolist()
    boolean_cols = X_features.select_dtypes(include="bool").columns.tolist()

    for col in boolean_cols:
        X_features[col] = X_features[col].astype(int)

    numerical_cols = list(dict.fromkeys(numerical_cols + boolean_cols))
    categorical_cols = [col for col in categorical_cols if col not in numerical_cols]

    transformers = []
    if numerical_cols:
        transformers.append(("num", StandardScaler(), numerical_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols))

    if not transformers:
        return "passthrough", numerical_cols, categorical_cols

    return ColumnTransformer(transformers=transformers, remainder="passthrough"), numerical_cols, categorical_cols



def train_and_save_pytorch_regressor(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    target_col: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
    save_dir: str = "saved_pytorch_regressor",
) -> Tuple[str, Optional[str], str, Dict]:
    print("--- Start preprocessing and regression model training ---")

    if y_df.shape[1] != 1:
        raise ValueError("y_df must contain exactly one target column.")
    if target_col != y_df.columns[0]:
        raise ValueError(f"target_col='{target_col}' does not match y_df column '{y_df.columns[0]}'.")
    if not pd.api.types.is_numeric_dtype(y_df.iloc[:, 0]):
        raise ValueError("The selected target column must be continuous / numeric.")

    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG if model_config is None else model_config)
    training_config = copy.deepcopy(DEFAULT_TRAINING_CONFIG if training_config is None else training_config)

    y_np_raw = y_df.iloc[:, 0].astype(float).values.reshape(-1, 1)
    print(f"Target column: {target_col}")
    print(f"Target stats: mean={y_np_raw.mean():.6f}, std={y_np_raw.std():.6f}, min={y_np_raw.min():.6f}, max={y_np_raw.max():.6f}")

    preprocessor, numerical_cols, categorical_cols = _build_feature_preprocessor(X_df)
    print(f"Numerical feature columns: {numerical_cols}")
    print(f"Categorical feature columns: {categorical_cols}")

    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_df.copy(), y_np_raw, test_size=0.3, random_state=42
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw, y_temp_raw, test_size=0.5, random_state=42
    )
    print(f"Split sizes: train={len(y_train_raw)}, val={len(y_val_raw)}, test={len(y_test_raw)}")

    if preprocessor != "passthrough":
        X_train_np = preprocessor.fit_transform(X_train_raw)
        X_val_np = preprocessor.transform(X_val_raw)
        X_test_np = preprocessor.transform(X_test_raw)
        preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
    else:
        X_train_np = X_train_raw.to_numpy(dtype=np.float32)
        X_val_np = X_val_raw.to_numpy(dtype=np.float32)
        X_test_np = X_test_raw.to_numpy(dtype=np.float32)
        preprocessor_path = None

    if np.isnan(X_train_np).any() or np.isnan(X_val_np).any() or np.isnan(X_test_np).any():
        raise ValueError("NaN detected in processed features. Please clean or impute the input data first.")

    use_target_scaler = float(np.std(y_train_raw)) > 0
    if use_target_scaler:
        target_scaler: Union[StandardScaler, IdentityTargetScaler] = StandardScaler()
    else:
        target_scaler = IdentityTargetScaler()

    y_train_np = target_scaler.fit(y_train_raw).transform(y_train_raw).astype(np.float32)
    y_val_np = target_scaler.transform(y_val_raw).astype(np.float32)
    y_test_np = target_scaler.transform(y_test_raw).astype(np.float32)

    input_dim = X_train_np.shape[1]
    current_model_config = copy.deepcopy(model_config)
    current_model_config["input_dim"] = input_dim

    model = CustomTorchRegressor(**current_model_config).to(DEVICE)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(training_config.get("learning_rate", 5e-4)))

    batch_size = int(training_config.get("batch_size", 128))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_np, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_np, dtype=torch.float32),
            torch.tensor(y_val_np, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    epochs = int(training_config.get("epochs", 100))
    early_stopping_patience = int(training_config.get("early_stopping_patience", 15))
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    history = {"train_loss": [], "val_loss": []}

    print("--- Start training ---")
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * xb.size(0)

        epoch_train_loss = train_loss_sum / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss_sum += loss.item() * xb.size(0)

        epoch_val_loss = val_loss_sum / len(val_loader.dataset)
        history["val_loss"].append(epoch_val_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} - train_loss: {epoch_train_loss:.6f} - val_loss: {epoch_val_loss:.6f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Validation improved to {best_val_loss:.6f}. Saving best state.")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {early_stopping_patience} epochs without improvement.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
        test_y = torch.tensor(y_test_np, dtype=torch.float32).to(DEVICE)
        test_preds = model(test_x)
        test_loss = criterion(test_preds, test_y).item()

        test_preds_orig = target_scaler.inverse_transform(test_preds.cpu().numpy())
        test_y_orig = target_scaler.inverse_transform(test_y.cpu().numpy())
        test_mae = float(np.mean(np.abs(test_preds_orig - test_y_orig)))
        test_rmse = float(np.sqrt(np.mean((test_preds_orig - test_y_orig) ** 2)))

    print(f"Test scaled MSE: {test_loss:.6f}")
    print(f"Test MAE (original scale): {test_mae:.6f}")
    print(f"Test RMSE (original scale): {test_rmse:.6f}")

    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "custom_pytorch_model.pth")
    torch.save(model.state_dict(), model_path)

    if preprocessor != "passthrough":
        joblib.dump(preprocessor, preprocessor_path)

    target_scaler_path = os.path.join(save_dir, "target_scaler.joblib")
    joblib.dump(target_scaler, target_scaler_path)

    metadata = {
        "target_col": target_col,
        "input_dim": input_dim,
        "feature_columns": list(X_df.columns),
        "model_config": model_config,
        "training_config": training_config,
        "preprocessor_type": "column_transformer" if preprocessor != "passthrough" else "passthrough",
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "target_scaler": target_scaler.__class__.__name__,
    }
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Model saved to: {model_path}")
    if preprocessor_path is not None:
        print(f"Preprocessor saved to: {preprocessor_path}")
    print(f"Target scaler saved to: {target_scaler_path}")
    print(f"Metadata saved to: {metadata_path}")

    return model_path, preprocessor_path, target_scaler_path, history



def _load_saved_components(save_dir: str):
    metadata_path = os.path.join(save_dir, "metadata.json")
    model_path = os.path.join(save_dir, "custom_pytorch_model.pth")
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
    target_scaler_path = os.path.join(save_dir, "target_scaler.joblib")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(target_scaler_path):
        raise FileNotFoundError(f"Missing target scaler file: {target_scaler_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocessor = None
    if metadata.get("preprocessor_type") != "passthrough":
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Missing preprocessor file: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

    target_scaler = joblib.load(target_scaler_path)

    model_config = copy.deepcopy(metadata["model_config"])
    model_config["input_dim"] = int(metadata["input_dim"])
    model = CustomTorchRegressor(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model, preprocessor, target_scaler, metadata



def model_output(
    dataname: str,
    X_input_df_main: pd.DataFrame,
    y_condition: Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame, List[float]],
    target_shape: int,
):
    """
    Return values aligned with the original Gcat.py interface:
      - condensed gradient: torch.Tensor of shape (N, target_shape)
      - prediction: torch.Tensor of shape (N, 1)
      - loss: float

    Notes:
      - This version uses a user-specified continuous target column.
      - It loads artifacts from ./model_{dataname}_new/
      - The returned prediction is on the ORIGINAL target scale.
    """
    save_dir = f"./model_{dataname}_new"
    model, preprocessor, target_scaler, metadata = _load_saved_components(save_dir)

    X_df = X_input_df_main.copy()
    for col in X_df.select_dtypes(include="bool").columns:
        X_df[col] = X_df[col].astype(int)

    if preprocessor is not None:
        transformed_sample = preprocessor.transform(X_df)
    else:
        transformed_sample = X_df.to_numpy(dtype=np.float32)

    input_tensor = torch.tensor(transformed_sample, dtype=torch.float32, device=DEVICE)
    input_tensor.requires_grad_()

    if isinstance(y_condition, pd.DataFrame):
        y_np = y_condition.iloc[:, 0].astype(float).values.reshape(-1, 1)
    elif isinstance(y_condition, pd.Series):
        y_np = y_condition.astype(float).values.reshape(-1, 1)
    elif isinstance(y_condition, torch.Tensor):
        y_np = y_condition.detach().cpu().numpy().astype(np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)
    else:
        y_np = np.asarray(y_condition, dtype=np.float32)
        if y_np.ndim == 1:
            y_np = y_np.reshape(-1, 1)

    y_scaled_np = target_scaler.transform(y_np).astype(np.float32)
    y_scaled = torch.tensor(y_scaled_np, dtype=torch.float32, device=DEVICE)

    predictions_scaled = model(input_tensor)

    min_len = min(predictions_scaled.shape[0], y_scaled.shape[0])
    loss_fn = nn.MSELoss()
    loss = loss_fn(predictions_scaled[:min_len], y_scaled[:min_len])

    model.zero_grad()
    loss.backward()

    full_gradient = input_tensor.grad.detach()
    grad_for_interpolate = full_gradient.unsqueeze(1)
    condensed_grad = F.interpolate(grad_for_interpolate, size=target_shape, mode="area")
    final_grad = condensed_grad.squeeze(1)

    predictions_scaled_np = predictions_scaled.detach().cpu().numpy()
    predictions_orig_np = target_scaler.inverse_transform(predictions_scaled_np)
    predictions_orig = torch.tensor(predictions_orig_np, dtype=torch.float32, device=DEVICE)

    return final_grad, predictions_orig.detach(), float(loss.item())



def run_training_from_csv(
    dataname: str,
    csv_path: str,
    target_col: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
):
    data_df = pd.read_csv(csv_path)
    if target_col not in data_df.columns:
        raise ValueError(f"target_col '{target_col}' not found in CSV columns: {list(data_df.columns)}")
    if not pd.api.types.is_numeric_dtype(data_df[target_col]):
        raise ValueError(f"target_col '{target_col}' must be continuous / numeric.")

    X_df = data_df.drop(columns=[target_col])
    y_df = pd.DataFrame(data_df[target_col].astype(float))

    save_dir = f"model_{dataname}_new"
    return train_and_save_pytorch_regressor(
        X_df=X_df,
        y_df=y_df,
        target_col=target_col,
        model_config=model_config,
        training_config=training_config,
        save_dir=save_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Gcat_new regression model with a user-specified continuous target column.")
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name. Artifacts are saved to model_{dataname}_new/")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--target_col", type=str, required=True, help="Name of the continuous target column")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--early_stopping_patience", type=int, default=15)
    args = parser.parse_args()

    training_config = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    training_config.update(
        {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "early_stopping_patience": args.early_stopping_patience,
        }
    )

    run_training_from_csv(
        dataname=args.dataname,
        csv_path=args.csv_path,
        target_col=args.target_col,
        model_config=DEFAULT_MODEL_CONFIG,
        training_config=training_config,
    )
