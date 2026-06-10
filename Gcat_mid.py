import argparse
import copy
import json
import math
import os
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {DEVICE} ---")


class CustomTorchClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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

        layers.append(nn.Linear(current_dim, output_dim))
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
    "early_stopping_patience": 10_000,
    "random_state": 42,
}

CHECKPOINT_STAGES = (0.2, 0.5, 0.8)


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


def _ensure_2d_numpy(X):
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError("Expected a pandas DataFrame for feature input.")


def _save_artifacts(
    save_dir: str,
    model_state_dict: Dict,
    preprocessor,
    label_encoder: LabelEncoder,
    metadata: Dict,
):
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "custom_pytorch_model.pth")
    torch.save(model_state_dict, model_path)

    if preprocessor is not None:
        joblib.dump(preprocessor, os.path.join(save_dir, "preprocessor.joblib"))

    joblib.dump(label_encoder, os.path.join(save_dir, "label_encoder.joblib"))

    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved checkpoint to: {save_dir}")


def train_and_save_pytorch_classifier_mid(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    dataname: str,
    target_col: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
) -> Dict:
    print("--- Start preprocessing and classifier training ---")

    if y_df.shape[1] != 1:
        raise ValueError("y_df must contain exactly one target column.")
    if target_col != y_df.columns[0]:
        raise ValueError(f"target_col='{target_col}' does not match y_df column '{y_df.columns[0]}'.")

    model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG if model_config is None else model_config)
    training_config = copy.deepcopy(DEFAULT_TRAINING_CONFIG if training_config is None else training_config)

    preprocessor, numerical_cols, categorical_cols = _build_feature_preprocessor(X_df)
    print(f"Target column: {target_col}")
    print(f"Numerical feature columns: {numerical_cols}")
    print(f"Categorical feature columns: {categorical_cols}")

    y_raw = y_df.iloc[:, 0]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw)
    num_classes = int(len(label_encoder.classes_))
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")

    X_train_raw, X_temp_raw, y_train_raw, y_temp_raw = train_test_split(
        X_df.copy(),
        y_encoded,
        test_size=0.3,
        random_state=int(training_config.get("random_state", 42)),
        stratify=y_encoded,
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_temp_raw,
        y_temp_raw,
        test_size=0.5,
        random_state=int(training_config.get("random_state", 42)),
        stratify=y_temp_raw,
    )
    print(f"Split sizes: train={len(y_train_raw)}, val={len(y_val_raw)}, test={len(y_test_raw)}")

    if preprocessor != "passthrough":
        X_train_np = preprocessor.fit_transform(X_train_raw)
        X_val_np = preprocessor.transform(X_val_raw)
        X_test_np = preprocessor.transform(X_test_raw)
        preprocessor_to_save = preprocessor
        preprocessor_type = "column_transformer"
    else:
        X_train_np = X_train_raw.to_numpy(dtype=np.float32)
        X_val_np = X_val_raw.to_numpy(dtype=np.float32)
        X_test_np = X_test_raw.to_numpy(dtype=np.float32)
        preprocessor_to_save = None
        preprocessor_type = "passthrough"

    X_train_np = np.asarray(X_train_np, dtype=np.float32)
    X_val_np = np.asarray(X_val_np, dtype=np.float32)
    X_test_np = np.asarray(X_test_np, dtype=np.float32)

    if np.isnan(X_train_np).any() or np.isnan(X_val_np).any() or np.isnan(X_test_np).any():
        raise ValueError("NaN detected in processed features. Please clean or impute the input data first.")

    input_dim = int(X_train_np.shape[1])
    current_model_config = copy.deepcopy(model_config)
    current_model_config["input_dim"] = input_dim
    current_model_config["output_dim"] = num_classes

    model = CustomTorchClassifier(**current_model_config).to(DEVICE)
    print(model)

    class_counts = np.bincount(y_train_raw, minlength=num_classes)
    class_weights = len(y_train_raw) / np.maximum(class_counts, 1)
    class_weights = class_weights / class_weights.mean()
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=float(training_config.get("learning_rate", 5e-4)))

    batch_size = int(training_config.get("batch_size", 128))
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_np, dtype=torch.float32),
            torch.tensor(y_train_raw, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val_np, dtype=torch.float32),
            torch.tensor(y_val_raw, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    epochs = int(training_config.get("epochs", 100))
    patience = int(training_config.get("early_stopping_patience", 10_000))
    best_val_loss = float("inf")
    best_model_state = None
    epochs_no_improve = 0

    checkpoint_epochs = {stage: max(1, min(epochs, math.ceil(epochs * stage))) for stage in CHECKPOINT_STAGES}
    saved_stages = set()

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "checkpoint_epochs": checkpoint_epochs,
    }

    print(f"Checkpoint epochs: {checkpoint_epochs}")
    print("--- Start training ---")

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += xb.size(0)

        epoch_train_loss = train_loss_sum / max(train_total, 1)
        epoch_train_acc = train_correct / max(train_total, 1)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)

        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss_sum += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == yb).sum().item()
                val_total += xb.size(0)

        epoch_val_loss = val_loss_sum / max(val_total, 1)
        epoch_val_acc = val_correct / max(val_total, 1)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"train_loss: {epoch_train_loss:.6f} - train_acc: {epoch_train_acc:.4f} - "
            f"val_loss: {epoch_val_loss:.6f} - val_acc: {epoch_val_acc:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print(f"Validation improved to {best_val_loss:.6f}. Saving best-state candidate.")
        else:
            epochs_no_improve += 1

        current_epoch_num = epoch + 1
        for stage, ckpt_epoch in checkpoint_epochs.items():
            if stage not in saved_stages and current_epoch_num >= ckpt_epoch:
                stage_dir = f"model_{dataname}_{stage:.1f}"
                stage_metadata = {
                    "target_col": target_col,
                    "input_dim": input_dim,
                    "output_dim": num_classes,
                    "feature_columns": list(X_df.columns),
                    "model_config": model_config,
                    "training_config": training_config,
                    "preprocessor_type": preprocessor_type,
                    "numerical_cols": numerical_cols,
                    "categorical_cols": categorical_cols,
                    "classes": [str(x) for x in label_encoder.classes_.tolist()],
                    "stage": float(stage),
                    "saved_at_epoch": int(current_epoch_num),
                    "checkpoint_epochs": checkpoint_epochs,
                }
                _save_artifacts(
                    save_dir=stage_dir,
                    model_state_dict=copy.deepcopy(model.state_dict()),
                    preprocessor=preprocessor_to_save,
                    label_encoder=label_encoder,
                    metadata=stage_metadata,
                )
                saved_stages.add(stage)

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_np, dtype=torch.float32, device=DEVICE)
        test_y = torch.tensor(y_test_raw, dtype=torch.long, device=DEVICE)
        test_logits = model(test_x)
        test_loss = criterion(test_logits, test_y).item()
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (test_preds == test_y).float().mean().item()

    print(f"Test CE loss: {test_loss:.6f}")
    print(f"Test accuracy: {test_acc:.4f}")

    return {
        "history": history,
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "checkpoint_epochs": checkpoint_epochs,
        "saved_stages": sorted(float(x) for x in saved_stages),
    }


def _load_saved_components(save_dir: str):
    metadata_path = os.path.join(save_dir, "metadata.json")
    model_path = os.path.join(save_dir, "custom_pytorch_model.pth")
    preprocessor_path = os.path.join(save_dir, "preprocessor.joblib")
    label_encoder_path = os.path.join(save_dir, "label_encoder.joblib")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Missing label encoder file: {label_encoder_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    preprocessor = None
    if metadata.get("preprocessor_type") != "passthrough":
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Missing preprocessor file: {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)

    label_encoder = joblib.load(label_encoder_path)

    model_config = copy.deepcopy(metadata["model_config"])
    model_config["input_dim"] = int(metadata["input_dim"])
    model_config["output_dim"] = int(metadata["output_dim"])

    model = CustomTorchClassifier(**model_config).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    return model, preprocessor, label_encoder, metadata


def model_output(
    dataname: str,
    X_input_df_main: pd.DataFrame,
    y_condition: Union[np.ndarray, torch.Tensor, pd.Series, pd.DataFrame, List, List[float], List[int]],
    target_shape: int,
    stage: float = 0.8,
):
    """
    Return values aligned with the original Gcat.py style:
      - condensed gradient: torch.Tensor of shape (N, target_shape)
      - prediction: torch.Tensor of shape (N, 1) storing predicted class indices
      - loss: float (cross-entropy)

    stage controls which checkpoint to load:
      - 0.2 -> ./model_{dataname}_0.2/
      - 0.5 -> ./model_{dataname}_0.5/
      - 0.8 -> ./model_{dataname}_0.8/
    """
    stage = float(stage)
    valid_stages = {0.2, 0.5, 0.8}
    if stage not in valid_stages:
        raise ValueError(f"stage must be one of {sorted(valid_stages)}, got {stage}")

    save_dir = f"./model_{dataname}_{stage:.1f}"
    model, preprocessor, label_encoder, metadata = _load_saved_components(save_dir)

    X_df = _ensure_2d_numpy(X_input_df_main).copy()
    for col in X_df.select_dtypes(include="bool").columns:
        X_df[col] = X_df[col].astype(int)

    if preprocessor is not None:
        transformed_sample = preprocessor.transform(X_df)
    else:
        transformed_sample = X_df.to_numpy(dtype=np.float32)
    transformed_sample = np.asarray(transformed_sample, dtype=np.float32)

    input_tensor = torch.tensor(transformed_sample, dtype=torch.float32, device=DEVICE)
    input_tensor.requires_grad_()

    if isinstance(y_condition, pd.DataFrame):
        y_raw = y_condition.iloc[:, 0].to_numpy()
    elif isinstance(y_condition, pd.Series):
        y_raw = y_condition.to_numpy()
    elif isinstance(y_condition, torch.Tensor):
        y_raw = y_condition.detach().cpu().numpy()
    else:
        y_raw = np.asarray(y_condition)

    if y_raw.ndim > 1:
        y_raw = y_raw.reshape(-1)

    if np.issubdtype(y_raw.dtype, np.number):
        y_encoded = y_raw.astype(np.int64)
    else:
        y_encoded = label_encoder.transform(y_raw)

    y_tensor = torch.tensor(y_encoded, dtype=torch.long, device=DEVICE)

    logits = model(input_tensor)
    min_len = min(logits.shape[0], y_tensor.shape[0])
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits[:min_len], y_tensor[:min_len])

    model.zero_grad()
    loss.backward()

    full_gradient = input_tensor.grad.detach()
    grad_for_interpolate = full_gradient.unsqueeze(1)
    condensed_grad = F.interpolate(grad_for_interpolate, size=target_shape, mode="area")
    final_grad = condensed_grad.squeeze(1)

    pred_class_idx = torch.argmax(logits, dim=1, keepdim=True).detach()
    return final_grad, pred_class_idx, float(loss.item())


def run_training_from_csv(
    dataname: str,
    csv_path: str,
    model_config: Optional[Dict] = None,
    training_config: Optional[Dict] = None,
):
    data_df = pd.read_csv(csv_path)
    if data_df.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and one target column.")

    target_col = data_df.columns[-1]
    print(f"Using last column as target: {target_col}")

    X_df = data_df.iloc[:, :-1].copy()
    y_df = pd.DataFrame(data_df.iloc[:, -1].copy())

    return train_and_save_pytorch_classifier_mid(
        X_df=X_df,
        y_df=y_df,
        dataname=dataname,
        target_col=target_col,
        model_config=model_config,
        training_config=training_config,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train midpoint classifier checkpoints at 20%, 50%, 80% using the LAST discrete target column."
    )
    parser.add_argument("--dataname", type=str, required=True, help="Dataset name. Saves to model_{dataname}_0.2/0.5/0.8")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10000,
        help="Large default to avoid stopping before the 20/50/80 checkpoints are saved.",
    )
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
        model_config=DEFAULT_MODEL_CONFIG,
        training_config=training_config,
    )
