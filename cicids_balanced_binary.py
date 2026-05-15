from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


RANDOM_STATE = 42
CLASS_NAMES = {0: "BENIGN", 1: "ATTACK"}


class BinaryFlowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sequence: bool = False):
        self.features = torch.as_tensor(np.ascontiguousarray(features), dtype=torch.float32)
        self.labels = torch.as_tensor(np.ascontiguousarray(labels), dtype=torch.long)
        self.sequence = sequence

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features[index]
        if self.sequence:
            features = features.unsqueeze(0)
        return features, self.labels[index]


class MLPBinaryClassifier(nn.Module):
    def __init__(self, input_size: int, dropout: float = 0.25):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.network(batch)


class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(batch)
        return self.classifier(outputs[:, -1, :])


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stratified_sample(df: pd.DataFrame, row_limit: int | None, random_state: int) -> pd.DataFrame:
    if row_limit is None or row_limit >= len(df):
        return df

    sampled_frames = []
    class_counts = df["Label_Binary"].value_counts().sort_index()
    for label, count in class_counts.items():
        target_rows = max(1, int(round(row_limit * count / len(df))))
        target_rows = min(target_rows, int(count))
        sampled_frames.append(
            df[df["Label_Binary"] == label].sample(n=target_rows, random_state=random_state)
        )
    return pd.concat(sampled_frames, axis=0).sample(frac=1.0, random_state=random_state)


def load_binary_dataset(
    dataset_path: str | Path,
    row_limit: int | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_parquet(dataset_path)
    df = _stratified_sample(df, row_limit=row_limit, random_state=random_state)

    feature_names = [
        column
        for column in df.columns
        if column not in {"Label", "Label_Binary", "Label_Multiclass"}
    ]
    X = df[feature_names].astype(np.float32)
    y = df["Label_Binary"].astype(np.int64)
    return X, y, feature_names


def make_stratified_splits(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val_scaled = scaler.transform(X_val).astype(np.float32, copy=False)
    X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=False)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train.to_numpy(dtype=np.int64, copy=True),
        "y_val": y_val.to_numpy(dtype=np.int64, copy=True),
        "y_test": y_test.to_numpy(dtype=np.int64, copy=True),
        "X_train_scaled": X_train_scaled,
        "X_val_scaled": X_val_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
    }


def build_split_summary(splits: dict[str, object]) -> pd.DataFrame:
    rows = []
    total = sum(len(splits[f"y_{split}"]) for split in ("train", "val", "test"))
    for split, display_name in (("train", "train"), ("val", "validation"), ("test", "test")):
        y_split = np.asarray(splits[f"y_{split}"])
        counts = pd.Series(y_split).value_counts().reindex([0, 1], fill_value=0)
        rows.append(
            {
                "split": display_name,
                "rows": int(len(y_split)),
                "pct_total": len(y_split) / total,
                "benign_rows": int(counts.loc[0]),
                "attack_rows": int(counts.loc[1]),
                "attack_pct": float(counts.loc[1] / len(y_split)),
            }
        )
    return pd.DataFrame(rows)


def build_balance_summary(y_train: np.ndarray) -> pd.DataFrame:
    counts = pd.Series(y_train).value_counts().reindex([0, 1], fill_value=0)
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    ml_effective_weight = counts.to_numpy(dtype=float) * class_weights
    sampler_mass = counts.to_numpy(dtype=float) * (1.0 / np.maximum(counts.to_numpy(dtype=float), 1.0))
    return pd.DataFrame(
        {
            "class_id": [0, 1],
            "class_name": [CLASS_NAMES[0], CLASS_NAMES[1]],
            "train_rows": [int(counts.loc[0]), int(counts.loc[1])],
            "train_pct_original": [float(counts.loc[0] / len(y_train)), float(counts.loc[1] / len(y_train))],
            "class_weight": [float(class_weights[0]), float(class_weights[1])],
            "ml_effective_weight_total": [float(ml_effective_weight[0]), float(ml_effective_weight[1])],
            "ml_effective_pct": [
                float(ml_effective_weight[0] / ml_effective_weight.sum()),
                float(ml_effective_weight[1] / ml_effective_weight.sum()),
            ],
            "sampler_expected_pct": [
                float(sampler_mass[0] / sampler_mass.sum()),
                float(sampler_mass[1] / sampler_mass.sum()),
            ],
        }
    )


def _binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    precision_by_class, recall_by_class, f1_by_class, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average=None, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_attack": precision_by_class[1],
        "recall_attack": recall_by_class[1],
        "f1_attack": f1_by_class[1],
        "precision_benign": precision_by_class[0],
        "recall_benign": recall_by_class[0],
        "f1_benign": f1_by_class[0],
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
    return metrics


def _evaluate_sklearn_result(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, object]:
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
    return {
        "metrics": _binary_metrics(y, y_pred, y_score),
        "y_true": y,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def _evaluate_sklearn_model(model, X: pd.DataFrame, y: np.ndarray) -> dict[str, float]:
    return _evaluate_sklearn_result(model, X, y)["metrics"]


def train_ml_models(
    splits: dict[str, object],
    random_state: int = RANDOM_STATE,
    random_forest_estimators: int = 60,
    random_forest_max_samples: int = 250_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    y_train = np.asarray(splits["y_train"])
    train_size = len(y_train)
    rf_max_samples = min(random_forest_max_samples, train_size)

    models = {
        "decision_tree_balanced": DecisionTreeClassifier(
            random_state=random_state,
            class_weight="balanced",
            criterion="entropy",
            max_depth=16,
            min_samples_leaf=10,
        ),
        "random_forest_balanced": RandomForestClassifier(
            random_state=random_state,
            n_estimators=random_forest_estimators,
            class_weight="balanced_subsample",
            max_depth=20,
            min_samples_leaf=5,
            max_features="sqrt",
            bootstrap=True,
            max_samples=rf_max_samples,
            n_jobs=-1,
        ),
    }

    validation_rows = []
    test_rows = []
    fitted_models = {}
    for model_name, model in models.items():
        start_time = time.perf_counter()
        model.fit(splits["X_train"], y_train)
        train_seconds = time.perf_counter() - start_time
        fitted_models[model_name] = model

        val_metrics = _evaluate_sklearn_model(model, splits["X_val"], splits["y_val"])
        test_metrics = _evaluate_sklearn_model(model, splits["X_test"], splits["y_test"])

        validation_rows.append({"model": model_name, "family": "ML", "train_seconds": train_seconds, **val_metrics})
        test_rows.append({"model": model_name, "family": "ML", "train_seconds": train_seconds, **test_metrics})

    return pd.DataFrame(validation_rows), pd.DataFrame(test_rows), fitted_models


def _build_train_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    samples_per_epoch: int,
    sequence: bool,
) -> DataLoader:
    dataset = BinaryFlowDataset(features, labels, sequence=sequence)
    counts = np.bincount(labels, minlength=2)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(samples_per_epoch),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _build_eval_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    sequence: bool,
) -> DataLoader:
    dataset = BinaryFlowDataset(features, labels, sequence=sequence)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _evaluate_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)[:, 1]
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            true_labels.append(batch_y.numpy())

    y_true = np.concatenate(true_labels)
    y_pred = np.concatenate(predictions)
    y_score = np.concatenate(probabilities)
    return {
        "metrics": _binary_metrics(y_true, y_pred, y_score),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def _train_torch_model(
    model_name: str,
    model: nn.Module,
    splits: dict[str, object],
    sequence: bool,
    device: torch.device,
    epochs: int,
    batch_size: int,
    samples_per_epoch: int,
    lr: float,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame, nn.Module]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_loader = _build_train_loader(
        splits["X_train_scaled"],
        splits["y_train"],
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        sequence=sequence,
    )
    val_loader = _build_eval_loader(splits["X_val_scaled"], splits["y_val"], batch_size, sequence)
    test_loader = _build_eval_loader(splits["X_test_scaled"], splits["y_test"], batch_size, sequence)

    best_state = deepcopy(model.state_dict())
    best_f1 = -np.inf
    history_rows = []
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen_rows = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_y.size(0)
            seen_rows += batch_y.size(0)

        val_result = _evaluate_torch_model(model, val_loader, device)
        val_metrics = val_result["metrics"]
        mean_loss = running_loss / max(seen_rows, 1)
        history_rows.append(
            {
                "model": model_name,
                "epoch": epoch,
                "train_loss": mean_loss,
                "val_f1_macro": val_metrics["f1_macro"],
                "val_recall_attack": val_metrics["recall_attack"],
                "val_precision_attack": val_metrics["precision_attack"],
            }
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_state = deepcopy(model.state_dict())

    train_seconds = time.perf_counter() - start_time
    model.load_state_dict(best_state)

    val_metrics = _evaluate_torch_model(model, val_loader, device)["metrics"]
    test_metrics = _evaluate_torch_model(model, test_loader, device)["metrics"]
    val_metrics = {"model": model_name, "family": "DL", "train_seconds": train_seconds, **val_metrics}
    test_metrics = {"model": model_name, "family": "DL", "train_seconds": train_seconds, **test_metrics}
    return val_metrics, test_metrics, pd.DataFrame(history_rows), model


def train_deep_models(
    splits: dict[str, object],
    feature_count: int,
    epochs: int = 5,
    batch_size: int = 4096,
    samples_per_epoch: int = 250_000,
    lr: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, nn.Module]]:
    device = get_device()
    samples_per_epoch = min(samples_per_epoch, len(splits["y_train"]))
    model_specs = {
        "mlp_weighted_sampler": (MLPBinaryClassifier(feature_count), False),
        "lstm_weighted_sampler": (LSTMBinaryClassifier(feature_count), True),
    }

    validation_rows = []
    test_rows = []
    history_frames = []
    trained_models = {}

    for model_name, (model, sequence) in model_specs.items():
        val_metrics, test_metrics, history, trained_model = _train_torch_model(
            model_name=model_name,
            model=model,
            splits=splits,
            sequence=sequence,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            samples_per_epoch=samples_per_epoch,
            lr=lr,
        )
        validation_rows.append(val_metrics)
        test_rows.append(test_metrics)
        history_frames.append(history)
        trained_models[model_name] = trained_model

    return (
        pd.DataFrame(validation_rows),
        pd.DataFrame(test_rows),
        pd.concat(history_frames, axis=0, ignore_index=True),
        trained_models,
    )


def build_test_artifacts(
    splits: dict[str, object],
    ml_models: dict[str, object],
    dl_models: dict[str, nn.Module],
    batch_size: int,
) -> dict[str, dict[str, object]]:
    artifacts: dict[str, dict[str, object]] = {}

    for model_name, model in ml_models.items():
        artifacts[model_name] = {
            "family": "ML",
            **_evaluate_sklearn_result(model, splits["X_test"], splits["y_test"]),
        }

    sequence_by_model = {
        "mlp_weighted_sampler": False,
        "lstm_weighted_sampler": True,
    }
    for model_name, model in dl_models.items():
        device = next(model.parameters()).device
        loader = _build_eval_loader(
            splits["X_test_scaled"],
            splits["y_test"],
            batch_size=batch_size,
            sequence=sequence_by_model[model_name],
        )
        artifacts[model_name] = {
            "family": "DL",
            **_evaluate_torch_model(model, loader, device),
        }

    return artifacts


def run_balanced_binary_experiment(
    dataset_path: str | Path,
    row_limit: int | None = None,
    deep_epochs: int = 5,
    deep_batch_size: int = 4096,
    deep_samples_per_epoch: int = 250_000,
    random_forest_estimators: int = 60,
    random_forest_max_samples: int = 250_000,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    X, y, feature_names = load_binary_dataset(
        dataset_path=dataset_path,
        row_limit=row_limit,
        random_state=random_state,
    )
    splits = make_stratified_splits(X, y, random_state=random_state)

    ml_validation, ml_test, ml_models = train_ml_models(
        splits,
        random_state=random_state,
        random_forest_estimators=random_forest_estimators,
        random_forest_max_samples=random_forest_max_samples,
    )
    dl_validation, dl_test, dl_history, dl_models = train_deep_models(
        splits,
        feature_count=len(feature_names),
        epochs=deep_epochs,
        batch_size=deep_batch_size,
        samples_per_epoch=deep_samples_per_epoch,
    )

    validation_metrics = pd.concat([ml_validation, dl_validation], axis=0, ignore_index=True)
    test_metrics = pd.concat([ml_test, dl_test], axis=0, ignore_index=True)

    sort_columns = ["f1_macro", "recall_attack", "balanced_accuracy"]
    validation_metrics = validation_metrics.sort_values(sort_columns, ascending=False).reset_index(drop=True)
    test_metrics = test_metrics.sort_values(sort_columns, ascending=False).reset_index(drop=True)

    best_validation_model = validation_metrics.iloc[0]["model"]
    best_test_model = test_metrics.iloc[0]["model"]

    return {
        "feature_names": feature_names,
        "split_summary": build_split_summary(splits),
        "balance_summary": build_balance_summary(splits["y_train"]),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "deep_history": dl_history,
        "best_validation_model": best_validation_model,
        "best_test_model": best_test_model,
        "models": {**ml_models, **dl_models},
        "test_artifacts": build_test_artifacts(
            splits=splits,
            ml_models=ml_models,
            dl_models=dl_models,
            batch_size=deep_batch_size,
        ),
    }
