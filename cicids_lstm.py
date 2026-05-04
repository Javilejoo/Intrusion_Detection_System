from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


RANDOM_STATE = 42


class FlowSequenceDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)
        self.features = torch.from_numpy(features).unsqueeze(1)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        projection_size = max(hidden_size // 2, 32)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(projection_size, num_classes),
        )

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(batch)
        last_hidden = outputs[:, -1, :]
        return self.classifier(last_hidden)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_lstm_data(dataset_path: str | Path) -> dict[str, object]:
    dataset_path = Path(dataset_path)
    df_model = pd.read_parquet(dataset_path)

    label_lookup = (
        df_model[["Label", "Label_Multiclass"]]
        .drop_duplicates()
        .sort_values("Label_Multiclass")
        .reset_index(drop=True)
    )
    class_names = label_lookup["Label"].tolist()

    X = df_model.drop(columns=["Label", "Label_Binary", "Label_Multiclass"]).astype(np.float32)
    y = df_model["Label_Multiclass"].astype(np.int64)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32, copy=True)
    X_val_scaled = scaler.transform(X_val).astype(np.float32, copy=True)
    X_test_scaled = scaler.transform(X_test).astype(np.float32, copy=True)

    return {
        "dataset_path": dataset_path,
        "feature_names": X.columns.tolist(),
        "class_names": class_names,
        "label_lookup": label_lookup,
        "scaler": scaler,
        "X_train": X_train_scaled,
        "X_val": X_val_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train.to_numpy(dtype=np.int64, copy=True),
        "y_val": y_val.to_numpy(dtype=np.int64, copy=True),
        "y_test": y_test.to_numpy(dtype=np.int64, copy=True),
    }


def build_split_summary(lstm_data: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y_train = pd.Series(lstm_data["y_train"])
    y_val = pd.Series(lstm_data["y_val"])
    y_test = pd.Series(lstm_data["y_test"])
    class_names = lstm_data["class_names"]

    split_summary = pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "rows": [len(y_train), len(y_val), len(y_test)],
        }
    )
    split_summary["pct"] = split_summary["rows"] / split_summary["rows"].sum()

    class_summary = pd.concat(
        [
            y_train.value_counts(normalize=True).rename("train_pct"),
            y_val.value_counts(normalize=True).rename("validation_pct"),
            y_test.value_counts(normalize=True).rename("test_pct"),
            y_train.value_counts().add(y_val.value_counts(), fill_value=0).add(y_test.value_counts(), fill_value=0).rename("total_rows"),
        ],
        axis=1,
    ).fillna(0)
    class_summary.index = [class_names[idx] for idx in class_summary.index]
    class_summary = class_summary.sort_values("total_rows", ascending=False)
    return split_summary, class_summary


def _build_train_loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    samples_per_epoch: int,
) -> DataLoader:
    dataset = FlowSequenceDataset(features, labels)
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    class_weights = 1.0 / np.maximum(counts, 1)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=int(samples_per_epoch),
        replacement=True,
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _build_eval_loader(features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    dataset = FlowSequenceDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def _evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    probs_list: list[np.ndarray] = []
    preds_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

            total_loss += loss.item() * batch_y.size(0)
            probs_list.append(probabilities.cpu().numpy())
            preds_list.append(predictions.cpu().numpy())
            true_list.append(batch_y.cpu().numpy())

    y_true = np.concatenate(true_list)
    y_pred = np.concatenate(preds_list)
    y_proba = np.concatenate(probs_list)
    metrics = _metric_row(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def _copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_lstm_candidate(
    lstm_data: dict[str, object],
    params: dict[str, object],
    device: torch.device | None = None,
) -> dict[str, object]:
    device = device or get_device()
    input_size = len(lstm_data["feature_names"])
    num_classes = len(lstm_data["class_names"])

    model = LSTMClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=int(params["hidden_size"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"]),
    ).to(device)

    train_loader = _build_train_loader(
        features=lstm_data["X_train"],
        labels=lstm_data["y_train"],
        batch_size=int(params["batch_size"]),
        samples_per_epoch=int(params["samples_per_epoch"]),
    )
    val_loader = _build_eval_loader(
        features=lstm_data["X_val"],
        labels=lstm_data["y_val"],
        batch_size=int(params.get("eval_batch_size", 4096)),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(params.get("label_smoothing", 0.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))

    best_state = None
    best_epoch = 0
    best_score = -np.inf
    patience_counter = 0
    history_rows = []

    for epoch in range(1, int(params["epochs"]) + 1):
        model.train()
        train_loss_accumulator = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accumulator += loss.item() * batch_y.size(0)

        train_loss = train_loss_accumulator / int(params["samples_per_epoch"])
        evaluation = _evaluate_model(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": evaluation["metrics"]["loss"],
            **{f"val_{k}": v for k, v in evaluation["metrics"].items() if k != "loss"},
        }
        history_rows.append(row)

        current_score = row["val_f1_macro"]
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            best_state = _copy_state_dict(model)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= int(params["patience"]):
            break

    history_df = pd.DataFrame(history_rows)
    best_row = history_df.loc[history_df["val_f1_macro"].idxmax()].to_dict()

    return {
        "params": deepcopy(params),
        "history": history_df,
        "best_epoch": best_epoch,
        "best_metrics": best_row,
        "best_state_dict": best_state,
    }


def run_validation_search(
    lstm_data: dict[str, object],
    candidate_params: list[dict[str, object]],
    device: torch.device | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    results = []
    artifacts = []

    for idx, params in enumerate(candidate_params, start=1):
        artifact = train_lstm_candidate(lstm_data=lstm_data, params=params, device=device)
        artifacts.append(artifact)
        results.append(
            {
                "candidate_id": idx,
                **artifact["params"],
                "best_epoch": artifact["best_epoch"],
                "accuracy": artifact["best_metrics"]["val_accuracy"],
                "precision_macro": artifact["best_metrics"]["val_precision_macro"],
                "recall_macro": artifact["best_metrics"]["val_recall_macro"],
                "f1_macro": artifact["best_metrics"]["val_f1_macro"],
                "precision_weighted": artifact["best_metrics"]["val_precision_weighted"],
                "recall_weighted": artifact["best_metrics"]["val_recall_weighted"],
                "f1_weighted": artifact["best_metrics"]["val_f1_weighted"],
                "val_loss": artifact["best_metrics"]["val_loss"],
            }
        )

    results_df = pd.DataFrame(results).sort_values(
        by=["f1_macro", "recall_macro", "accuracy"],
        ascending=False,
    ).reset_index(drop=True)
    return results_df, artifacts


def retrain_and_evaluate_best_lstm(
    lstm_data: dict[str, object],
    best_config: dict[str, object],
    device: torch.device | None = None,
) -> dict[str, object]:
    device = device or get_device()
    X_trainval = np.concatenate([lstm_data["X_train"], lstm_data["X_val"]], axis=0)
    y_trainval = np.concatenate([lstm_data["y_train"], lstm_data["y_val"]], axis=0)

    model = LSTMClassifier(
        input_size=len(lstm_data["feature_names"]),
        num_classes=len(lstm_data["class_names"]),
        hidden_size=int(best_config["hidden_size"]),
        num_layers=int(best_config["num_layers"]),
        dropout=float(best_config["dropout"]),
    ).to(device)

    train_loader = _build_train_loader(
        features=X_trainval,
        labels=y_trainval,
        batch_size=int(best_config["batch_size"]),
        samples_per_epoch=int(best_config["samples_per_epoch"]),
    )
    test_loader = _build_eval_loader(
        features=lstm_data["X_test"],
        labels=lstm_data["y_test"],
        batch_size=int(best_config.get("eval_batch_size", 4096)),
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=float(best_config.get("label_smoothing", 0.0)))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(best_config["lr"]))

    history_rows = []
    total_epochs = int(best_config["best_epoch"])
    for epoch in range(1, total_epochs + 1):
        model.train()
        train_loss_accumulator = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_accumulator += loss.item() * batch_y.size(0)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_accumulator / int(best_config["samples_per_epoch"]),
            }
        )

    test_artifact = _evaluate_model(model, test_loader, criterion, device)
    history_df = pd.DataFrame(history_rows)

    report = classification_report(
        test_artifact["y_true"],
        test_artifact["y_pred"],
        labels=list(range(len(lstm_data["class_names"]))),
        target_names=lstm_data["class_names"],
        output_dict=True,
        zero_division=0,
    )
    report_df = (
        pd.DataFrame(report)
        .T.reset_index()
        .rename(columns={"index": "label"})
    )
    report_df = report_df[report_df["label"].isin(lstm_data["class_names"])]
    report_df = report_df[["label", "precision", "recall", "f1-score", "support"]]
    report_df = report_df.sort_values("support", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "history": history_df,
        "test_metrics": pd.DataFrame([test_artifact["metrics"]]),
        "report_df": report_df,
        "y_true": test_artifact["y_true"],
        "y_pred": test_artifact["y_pred"],
        "y_proba": test_artifact["y_proba"],
    }


def build_roc_artifacts(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
) -> dict[str, object]:
    class_ids = np.arange(len(class_names))
    y_bin = label_binarize(y_true, classes=class_ids)

    fpr: dict[str, np.ndarray] = {}
    tpr: dict[str, np.ndarray] = {}
    roc_auc: dict[str, float] = {}
    rows = []

    for idx, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_bin[:, idx], y_proba[:, idx])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        rows.append({"class": class_name, "roc_auc": roc_auc[class_name]})

    roc_auc_macro = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="macro")
    roc_auc_weighted = roc_auc_score(y_bin, y_proba, multi_class="ovr", average="weighted")
    roc_auc_micro = roc_auc_score(y_bin, y_proba, average="micro")

    fpr["micro"], tpr["micro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())

    return {
        "roc_table": pd.DataFrame(rows).sort_values("roc_auc").reset_index(drop=True),
        "fpr": fpr,
        "tpr": tpr,
        "roc_auc": roc_auc,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "roc_auc_micro": roc_auc_micro,
    }
