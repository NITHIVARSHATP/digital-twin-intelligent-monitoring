"""Evaluate a production artifact on the reserved leakage-safe test split."""

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from training_utils import TRAINING_SPLIT_METHOD, dataset_fingerprint, load_labelled_dataset, split_dataframe


REPORT_FILE = config.RUNTIME_DIR / "evaluation_report_leakage_safe.txt"
MATRIX_FILE = config.RUNTIME_DIR / "confusion_matrix_leakage_safe.csv"


class MotorFaultNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)


def main():
    data = load_labelled_dataset()
    _, _, test_indices = split_dataframe(data)
    checkpoint = torch.load(config.MODEL_FILE, map_location="cpu", weights_only=False)
    scaler = joblib.load(config.SCALER_FILE)
    classes = np.array(checkpoint["classes"], dtype=int)
    class_to_index = {int(value): index for index, value in enumerate(classes)}
    labels = np.array(
        [class_to_index[int(value)] for value in data[config.TARGET_COLUMN]],
        dtype=np.int64,
    )
    features = data[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    test_features = scaler.transform(features[test_indices])

    model = MotorFaultNN(checkpoint["input_size"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    inputs = torch.tensor(test_features, dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        predictions = model(inputs).argmax(dim=1).numpy()
    inference_seconds = time.perf_counter() - started
    latency_ms = inference_seconds / len(test_indices) * 1000

    test_labels = labels[test_indices]
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels,
        predictions,
        labels=np.arange(len(classes)),
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=np.arange(len(classes)),
    )
    report = classification_report(
        test_labels,
        predictions,
        labels=np.arange(len(classes)),
        target_names=[f"Fault {int(value)}" for value in classes],
        zero_division=0,
    )

    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    matrix_frame = np.asarray(matrix)
    np.savetxt(MATRIX_FILE, matrix_frame, delimiter=",", fmt="%d")
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write("Leakage-Safe Production Evaluation\n")
        file.write("=================================\n\n")
        file.write(f"Dataset fingerprint: {dataset_fingerprint(data)}\n")
        file.write(f"Split methodology: {TRAINING_SPLIT_METHOD}\n")
        file.write(
            f"Training records: {checkpoint.get('train_rows', 'UNKNOWN')}\n"
        )
        file.write(
            f"Validation records: {checkpoint.get('validation_rows', 'UNKNOWN')}\n"
        )
        file.write(f"Test records: {len(test_indices)}\n")
        file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
        file.write(f"Macro precision: {precision.mean() * 100:.2f}%\n")
        file.write(f"Macro recall: {recall.mean() * 100:.2f}%\n")
        file.write(f"Macro F1: {f1.mean() * 100:.2f}%\n")
        file.write(f"Inference latency milliseconds per record: {latency_ms:.4f}\n")
        file.write(
            "Test set untouched during model selection: "
            f"{checkpoint.get('test_used_for_model_selection', 'UNKNOWN') is False}\n"
        )
        file.write(
            "Known limitations: this evaluates the current artifact; artifacts "
            "without leakage-safe metadata cannot prove historical test isolation.\n\n"
        )
        file.write("Per-class metrics\n")
        for index, value in enumerate(classes):
            file.write(
                f"Fault {int(value)} | Precision: {precision[index] * 100:.2f}% | "
                f"Recall: {recall[index] * 100:.2f}% | F1: {f1[index] * 100:.2f}%\n"
            )
        file.write("\nClassification report\n")
        file.write(report)
        file.write("\nConfusion matrix file: " + str(MATRIX_FILE) + "\n")
    print(f"Saved leakage-safe report: {REPORT_FILE}")
    print(f"Saved confusion matrix: {MATRIX_FILE}")


if __name__ == "__main__":
    main()
