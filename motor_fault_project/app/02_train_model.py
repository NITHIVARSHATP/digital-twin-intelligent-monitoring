"""Train the production motor model with leakage-safe checkpoint selection."""

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from training_utils import TRAINING_SPLIT_METHOD, dataset_fingerprint, load_labelled_dataset, split_dataframe


BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.0005
REPORT_FILE = config.RUNTIME_DIR / "evaluation_report_leakage_safe.txt"


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


def set_random_seed():
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_STATE)


def create_loader(features, labels):
    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(config.RANDOM_STATE)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )


def main():
    set_random_seed()
    started = time.perf_counter()
    data = load_labelled_dataset()
    train_indices, validation_indices, test_indices = split_dataframe(data)
    classes = np.sort(data[config.TARGET_COLUMN].astype(int).unique())
    class_to_index = {int(value): index for index, value in enumerate(classes)}

    features = data[config.FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    labels = np.array(
        [class_to_index[int(value)] for value in data[config.TARGET_COLUMN]],
        dtype=np.int64,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(features[train_indices])
    x_validation = scaler.transform(features[validation_indices])
    x_test = scaler.transform(features[test_indices])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotorFaultNN(len(config.FEATURE_COLUMNS), len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()
    train_loader = create_loader(x_train, labels[train_indices])
    validation_tensor = torch.tensor(x_validation, dtype=torch.float32).to(device)
    validation_labels = torch.tensor(labels[validation_indices], dtype=torch.long).to(device)
    best_validation_accuracy = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features.to(device))
            loss_function(outputs, batch_labels.to(device)).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_predictions = model(validation_tensor).argmax(dim=1)
        validation_accuracy = accuracy_score(
            validation_labels.cpu().numpy(),
            validation_predictions.cpu().numpy(),
        )
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} | "
            f"Validation accuracy: {validation_accuracy * 100:.2f}%"
        )

    if best_state is None:
        raise RuntimeError("Training did not produce a validation-selected model")

    model.load_state_dict(best_state)
    model.eval()
    test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        test_predictions = model(test_tensor).argmax(dim=1).cpu().numpy()
    test_accuracy = accuracy_score(labels[test_indices], test_predictions)
    duration = time.perf_counter() - started

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": best_state,
        "input_size": len(config.FEATURE_COLUMNS),
        "num_classes": len(classes),
        "classes": [int(value) for value in classes],
        "feature_columns": config.FEATURE_COLUMNS,
        "training_methodology": TRAINING_SPLIT_METHOD,
        "dataset_fingerprint": dataset_fingerprint(data),
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "test_rows": len(test_indices),
        "best_validation_accuracy": float(best_validation_accuracy),
        "final_test_accuracy": float(test_accuracy),
        "test_used_for_model_selection": False,
    }
    torch.save(checkpoint, config.MODEL_FILE)
    joblib.dump(scaler, config.SCALER_FILE)
    with config.CLASSES_FILE.open("w", encoding="utf-8") as file:
        json.dump([int(value) for value in classes], file, indent=2)

    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_FILE.open("w", encoding="utf-8") as file:
        file.write("Leakage-Safe Production Training Evaluation\n")
        file.write("===========================================\n\n")
        file.write(f"Dataset fingerprint: {checkpoint['dataset_fingerprint']}\n")
        file.write(f"Split methodology: {TRAINING_SPLIT_METHOD}\n")
        file.write(f"Training records: {len(train_indices)}\n")
        file.write(f"Validation records: {len(validation_indices)}\n")
        file.write(f"Test records: {len(test_indices)}\n")
        file.write(f"Best validation accuracy: {best_validation_accuracy * 100:.2f}%\n")
        file.write(f"Final test accuracy: {test_accuracy * 100:.2f}%\n")
        file.write(f"Training duration seconds: {duration:.3f}\n")
        file.write("Test set untouched during model selection: True\n")
        file.write(
            "Known limitations: exact duplicate feature/label rows are grouped; "
            "unrecorded provenance relationships cannot be inferred.\n"
        )
    print(f"Saved leakage-safe report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
