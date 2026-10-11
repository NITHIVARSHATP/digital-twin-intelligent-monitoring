import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.0005


class MotorFaultNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
    ):
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
    print("Loading dataset...")

    data = pd.read_excel(
        config.DATA_FILE,
        sheet_name="Sheet1",
    )

    data = data.dropna(
        subset=(
            config.FEATURE_COLUMNS
            + [config.TARGET_COLUMN]
        )
    )

    features = data[
        config.FEATURE_COLUMNS
    ].values.astype(np.float32)

    labels = data[
        config.TARGET_COLUMN
    ].values.astype(np.int64)

    classes = np.sort(
        np.unique(labels)
    )

    class_to_index = {
        int(class_value): index
        for index, class_value in enumerate(classes)
    }

    encoded_labels = np.array(
        [
            class_to_index[int(label)]
            for label in labels
        ],
        dtype=np.int64,
    )

    print("Rows:", len(data))
    print(
        "Features:",
        config.FEATURE_COLUMNS,
    )
    print(
        "Classes:",
        classes.tolist(),
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        encoded_labels,
        test_size=0.20,
        random_state=config.RANDOM_STATE,
        stratify=encoded_labels,
    )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    x_train = torch.tensor(
        x_train,
        dtype=torch.float32,
    )

    x_test = torch.tensor(
        x_test,
        dtype=torch.float32,
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long,
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long,
    )

    train_dataset = TensorDataset(
        x_train,
        y_train,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Training device:", device)

    model = MotorFaultNN(
        input_size=len(
            config.FEATURE_COLUMNS
        ),
        num_classes=len(classes),
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    best_accuracy = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_samples = 0
        correct_samples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)

            loss = loss_function(
                outputs,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_labels)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct_samples += (
                predictions == batch_labels
            ).sum().item()

            total_samples += len(
                batch_labels
            )

        model.eval()

        with torch.no_grad():
            test_outputs = model(
                x_test.to(device)
            )

            test_predictions = (
                test_outputs.argmax(dim=1)
            )

            test_accuracy = (
                test_predictions
                == y_test.to(device)
            ).float().mean().item()

        average_loss = (
            total_loss / total_samples
        )

        train_accuracy = (
            correct_samples / total_samples
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.state_dict().items()
                )
            }

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Loss: {average_loss:.4f} | "
                f"Train accuracy: "
                f"{train_accuracy * 100:.2f}% | "
                f"Test accuracy: "
                f"{test_accuracy * 100:.2f}%"
            )

    if best_state is None:
        raise RuntimeError(
            "Training did not produce a model."
        )

    config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_checkpoint = {
        "model_state_dict": best_state,
        "input_size": len(
            config.FEATURE_COLUMNS
        ),
        "num_classes": len(classes),
        "classes": [
            int(class_value)
            for class_value in classes
        ],
        "feature_columns": (
            config.FEATURE_COLUMNS
        ),
    }

    torch.save(
        model_checkpoint,
        config.MODEL_FILE,
    )

    joblib.dump(
        scaler,
        config.SCALER_FILE,
    )

    with open(
        config.CLASSES_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                int(class_value)
                for class_value in classes
            ],
            file,
            indent=2,
        )

    print()
    print("Training completed.")
    print(
        "Best test accuracy:",
        f"{best_accuracy * 100:.2f}%",
    )
    print()
    print("Saved files:")
    print(config.MODEL_FILE)
    print(config.SCALER_FILE)
    print(config.CLASSES_FILE)


if __name__ == "__main__":
    main()