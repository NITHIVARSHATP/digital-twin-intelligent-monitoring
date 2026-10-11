import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


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

    scaler = joblib.load(
        config.SCALER_FILE
    )

    x_test = scaler.transform(x_test)

    checkpoint = torch.load(
        config.MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    model = MotorFaultNN(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    test_tensor = torch.tensor(
        x_test,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = model(test_tensor)
        predictions = outputs.argmax(
            dim=1
        ).numpy()

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    class_names = [
        f"Fault {int(class_value)}"
        for class_value in classes
    ]

    report = classification_report(
        y_test,
        predictions,
        labels=np.arange(len(classes)),
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(classes)),
    )

    print()
    print("Test accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print()
    print("Classification report:")
    print(report)

    print("Confusion matrix:")
    print(matrix)

    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_file = (
        config.RUNTIME_DIR
        / "evaluation_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"Test accuracy: "
            f"{accuracy * 100:.2f}%\n\n"
        )
        file.write(
            "Classification report:\n"
        )
        file.write(report)
        file.write(
            "\nConfusion matrix:\n"
        )
        file.write(str(matrix))

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Motor Fault Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.tight_layout()

    figure_file = (
        config.RUNTIME_DIR
        / "confusion_matrix.png"
    )

    plt.savefig(
        figure_file,
        dpi=200,
    )

    plt.close()

    print()
    print("Saved evaluation report:")
    print(report_file)

    print("Saved confusion matrix:")
    print(figure_file)


if __name__ == "__main__":
    main()