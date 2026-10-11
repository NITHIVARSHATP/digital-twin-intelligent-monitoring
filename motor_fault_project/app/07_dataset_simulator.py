import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from torch import nn


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


SAMPLE_INTERVAL_SECONDS = 2

# Number of consecutive readings generated
# from one class before the simulator switches.
READINGS_PER_CLASS = 12


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


def load_model():
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

    scaler = joblib.load(
        config.SCALER_FILE
    )

    classes = np.array(
        checkpoint["classes"]
    )

    return model, scaler, classes


def load_dataset():
    data = pd.read_excel(
        config.DATA_FILE,
        sheet_name="Sheet1",
    )

    required_columns = (
        config.FEATURE_COLUMNS
        + [config.TARGET_COLUMN]
    )

    data = data.dropna(
        subset=required_columns
    )

    return data


def create_output_file():
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        config.RUNTIME_DIR
        / "continuous_learning_log.csv"
    )

    if output_file.exists():
        output_file.unlink()

    return output_file


def append_row(output_file, row):
    file_exists = output_file.exists()

    fieldnames = [
        "timestamp",
        *config.FEATURE_COLUMNS,
        "Actual_Label",
        "Predicted_Fault",
        "Confidence",
        "Is_Correct",
    ]

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def predict_fault(
    model,
    scaler,
    classes,
    sensor_data,
):
    input_data = np.array(
        [
            [
                sensor_data[column]
                for column in config.FEATURE_COLUMNS
            ]
        ],
        dtype=np.float32,
    )

    scaled_data = scaler.transform(
        input_data
    )

    input_tensor = torch.tensor(
        scaled_data,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    predicted_index = probabilities.argmax(
        dim=1
    ).item()

    predicted_fault = int(
        classes[predicted_index]
    )

    confidence = float(
        probabilities[
            0,
            predicted_index,
        ].item()
    )

    return predicted_fault, confidence


def main():
    model, scaler, classes = load_model()
    data = load_dataset()
    output_file = create_output_file()

    available_classes = sorted(
        data[config.TARGET_COLUMN]
        .unique()
        .tolist()
    )

    print("Dataset-driven simulator started.")
    print("Available classes:", available_classes)
    print("Readings per class:", READINGS_PER_CLASS)
    print("Output file:", output_file)
    print("Press Ctrl+C to stop.")
    print()

    class_index = 0
    reading_count = 0

    current_class = available_classes[
        class_index
    ]

    while True:
        if reading_count >= READINGS_PER_CLASS:
            class_index = (
                class_index + 1
            ) % len(available_classes)

            current_class = available_classes[
                class_index
            ]

            reading_count = 0

            print()
            print(
                "=" * 60
            )
            print(
                "Switching simulated condition to "
                f"Fault {current_class}"
            )
            print(
                "=" * 60
            )

        class_rows = data[
            data[config.TARGET_COLUMN]
            == current_class
        ]

        selected_row = class_rows.sample(
            n=1,
            replace=False,
        ).iloc[0]

        sensor_data = {
            column: float(
                selected_row[column]
            )
            for column in config.FEATURE_COLUMNS
        }

        actual_label = int(
            selected_row[
                config.TARGET_COLUMN
            ]
        )

        predicted_fault, confidence = (
            predict_fault(
                model,
                scaler,
                classes,
                sensor_data,
            )
        )

        is_correct = int(
            predicted_fault == actual_label
        )

        row = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            **sensor_data,
            "Actual_Label": actual_label,
            "Predicted_Fault": predicted_fault,
            "Confidence": confidence,
            "Is_Correct": is_correct,
        }

        append_row(
            output_file,
            row,
        )

        result = (
            "Correct"
            if is_correct == 1
            else "Incorrect"
        )

        print(
            f"{row['timestamp']} | "
            f"Actual: Fault {actual_label} | "
            f"Predicted: Fault {predicted_fault} | "
            f"Confidence: "
            f"{confidence * 100:.2f}% | "
            f"{result}"
        )

        reading_count += 1

        time.sleep(
            SAMPLE_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nDataset-driven simulator stopped."
        )