import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch import nn


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


SAMPLE_INTERVAL_SECONDS = 0.5

NORMAL_READINGS = 60
DEGRADATION_READINGS = 20
FAULT_READINGS = 15
RECOVERY_READINGS = 15

FAULT_SEQUENCE = [1, 2, 3, 4, 5]

OUTPUT_FILE = (
    config.RUNTIME_DIR
    / "operational_log.csv"
)


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

    scaler = __import__(
        "joblib"
    ).load(
        config.SCALER_FILE
    )

    classes = np.array(
        checkpoint["classes"]
    )

    return model, scaler, classes


def load_class_profiles():
    data = pd.read_excel(
        config.DATA_FILE,
        sheet_name="Sheet1",
    )

    grouped = data.groupby(
        config.TARGET_COLUMN
    )

    profiles = {}

    for class_value, class_data in grouped:
        profiles[int(class_value)] = {}

        for column in config.FEATURE_COLUMNS:
            profiles[int(class_value)][column] = {
                "mean": float(
                    class_data[column].mean()
                ),
                "std": float(
                    class_data[column].std()
                ),
                "min": float(
                    class_data[column].min()
                ),
                "max": float(
                    class_data[column].max()
                ),
            }

    return profiles


def create_output_file():
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Keep historical operational records.  append_row writes the header only
    # when this is a new file.


def append_row(row):
    file_exists = OUTPUT_FILE.exists()

    fieldnames = [
        "timestamp",
        "Phase",
        "Target_Fault",
        "Actual_Label",
        *config.FEATURE_COLUMNS,
        "Predicted_Fault",
        "Confidence",
        "Is_Correct",
    ]

    with open(
        OUTPUT_FILE,
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


def make_target_reading(
    profiles,
    target_class,
):
    reading = {}

    for column in config.FEATURE_COLUMNS:
        profile = profiles[target_class]

        value = np.random.normal(
            profile[column]["mean"],
            profile[column]["std"] * 0.25,
        )

        value = np.clip(
            value,
            profile[column]["min"],
            profile[column]["max"],
        )

        reading[column] = float(value)

    return reading


def blend_readings(
    previous,
    target,
    alpha,
):
    result = {}

    for column in config.FEATURE_COLUMNS:
        previous_value = previous[column]
        target_value = target[column]

        noise = np.random.normal(
            0,
            abs(target_value) * 0.005
            + 0.01,
        )

        result[column] = (
            previous_value
            + alpha
            * (target_value - previous_value)
            + noise
        )

    return result


def predict(
    model,
    scaler,
    classes,
    sensor_data,
):
    values = np.array(
        [
            [
                sensor_data[column]
                for column in config.FEATURE_COLUMNS
            ]
        ],
        dtype=np.float32,
    )

    scaled_values = scaler.transform(
        values
    )

    tensor = torch.tensor(
        scaled_values,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

    predicted_index = probabilities.argmax(
        dim=1
    ).item()

    predicted_class = int(
        classes[predicted_index]
    )

    confidence = float(
        probabilities[
            0,
            predicted_index,
        ].item()
    )

    return predicted_class, confidence


def main():
    model, scaler, classes = load_model()
    profiles = load_class_profiles()

    create_output_file()

    print(
        "Synthetic real-time simulator started."
    )
    print(
        "Output file:",
        OUTPUT_FILE,
    )
    print(
        "Cycle: normal → degradation → fault "
        "→ recovery"
    )
    print(
        "Fault sequence:",
        FAULT_SEQUENCE,
    )
    print("Press Ctrl+C to stop.")
    print()

    previous_reading = make_target_reading(
        profiles,
        0,
    )

    fault_index = 0

    while True:
        target_fault = FAULT_SEQUENCE[
            fault_index
        ]

        normal_target = make_target_reading(
            profiles,
            0,
        )

        fault_target = make_target_reading(
            profiles,
            target_fault,
        )

        phases = [
            (
                "NORMAL",
                NORMAL_READINGS,
                normal_target,
                0,
            ),
            (
                "DEGRADING",
                DEGRADATION_READINGS,
                fault_target,
                0,
            ),
            (
                "FAULT",
                FAULT_READINGS,
                fault_target,
                target_fault,
            ),
            (
                "RECOVERY",
                RECOVERY_READINGS,
                normal_target,
                0,
            ),
        ]

        for (
            phase_name,
            phase_length,
            target_reading,
            actual_label,
        ) in phases:
            for phase_step in range(
                phase_length
            ):
                if phase_name == "NORMAL":
                    alpha = 0.12
                elif phase_name == "DEGRADING":
                    alpha = 0.08
                elif phase_name == "FAULT":
                    alpha = 0.20
                else:
                    alpha = 0.12

                previous_reading = blend_readings(
                    previous_reading,
                    target_reading,
                    alpha,
                )

                predicted_fault, confidence = (
                    predict(
                        model,
                        scaler,
                        classes,
                        previous_reading,
                    )
                )

                definitive_label = (
                    actual_label
                    if phase_name in {"NORMAL", "FAULT"}
                    else None
                )
                is_correct = (
                    int(predicted_fault == definitive_label)
                    if definitive_label is not None
                    else None
                )

                row = {
                    "timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "Phase": phase_name,
                    "Target_Fault": target_fault,
                    "Actual_Label": definitive_label,
                    **previous_reading,
                    "Predicted_Fault": predicted_fault,
                    "Confidence": confidence,
                    "Is_Correct": is_correct,
                }

                append_row(row)

                print(
                    f"{row['timestamp']} | "
                    f"Phase: {phase_name:<10} | "
                    f"Target: Fault {target_fault} | "
                    f"Actual: Fault {actual_label} | "
                    f"Predicted: Fault "
                    f"{predicted_fault} | "
                    f"Confidence: "
                    f"{confidence * 100:.1f}% | "
                    f"Temp1: "
                    f"{previous_reading['Temp1']:.1f}"
                )

                time.sleep(
                    SAMPLE_INTERVAL_SECONDS
                )

        print()
        print(
            f"Completed cycle for Fault "
            f"{target_fault}"
        )
        print()

        fault_index = (
            fault_index + 1
        ) % len(FAULT_SEQUENCE)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            "\nSynthetic simulator stopped."
        )