import csv
import json
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


def generate_sensor_reading():
    return {
        "Voltage": random.uniform(225, 230),
        "Current": random.uniform(8, 16),
        "RPM": random.uniform(1400, 1800),
        "Temp1": random.uniform(25, 55),
        "Temp2": random.uniform(25, 55),
        "X": random.uniform(-5, 5),
        "Y": random.uniform(-5, 5),
        "Z": random.uniform(7, 13),
    }


def append_prediction(row):
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        config.LIVE_DATA_FILE.exists()
    )

    fieldnames = [
        "timestamp",
        *config.FEATURE_COLUMNS,
        "Predicted_Fault",
        "Confidence",
    ]

    with open(
        config.LIVE_DATA_FILE,
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


def main():
    model, scaler, classes = load_model()

    print("Live simulator started.")
    print(
        "Writing predictions to:",
        config.LIVE_DATA_FILE,
    )
    print("Press Ctrl+C to stop.")
    print()

    while True:
        sensor_data = generate_sensor_reading()

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

        row = {
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            **sensor_data,
            "Predicted_Fault": predicted_fault,
            "Confidence": confidence,
        }

        append_prediction(row)

        print(
            f"{row['timestamp']} | "
            f"Fault: {predicted_fault} | "
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

        time.sleep(
            SAMPLE_INTERVAL_SECONDS
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLive simulator stopped.")