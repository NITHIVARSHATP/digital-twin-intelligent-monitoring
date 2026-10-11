import json
import sys
import time
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


RAW_SENSOR_FILE = (
    config.RUNTIME_DIR
    / "real_sensor_input.csv"
)

REAL_OPERATIONAL_FILE = (
    config.RUNTIME_DIR
    / "real_motor_operational_log.csv"
)

POLL_INTERVAL_SECONDS = 0.5


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


def load_production_artifacts():
    required_files = [
        config.MODEL_FILE,
        config.SCALER_FILE,
        config.CLASSES_FILE,
    ]

    missing_files = [
        file_path
        for file_path in required_files
        if not file_path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required production artifacts are "
            f"missing: {missing_files}"
        )

    model_checkpoint = torch.load(
        config.MODEL_FILE,
        map_location="cpu",
        weights_only=False,
    )

    scaler = joblib.load(
        config.SCALER_FILE
    )

    with open(
        config.CLASSES_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        classes = json.load(file)

    classes = [
        int(class_value)
        for class_value in classes
    ]

    if model_checkpoint["feature_columns"] != (
        config.FEATURE_COLUMNS
    ):
        raise ValueError(
            "Production model feature columns "
            "do not match config.py."
        )

    if model_checkpoint["classes"] != classes:
        raise ValueError(
            "Model checkpoint classes do not "
            "match classes.json."
        )

    model = MotorFaultNN(
        input_size=model_checkpoint["input_size"],
        num_classes=model_checkpoint["num_classes"],
    )

    model.load_state_dict(
        model_checkpoint["model_state_dict"]
    )

    model.eval()

    return model, scaler, classes


def validate_raw_data(
    raw_data,
):
    required_columns = (
        ["timestamp"]
        + config.FEATURE_COLUMNS
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in raw_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Raw sensor file is missing columns: "
            f"{missing_columns}"
        )

    valid_data = raw_data.dropna(
        subset=required_columns
    ).copy()

    for feature in config.FEATURE_COLUMNS:
        valid_data[feature] = pd.to_numeric(
            valid_data[feature],
            errors="coerce",
        )

    valid_data = valid_data.dropna(
        subset=config.FEATURE_COLUMNS
    ).copy()

    return valid_data


def predict_rows(
    model,
    scaler,
    classes,
    rows,
):
    features = rows[
        config.FEATURE_COLUMNS
    ].values.astype(np.float32)

    scaled_features = scaler.transform(
        features
    )

    feature_tensor = torch.tensor(
        scaled_features,
        dtype=torch.float32,
    )

    with torch.no_grad():
        outputs = model(feature_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        ).numpy()

    predicted_indexes = probabilities.argmax(
        axis=1
    )

    confidences = probabilities.max(
        axis=1
    )

    predicted_faults = [
        classes[index]
        for index in predicted_indexes
    ]

    prediction_data = rows.copy()

    prediction_data["Predicted_Fault"] = (
        predicted_faults
    )

    prediction_data["Confidence"] = confidences

    prediction_data["Prediction_Source"] = (
        "production_model"
    )

    return prediction_data


def append_predictions(
    prediction_data,
):
    write_header = not REAL_OPERATIONAL_FILE.exists()

    prediction_data.to_csv(
        REAL_OPERATIONAL_FILE,
        mode="a",
        header=write_header,
        index=False,
    )


def main():
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if REAL_OPERATIONAL_FILE.exists():
        REAL_OPERATIONAL_FILE.unlink()

    print("Loading active production model...")

    model, scaler, classes = (
        load_production_artifacts()
    )

    print("Production model loaded.")
    print("Waiting for raw sensor input:")
    print(RAW_SENSOR_FILE)
    print("Writing predictions to:")
    print(REAL_OPERATIONAL_FILE)

    processed_row_count = 0

    try:
        while True:
            if not RAW_SENSOR_FILE.exists():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            try:
                raw_data = pd.read_csv(
                    RAW_SENSOR_FILE
                )

                valid_data = validate_raw_data(
                    raw_data
                )

                new_data = valid_data.iloc[
                    processed_row_count:
                ].copy()

                if new_data.empty:
                    time.sleep(
                        POLL_INTERVAL_SECONDS
                    )
                    continue

                prediction_data = predict_rows(
                    model,
                    scaler,
                    classes,
                    new_data,
                )

                append_predictions(
                    prediction_data
                )

                processed_row_count = len(
                    valid_data
                )

                for _, row in prediction_data.iterrows():
                    print(
                        "Predicted:",
                        row["timestamp"],
                        "| Fault",
                        int(row["Predicted_Fault"]),
                        "| Confidence:",
                        f"{float(row['Confidence']) * 100:.2f}%",
                    )

            except (
                pd.errors.EmptyDataError,
                PermissionError,
            ):
                time.sleep(POLL_INTERVAL_SECONDS)

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print()
        print("Real-time predictor stopped.")


if __name__ == "__main__":
    main()