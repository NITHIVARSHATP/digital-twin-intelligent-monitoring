"""Production model inference for validated motor telemetry records."""

import hashlib
import json
import threading
from pathlib import Path

import joblib
import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

import config


if nn is not None:
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


class ProductionPredictor:
    """Loads production artifacts once and predicts validated sensor records."""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.classes = None
        self.version = None
        self.load_error = None
        self._lock = threading.RLock()
        self._load()

    @property
    def available(self):
        return self.load_error is None

    def _load(self):
        if torch is None:
            self.load_error = (
                "PyTorch is not installed; production inference is unavailable"
            )
            return

        required_files = (
            config.MODEL_FILE,
            config.SCALER_FILE,
            config.CLASSES_FILE,
        )
        missing = [path for path in required_files if not path.exists()]
        if missing:
            self.load_error = (
                "Production model artifacts are missing: "
                + ", ".join(str(path) for path in missing)
            )
            return

        try:
            checkpoint = torch.load(
                config.MODEL_FILE,
                map_location="cpu",
                weights_only=False,
            )
            scaler = joblib.load(config.SCALER_FILE)
            with config.CLASSES_FILE.open("r", encoding="utf-8") as file:
                classes = [int(value) for value in json.load(file)]

            if checkpoint.get("feature_columns") != config.FEATURE_COLUMNS:
                raise ValueError(
                    "Production model feature columns do not match config.py"
                )
            if checkpoint.get("classes") != classes:
                raise ValueError(
                    "Production model classes do not match classes.json"
                )

            model = MotorFaultNN(
                input_size=checkpoint["input_size"],
                num_classes=checkpoint["num_classes"],
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.model = model
            self.scaler = scaler
            self.classes = classes
            self.version = checkpoint.get(
                "model_version",
                self._artifact_version(required_files),
            )
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            self.load_error = f"Production model could not be loaded: {exc}"

    def reload(self):
        """Atomically replace the in-memory model after artifact promotion."""
        with self._lock:
            self.model = None
            self.scaler = None
            self.classes = None
            self.version = None
            self.load_error = None
            self._load()
            return self.available

    @staticmethod
    def _artifact_version(paths):
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
        return f"sha256:{digest.hexdigest()[:12]}"

    def predict(self, sensor_data):
        with self._lock:
            if not self.available:
                return {
                "Predicted_Fault": None,
                "Confidence": None,
                "prediction_source": None,
                "model_version": None,
                "prediction_error": self.load_error,
            }

            missing = [
                field
                for field in config.FEATURE_COLUMNS
                if field not in sensor_data
            ]
            if missing:
                return {
                "Predicted_Fault": None,
                "Confidence": None,
                "prediction_source": None,
                "model_version": self.version,
                "prediction_error": f"Missing model features: {missing}",
            }

            try:
                values = np.array(
                [[float(sensor_data[field]) for field in config.FEATURE_COLUMNS]],
                dtype=np.float32,
            )
                scaled_values = self.scaler.transform(values)
                inputs = torch.tensor(scaled_values, dtype=torch.float32)
                with torch.no_grad():
                    probabilities = torch.softmax(self.model(inputs), dim=1)[0]
                predicted_index = int(probabilities.argmax().item())
                return {
                "Predicted_Fault": int(self.classes[predicted_index]),
                "Confidence": float(probabilities[predicted_index].item()),
                "prediction_source": "production_model",
                "model_version": self.version,
                "prediction_error": None,
            }
            except (TypeError, ValueError, RuntimeError) as exc:
                return {
                "Predicted_Fault": None,
                "Confidence": None,
                "prediction_source": None,
                "model_version": self.version,
                "prediction_error": f"Prediction failed: {exc}",
            }
