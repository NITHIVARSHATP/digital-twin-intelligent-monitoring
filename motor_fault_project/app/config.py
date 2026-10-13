import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "Dataset_augmented.xlsx"
)

MODEL_DIR = PROJECT_ROOT / "models"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

MODEL_FILE = MODEL_DIR / "motor_model.pth"
SCALER_FILE = MODEL_DIR / "scaler.pkl"
CLASSES_FILE = MODEL_DIR / "classes.json"

LIVE_DATA_FILE = (
    RUNTIME_DIR
    / "live_predictions.csv"
)

FEATURE_COLUMNS = [
    "Voltage",
    "Current",
    "RPM",
    "Temp1",
    "Temp2",
    "X",
    "Y",
    "Z",
]

TARGET_COLUMN = "Fault Status"

RANDOM_STATE = 42

# Model governance policy. These values are intentionally configurable through
# environment variables so demonstrations do not require source changes.
RETRAINING_INTERVAL_SECONDS = int(
    os.getenv("MODEL_RETRAINING_INTERVAL_SECONDS", "86400")
)
RETRAINING_MIN_LABELLED_RECORDS = int(
    os.getenv("MODEL_RETRAINING_MIN_LABELLED_RECORDS", "250")
)
RETRAINING_POLL_SECONDS = int(
    os.getenv("MODEL_RETRAINING_POLL_SECONDS", "30")
)
RETRAINING_APPROVAL_REQUIRED = (
    os.getenv("MODEL_RETRAINING_APPROVAL_REQUIRED", "true").lower()
    == "true"
)
RETRAINING_MIN_CANDIDATE_ACCURACY = float(
    os.getenv("MODEL_RETRAINING_MIN_ACCURACY", "0.95")
)
RETRAINING_MAX_ACCURACY_DROP = float(
    os.getenv("MODEL_RETRAINING_MAX_ACCURACY_DROP", "0.00")
)
RETRAINING_MAX_MACRO_F1_DROP = float(
    os.getenv("MODEL_RETRAINING_MAX_MACRO_F1_DROP", "0.00")
)
RETRAINING_MIN_CLASS_RECALL = float(
    os.getenv("MODEL_RETRAINING_MIN_CLASS_RECALL", "0.90")
)
MODEL_OPERATOR_TOKEN = os.getenv(
    "MODEL_OPERATOR_TOKEN",
    "local-development-operator",
)

# Firebase Realtime Database
FIREBASE_DATABASE_URL = (
    "https://predictive-maintenance-23249-default-rtdb.asia-southeast1.firebasedatabase.app"
)
FIREBASE_CREDENTIALS_FILE = PROJECT_ROOT / "firebase_credentials.json"