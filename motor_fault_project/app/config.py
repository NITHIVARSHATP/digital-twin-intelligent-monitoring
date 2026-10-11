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