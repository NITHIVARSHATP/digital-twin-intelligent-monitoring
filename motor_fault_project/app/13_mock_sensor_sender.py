import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


RAW_SENSOR_FILE = (
    config.RUNTIME_DIR
    / "real_sensor_input.csv"
)

SAMPLE_INTERVAL_SECONDS = 1.0

RANDOM_STATE = 42

NORMAL_MEANS = {
    "Voltage": 220.0,
    "Current": 5.0,
    "RPM": 1500.0,
    "Temp1": 42.0,
    "Temp2": 40.0,
    "X": 0.20,
    "Y": 0.20,
    "Z": 0.20,
}

NORMAL_STDS = {
    "Voltage": 2.0,
    "Current": 0.30,
    "RPM": 25.0,
    "Temp1": 1.5,
    "Temp2": 1.5,
    "X": 0.05,
    "Y": 0.05,
    "Z": 0.05,
}


def generate_reading(
    random_generator,
):
    reading = {
        "timestamp": datetime.now().isoformat(
            timespec="milliseconds"
        ),
    }

    for feature in config.FEATURE_COLUMNS:
        reading[feature] = float(
            random_generator.normal(
                loc=NORMAL_MEANS[feature],
                scale=NORMAL_STDS[feature],
            )
        )

    return reading


def main():
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    if RAW_SENSOR_FILE.exists():
        RAW_SENSOR_FILE.unlink()

    print("Starting mock real-sensor sender...")
    print("Writing raw measurements to:")
    print(RAW_SENSOR_FILE)

    try:
        while True:
            reading = generate_reading(
                random_generator
            )

            frame = pd.DataFrame(
                [reading]
            )

            write_header = not RAW_SENSOR_FILE.exists()

            frame.to_csv(
                RAW_SENSOR_FILE,
                mode="a",
                header=write_header,
                index=False,
            )

            print(
                "Sent raw sensor reading:",
                reading["timestamp"],
            )

            time.sleep(
                SAMPLE_INTERVAL_SECONDS
            )

    except KeyboardInterrupt:
        print()
        print("Mock sensor sender stopped.")


if __name__ == "__main__":
    main()