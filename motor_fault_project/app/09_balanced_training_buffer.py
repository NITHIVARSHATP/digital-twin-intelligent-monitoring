import sys
from pathlib import Path

import pandas as pd


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


OPERATIONAL_FILE = (
    config.RUNTIME_DIR
    / "operational_log.csv"
)

BUFFER_FILE = (
    config.RUNTIME_DIR
    / "verified_training_buffer.csv"
)

MAX_ROWS_PER_CLASS = 250

REQUIRED_CLASSES = [0, 1, 2, 3, 4, 5]

REQUIRED_COLUMNS = [
    *config.FEATURE_COLUMNS,
    "Actual_Label",
]


def main():
    if not OPERATIONAL_FILE.exists():
        raise FileNotFoundError(
            "Operational log not found: "
            f"{OPERATIONAL_FILE}"
        )

    data = pd.read_csv(
        OPERATIONAL_FILE
    )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Operational log is missing columns: "
            f"{missing_columns}"
        )

    data = data.dropna(
        subset=REQUIRED_COLUMNS
    ).copy()

    data["Actual_Label"] = data[
        "Actual_Label"
    ].astype(int)

    available_classes = sorted(
        data["Actual_Label"]
        .unique()
        .tolist()
    )

    missing_classes = [
        class_label
        for class_label in REQUIRED_CLASSES
        if class_label not in available_classes
    ]

    if missing_classes:
        raise ValueError(
            "Cannot create a six-class balanced "
            "buffer. Missing classes: "
            f"{missing_classes}"
        )

    class_counts = (
        data["Actual_Label"]
        .value_counts()
        .sort_index()
    )

    print("Rows before balancing:")
    print(class_counts)

    smallest_class_count = int(
        class_counts[
            REQUIRED_CLASSES
        ].min()
    )

    rows_per_class = min(
        smallest_class_count,
        MAX_ROWS_PER_CLASS,
    )

    print()
    print(
        "Using rows per class:",
        rows_per_class,
    )

    balanced_parts = []

    for class_label in REQUIRED_CLASSES:
        class_rows = data[
            data["Actual_Label"]
            == class_label
        ]

        selected_rows = class_rows.tail(
            rows_per_class
        )

        balanced_parts.append(
            selected_rows
        )

    balanced_data = pd.concat(
        balanced_parts,
        ignore_index=True,
    )

    balanced_data = balanced_data.sample(
        frac=1,
        random_state=config.RANDOM_STATE,
    ).reset_index(drop=True)

    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    balanced_data.to_csv(
        BUFFER_FILE,
        index=False,
    )

    print()
    print("Rows after balancing:")

    print(
        balanced_data["Actual_Label"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Saved balanced training buffer:")
    print(BUFFER_FILE)


if __name__ == "__main__":
    main()