import sys
from pathlib import Path

import pandas as pd


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


data = pd.read_excel(
    config.DATA_FILE,
    sheet_name="Sheet1",
)

profiles = (
    data.groupby(config.TARGET_COLUMN)[
        config.FEATURE_COLUMNS
    ]
    .agg(["count", "mean", "std", "min", "max"])
    .round(3)
)

means = (
    data.groupby(config.TARGET_COLUMN)[
        config.FEATURE_COLUMNS
    ]
    .mean()
    .round(3)
)

print("Average sensor values by class:")
print(means)

print()
print("Detailed profiles by class:")
print(profiles)

output_file = (
    config.RUNTIME_DIR
    / "class_sensor_profiles.xlsx"
)

config.RUNTIME_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with pd.ExcelWriter(output_file) as writer:
    means.to_excel(
        writer,
        sheet_name="Class_Means",
    )

    profiles.to_excel(
        writer,
        sheet_name="Detailed_Profiles",
    )

print()
print("Saved profile file:")
print(output_file)