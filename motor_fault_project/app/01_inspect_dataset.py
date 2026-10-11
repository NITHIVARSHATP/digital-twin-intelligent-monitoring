from pathlib import Path

import pandas as pd


DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "Dataset_augmented.xlsx"
)


print("Dataset file:")
print(DATA_FILE)
print()


if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )


excel_file = pd.ExcelFile(DATA_FILE)

print("Excel sheets:")
print(excel_file.sheet_names)
print()


for sheet_name in excel_file.sheet_names:
    print("=" * 70)
    print(f"SHEET: {sheet_name}")

    data = pd.read_excel(
        DATA_FILE,
        sheet_name=sheet_name,
    )

    print()
    print("Rows and columns:")
    print(data.shape)

    print()
    print("Column names:")
    print(list(data.columns))

    print()
    print("First five rows:")
    print(data.head())

    print()
    print("Missing values:")
    print(data.isna().sum())

    print()
    print("Data types:")
    print(data.dtypes)

    print()
    print("Unique values for possible label columns:")

    for column in data.columns:
        column_name = str(column).lower()

        if any(
            keyword in column_name
            for keyword in [
                "fault",
                "label",
                "class",
                "target",
                "condition",
            ]
        ):
            print(
                f"{column}: "
                f"{data[column].dropna().unique()[:20]}"
            )

    print()