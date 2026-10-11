import csv
import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


REVIEW_LOG_FILE = (
    config.RUNTIME_DIR
    / "model_review_log.csv"
)

FIELDNAMES = [
    "Review_Timestamp",
    "Candidate_Model",
    "Production_Accuracy",
    "Candidate_Accuracy",
    "Accuracy_Change_Percentage_Points",
    "Production_Fault_4_Recall",
    "Candidate_Fault_4_Recall",
    "Fault_4_Recall_Change_Percentage_Points",
    "System_Decision",
    "Reviewer_Decision",
    "Reviewer_Name",
    "Review_Notes",
]


def main():
    config.RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_row = {
        "Review_Timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "Candidate_Model": (
            "candidate_motor_model.pth"
        ),
        "Production_Accuracy": "97.55",
        "Candidate_Accuracy": "97.00",
        "Accuracy_Change_Percentage_Points": "-0.55",
        "Production_Fault_4_Recall": "92.11",
        "Candidate_Fault_4_Recall": "97.37",
        "Fault_4_Recall_Change_Percentage_Points": (
            "5.26"
        ),
        "System_Decision": "REVIEW_REQUIRED",
        "Reviewer_Decision": "PENDING",
        "Reviewer_Name": "",
        "Review_Notes": (
            "Candidate improved Fault 4 recall but "
            "reduced overall accuracy. Awaiting "
            "human approval."
        ),
    }

    file_exists = REVIEW_LOG_FILE.exists()

    with open(
        REVIEW_LOG_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(review_row)

    print("Saved model review record:")
    print(REVIEW_LOG_FILE)

    print()
    print("System decision:")
    print(review_row["System_Decision"])

    print("Reviewer decision:")
    print(review_row["Reviewer_Decision"])


if __name__ == "__main__":
    main()