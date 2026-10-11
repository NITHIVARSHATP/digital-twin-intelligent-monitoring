import csv
import shutil
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

CANDIDATE_MODEL_FILE = (
    config.MODEL_DIR
    / "candidate_motor_model.pth"
)

CANDIDATE_SCALER_FILE = (
    config.MODEL_DIR
    / "candidate_scaler.pkl"
)

CANDIDATE_CLASSES_FILE = (
    config.MODEL_DIR
    / "candidate_classes.json"
)

BACKUP_DIR = (
    config.MODEL_DIR
    / "backups"
)

VALID_DECISIONS = [
    "APPROVE",
    "REJECT",
    "REQUEST_MORE_DATA",
]


def get_user_decision():
    print("Available reviewer decisions:")

    for decision in VALID_DECISIONS:
        print(f"- {decision}")

    print()

    decision = input(
        "Enter reviewer decision: "
    ).strip().upper()

    if decision not in VALID_DECISIONS:
        raise ValueError(
            "Invalid reviewer decision. "
            "Choose one of: "
            f"{VALID_DECISIONS}"
        )

    reviewer_name = input(
        "Enter reviewer name: "
    ).strip()

    review_notes = input(
        "Enter review notes: "
    ).strip()

    if not reviewer_name:
        raise ValueError(
            "Reviewer name cannot be empty."
        )

    if not review_notes:
        raise ValueError(
            "Review notes cannot be empty."
        )

    return (
        decision,
        reviewer_name,
        review_notes,
    )


def update_latest_review(
    decision,
    reviewer_name,
    review_notes,
):
    if not REVIEW_LOG_FILE.exists():
        raise FileNotFoundError(
            "Review log not found. Run "
            "11_log_model_review.py first: "
            f"{REVIEW_LOG_FILE}"
        )

    with open(
        REVIEW_LOG_FILE,
        "r",
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        fieldnames = reader.fieldnames

        rows = list(reader)

    if not rows:
        raise ValueError(
            "Review log is empty."
        )

    rows[-1]["Reviewer_Decision"] = decision
    rows[-1]["Reviewer_Name"] = reviewer_name
    rows[-1]["Review_Notes"] = review_notes

    with open(
        REVIEW_LOG_FILE,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

def backup_production_files():
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR
        / f"production_{timestamp}"
    )

    backup_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    production_files = [
        config.MODEL_FILE,
        config.SCALER_FILE,
        config.CLASSES_FILE,
    ]

    for source_file in production_files:
        if not source_file.exists():
            raise FileNotFoundError(
                "Production file not found: "
                f"{source_file}"
            )

        shutil.copy2(
            source_file,
            backup_path / source_file.name,
        )

    return backup_path


def promote_candidate():
    candidate_files = [
        CANDIDATE_MODEL_FILE,
        CANDIDATE_SCALER_FILE,
        CANDIDATE_CLASSES_FILE,
    ]

    for candidate_file in candidate_files:
        if not candidate_file.exists():
            raise FileNotFoundError(
                "Candidate file not found: "
                f"{candidate_file}"
            )

    backup_path = backup_production_files()

    replacement_pairs = [
        (
            CANDIDATE_MODEL_FILE,
            config.MODEL_FILE,
        ),
        (
            CANDIDATE_SCALER_FILE,
            config.SCALER_FILE,
        ),
        (
            CANDIDATE_CLASSES_FILE,
            config.CLASSES_FILE,
        ),
    ]

    for source_file, destination_file in replacement_pairs:
        shutil.copy2(
            source_file,
            destination_file,
        )

    return backup_path


def main():
    (
        decision,
        reviewer_name,
        review_notes,
    ) = get_user_decision()

    update_latest_review(
        decision,
        reviewer_name,
        review_notes,
    )

    print()
    print("Updated latest model review record.")

    if decision == "APPROVE":
        backup_path = promote_candidate()

        print()
        print("Candidate promoted to production.")
        print("Previous production files backed up at:")
        print(backup_path)

    elif decision == "REJECT":
        print()
        print(
            "Candidate rejected. Production "
            "files were not changed."
        )

    else:
        print()
        print(
            "More data requested. Production "
            "files were not changed."
        )


if __name__ == "__main__":
    main()