import csv
import sys
from pathlib import Path


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config
import model_governance


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

def promote_candidate(reviewer_name, review_notes):
    """Promote through the shared atomic, versioned governance workflow."""
    result = model_governance.promote_candidate(
        reviewer_name,
        review_notes,
    )
    return result["promotion"]["backup_path"]


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
        backup_path = promote_candidate(
            reviewer_name,
            review_notes,
        )

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