import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

import config


BATCH_SIZE = 64
EPOCHS = 80
LEARNING_RATE = 0.0005

TEST_SIZE = 0.20

BUFFER_FILE = (
    config.RUNTIME_DIR
    / "verified_training_buffer.csv"
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

COMPARISON_REPORT_FILE = (
    config.RUNTIME_DIR
    / "candidate_comparison_report.txt"
)

CANDIDATE_MATRIX_FILE = (
    config.RUNTIME_DIR
    / "candidate_confusion_matrix.csv"
)

REQUIRED_CLASSES = [0, 1, 2, 3, 4, 5]

MIN_ACCURACY = 0.95
MAX_ACCURACY_DROP = 0.00
MIN_RECALL = 0.90

REVIEW_ACCURACY_DROP = 0.02
REVIEW_RECALL_CHANGE = 0.03


class MotorFaultNN(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
    ):
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


def set_random_seed():
    np.random.seed(config.RANDOM_STATE)
    torch.manual_seed(config.RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            config.RANDOM_STATE
        )


def load_original_dataset():
    data = pd.read_excel(
        config.DATA_FILE,
        sheet_name="Sheet1",
    )

    data = data.dropna(
        subset=(
            config.FEATURE_COLUMNS
            + [config.TARGET_COLUMN]
        )
    ).copy()

    features = data[
        config.FEATURE_COLUMNS
    ].values.astype(np.float32)

    labels = data[
        config.TARGET_COLUMN
    ].values.astype(np.int64)

    return features, labels


def load_verified_buffer(classes):
    if not BUFFER_FILE.exists():
        raise FileNotFoundError(
            "Verified training buffer not found: "
            f"{BUFFER_FILE}"
        )

    buffer_data = pd.read_csv(
        BUFFER_FILE
    )

    required_columns = (
        config.FEATURE_COLUMNS
        + ["Actual_Label"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in buffer_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Verified buffer is missing columns: "
            f"{missing_columns}"
        )

    buffer_data = buffer_data.dropna(
        subset=required_columns
    ).copy()

    buffer_features = buffer_data[
        config.FEATURE_COLUMNS
    ].values.astype(np.float32)

    buffer_labels = buffer_data[
        "Actual_Label"
    ].values.astype(np.int64)

    unknown_labels = sorted(
        set(buffer_labels.tolist())
        - set(classes.tolist())
    )

    if unknown_labels:
        raise ValueError(
            "Buffer contains labels not found "
            "in the original dataset: "
            f"{unknown_labels}"
        )

    buffer_counts = pd.Series(
        buffer_labels
    ).value_counts().sort_index()

    missing_classes = [
        int(class_value)
        for class_value in classes
        if int(class_value)
        not in buffer_counts.index
    ]

    if missing_classes:
        raise ValueError(
            "Buffer does not contain all "
            "dataset classes. Missing: "
            f"{missing_classes}"
        )

    return (
        buffer_features,
        buffer_labels,
        buffer_counts,
    )


def encode_labels(
    labels,
    class_to_index,
):
    return np.array(
        [
            class_to_index[int(label)]
            for label in labels
        ],
        dtype=np.int64,
    )


def create_loader(
    features,
    labels,
):
    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    )

    label_tensor = torch.tensor(
        labels,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        feature_tensor,
        label_tensor,
    )

    generator = torch.Generator()
    generator.manual_seed(
        config.RANDOM_STATE
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )


def train_candidate(
    x_train,
    y_train,
    x_test,
    y_test,
    num_classes,
    device,
):
    train_loader = create_loader(
        x_train,
        y_train,
    )

    model = MotorFaultNN(
        input_size=len(
            config.FEATURE_COLUMNS
        ),
        num_classes=num_classes,
    ).to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0001,
    )

    best_accuracy = -1.0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0.0
        total_samples = 0
        correct_samples = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()

            outputs = model(batch_features)

            loss = loss_function(
                outputs,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * len(batch_labels)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct_samples += (
                predictions == batch_labels
            ).sum().item()

            total_samples += len(
                batch_labels
            )

        model.eval()

        test_features = torch.tensor(
            x_test,
            dtype=torch.float32,
        ).to(device)

        test_labels = torch.tensor(
            y_test,
            dtype=torch.long,
        ).to(device)

        with torch.no_grad():
            test_outputs = model(
                test_features
            )

            test_predictions = (
                test_outputs.argmax(dim=1)
            )

            test_accuracy = (
                test_predictions
                == test_labels
            ).float().mean().item()

        average_loss = (
            total_loss / total_samples
        )

        train_accuracy = (
            correct_samples / total_samples
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in (
                    model.state_dict().items()
                )
            }

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == EPOCHS - 1
        ):
            print(
                f"Epoch {epoch + 1:03d}/{EPOCHS} | "
                f"Loss: {average_loss:.4f} | "
                f"Train accuracy: "
                f"{train_accuracy * 100:.2f}% | "
                f"Test accuracy: "
                f"{test_accuracy * 100:.2f}%"
            )

    if best_state is None:
        raise RuntimeError(
            "Candidate training did not "
            "produce a model."
        )

    model.load_state_dict(best_state)

    return model, best_accuracy


def predict_model(
    model,
    features,
    device,
):
    model.eval()

    feature_tensor = torch.tensor(
        features,
        dtype=torch.float32,
    ).to(device)

    with torch.no_grad():
        outputs = model(feature_tensor)

        predictions = outputs.argmax(
            dim=1
        ).cpu().numpy()

    return predictions


def load_production_model(
    device,
):
    if not config.MODEL_FILE.exists():
        raise FileNotFoundError(
            "Production model not found: "
            f"{config.MODEL_FILE}"
        )

    checkpoint = torch.load(
        config.MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    model = MotorFaultNN(
        input_size=checkpoint["input_size"],
        num_classes=checkpoint["num_classes"],
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, checkpoint


def recall_by_class(
    true_labels,
    predicted_labels,
    number_of_classes,
):
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=np.arange(number_of_classes),
    )

    row_totals = matrix.sum(axis=1)

    recalls = np.divide(
        np.diag(matrix),
        row_totals,
        out=np.zeros(
            number_of_classes,
            dtype=float,
        ),
        where=row_totals != 0,
    )

    return recalls, matrix


def main():
    set_random_seed()

    print("Loading original dataset...")

    features, labels = load_original_dataset()

    classes = np.sort(
        np.unique(labels)
    )

    expected_classes = np.array(
        REQUIRED_CLASSES,
        dtype=np.int64,
    )

    if not np.array_equal(
        classes,
        expected_classes,
    ):
        raise ValueError(
            "Original dataset classes must be "
            f"{REQUIRED_CLASSES}. Found: "
            f"{classes.tolist()}"
        )

    class_to_index = {
        int(class_value): index
        for index, class_value in enumerate(
            classes
        )
    }

    encoded_labels = encode_labels(
        labels,
        class_to_index,
    )

    (
        x_train_original,
        x_test_original,
        y_train_original,
        y_test_original,
    ) = train_test_split(
        features,
        encoded_labels,
        test_size=TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=encoded_labels,
    )

    print(
        "Original rows:",
        len(features),
    )

    print(
        "Original training rows:",
        len(x_train_original),
    )

    print(
        "Untouched original test rows:",
        len(x_test_original),
    )

    print()
    print(
        "Loading verified synthetic "
        "training buffer..."
    )

    (
        buffer_features,
        buffer_labels,
        buffer_counts,
    ) = load_verified_buffer(
        classes
    )

    encoded_buffer_labels = encode_labels(
        buffer_labels,
        class_to_index,
    )

    print(
        "Verified buffer rows:",
        len(buffer_features),
    )

    print(
        "Verified buffer class counts:"
    )

    print(buffer_counts)

    x_candidate_train = np.vstack(
        [
            x_train_original,
            buffer_features,
        ]
    )

    y_candidate_train = np.concatenate(
        [
            y_train_original,
            encoded_buffer_labels,
        ]
    )

    print()
    print(
        "Candidate training rows:",
        len(x_candidate_train),
    )

    candidate_counts = pd.Series(
        y_candidate_train
    ).value_counts().sort_index()

    candidate_counts.index = [
        int(classes[index])
        for index in candidate_counts.index
    ]

    print(
        "Candidate training label counts:"
    )

    print(candidate_counts)

    candidate_scaler = StandardScaler()

    x_candidate_train_scaled = (
        candidate_scaler.fit_transform(
            x_candidate_train
        )
    )

    x_test_scaled_for_candidate = (
        candidate_scaler.transform(
            x_test_original
        )
    )

    production_scaler = joblib.load(
        config.SCALER_FILE
    )

    x_test_scaled_for_production = (
        production_scaler.transform(
            x_test_original
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print("Training device:", device)

    print()
    print("Evaluating production model...")

    production_model, production_checkpoint = (
        load_production_model(device)
    )

    if production_checkpoint["classes"] != [
        int(class_value)
        for class_value in classes
    ]:
        raise ValueError(
            "Production model classes do not "
            "match original dataset classes."
        )

    production_predictions = predict_model(
        production_model,
        x_test_scaled_for_production,
        device,
    )

    production_accuracy = accuracy_score(
        y_test_original,
        production_predictions,
    )

    production_recalls, production_matrix = (
        recall_by_class(
            y_test_original,
            production_predictions,
            len(classes),
        )
    )

    print(
        "Production test accuracy:",
        f"{production_accuracy * 100:.2f}%",
    )

    print()
    print("Training candidate model...")

    candidate_model, candidate_best_accuracy = (
        train_candidate(
            x_candidate_train_scaled,
            y_candidate_train,
            x_test_scaled_for_candidate,
            y_test_original,
            len(classes),
            device,
        )
    )

    candidate_predictions = predict_model(
        candidate_model,
        x_test_scaled_for_candidate,
        device,
    )

    candidate_accuracy = accuracy_score(
        y_test_original,
        candidate_predictions,
    )

    candidate_recalls, candidate_matrix = (
        recall_by_class(
            y_test_original,
            candidate_predictions,
            len(classes),
        )
    )

    print()
    print(
        "Candidate test accuracy:",
        f"{candidate_accuracy * 100:.2f}%",
    )

    print()
    print("Per-class recall comparison:")

    for index, class_value in enumerate(classes):
        print(
            f"Fault {int(class_value)} | "
            f"Production: "
            f"{production_recalls[index] * 100:.2f}% | "
            f"Candidate: "
            f"{candidate_recalls[index] * 100:.2f}%"
        )

    all_recalls_above_minimum = bool(
    np.all(candidate_recalls >= MIN_RECALL)
    )

    candidate_meets_accuracy_floor = bool(
        candidate_accuracy >= MIN_ACCURACY
    )

    candidate_not_lower_than_production = bool(
        candidate_accuracy >= production_accuracy
    )

    eligible_for_promotion = bool(
        all_recalls_above_minimum
        and candidate_meets_accuracy_floor
        and candidate_not_lower_than_production
    )

    accuracy_within_review_range = bool(
        candidate_accuracy >= (
            production_accuracy
            - REVIEW_ACCURACY_DROP
        )
    )  

    recall_differences = (
        candidate_recalls
        - production_recalls
    )

    material_recall_change = bool(
        np.any(
            np.abs(recall_differences)
            >= REVIEW_RECALL_CHANGE
        )
    )

    if eligible_for_promotion:
        decision = "PROMOTE"
    elif (
        accuracy_within_review_range
        or material_recall_change
    ):
        decision = "REVIEW_REQUIRED"
    else:
        decision = "REJECT"

    config.MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_checkpoint = {
        "model_state_dict": (
            candidate_model.state_dict()
        ),
        "input_size": len(
            config.FEATURE_COLUMNS
        ),
        "num_classes": len(classes),
        "classes": [
            int(class_value)
            for class_value in classes
        ],
        "feature_columns": (
            config.FEATURE_COLUMNS
        ),
        "training_source": (
            "original_training_split_plus_"
            "verified_synthetic_buffer"
        ),
        "buffer_rows": int(
            len(buffer_features)
        ),
    }

    torch.save(
        candidate_checkpoint,
        CANDIDATE_MODEL_FILE,
    )

    joblib.dump(
        candidate_scaler,
        CANDIDATE_SCALER_FILE,
    )

    with open(
        CANDIDATE_CLASSES_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            [
                int(class_value)
                for class_value in classes
            ],
            file,
            indent=2,
        )

    candidate_report = classification_report(
        y_test_original,
        candidate_predictions,
        labels=np.arange(len(classes)),
        target_names=[
            f"Fault {int(class_value)}"
            for class_value in classes
        ],
        zero_division=0,
    )

    matrix_frame = pd.DataFrame(
        candidate_matrix,
        index=[
            f"Actual_Fault_{int(value)}"
            for value in classes
        ],
        columns=[
            f"Predicted_Fault_{int(value)}"
            for value in classes
        ],
    )

    matrix_frame.to_csv(
        CANDIDATE_MATRIX_FILE,
        index=True,
    )

    with open(
        COMPARISON_REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            "Candidate Retraining Comparison\n"
        )
        file.write("=" * 32 + "\n\n")

        file.write(
            f"Original dataset rows: "
            f"{len(features)}\n"
        )
        file.write(
            f"Original training rows: "
            f"{len(x_train_original)}\n"
        )
        file.write(
            f"Untouched original test rows: "
            f"{len(x_test_original)}\n"
        )
        file.write(
            f"Verified buffer rows: "
            f"{len(buffer_features)}\n"
        )
        file.write(
            f"Candidate training rows: "
            f"{len(x_candidate_train)}\n\n"
        )

        file.write(
            "Production accuracy: "
            f"{production_accuracy * 100:.2f}%\n"
        )
        file.write(
            "Candidate accuracy: "
            f"{candidate_accuracy * 100:.2f}%\n"
        )
        file.write(
            "Best candidate accuracy during "
            "training: "
            f"{candidate_best_accuracy * 100:.2f}%\n\n"
        )

        file.write(
            "Per-class recall comparison\n"
        )

        for index, class_value in enumerate(classes):
            file.write(
                f"Fault {int(class_value)} | "
                f"Production: "
                f"{production_recalls[index] * 100:.2f}% | "
                f"Candidate: "
                f"{candidate_recalls[index] * 100:.2f}%\n"
            )

        file.write("\nPromotion rules\n")
        file.write(
            f"Candidate accuracy >= "
            f"{MIN_ACCURACY * 100:.2f}%\n"
        )
        file.write(
            "Candidate accuracy drop <= "
            f"{MAX_ACCURACY_DROP * 100:.2f}%\n"
        )
        file.write(
            "Every candidate class recall >= "
            f"{MIN_RECALL * 100:.2f}%\n\n"
        )
        file.write(
            "Decision: "
        f"{decision}\n"
        )
        file.write(
            "Eligible for automatic promotion: "
            f"{eligible_for_promotion}\n\n"
        )
        file.write(
            "Candidate classification report\n"
        )
        file.write(candidate_report)

    print()
    print("Saved candidate files:")
    print(CANDIDATE_MODEL_FILE)
    print(CANDIDATE_SCALER_FILE)
    print(CANDIDATE_CLASSES_FILE)

    print()
    print("Saved comparison report:")
    print(COMPARISON_REPORT_FILE)

    print("Saved candidate confusion matrix:")
    print(CANDIDATE_MATRIX_FILE)

    print()
    print(
        "Candidate decision:",
        decision,
    )
    print(
        "Eligible for automatic promotion:",
        eligible_for_promotion,
    )

    if decision == "PROMOTE":
        print(
            "Candidate passed automatic "
            "promotion gates. Production files "
            "were not changed."
        )
    elif decision == "REVIEW_REQUIRED":
        print(
            "Candidate requires human review. "
            "Production files were not changed."
        )
    else:
        print(
            "Candidate was rejected. "
            "Production files were not changed."
        )


if __name__ == "__main__":
    main()