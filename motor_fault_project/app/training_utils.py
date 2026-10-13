"""Shared leakage-safe dataset utilities for model training and evaluation."""

import hashlib
import json
import zipfile
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config


TRAINING_SPLIT_METHOD = "stratified_group_60_20_20_v1"
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


def load_labelled_dataset():
    try:
        data = pd.read_excel(config.DATA_FILE, sheet_name="Sheet1")
    except ImportError:
        data = pd.DataFrame(_read_xlsx_rows(config.DATA_FILE))
    required = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    data = data.dropna(subset=required).copy()
    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=required).reset_index(drop=True)
    if data.empty:
        raise ValueError("The labelled dataset contains no valid rows")
    return data


def _read_xlsx_rows(filepath):
    namespace = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "office": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }
    with zipfile.ZipFile(filepath) as workbook:
        shared_strings = []
        if "xl/sharedStrings.xml" in workbook.namelist():
            shared_root = ElementTree.fromstring(
                workbook.read("xl/sharedStrings.xml")
            )
            shared_strings = [
                "".join(item.itertext())
                for item in shared_root.findall("main:si", namespace)
            ]
        workbook_root = ElementTree.fromstring(
            workbook.read("xl/workbook.xml")
        )
        relationships = ElementTree.fromstring(
            workbook.read("xl/_rels/workbook.xml.rels")
        )
        relationship_targets = {
            relation.attrib["Id"]: relation.attrib["Target"]
            for relation in relationships
        }
        sheet = workbook_root.find("main:sheets/main:sheet", namespace)
        relationship_id = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = relationship_targets[relationship_id].lstrip("/")
        worksheet_path = target if target.startswith("xl/") else f"xl/{target}"
        worksheet = ElementTree.fromstring(workbook.read(worksheet_path))
        rows = []
        for row in worksheet.findall(".//main:sheetData/main:row", namespace):
            values = []
            for cell in row.findall("main:c", namespace):
                value = cell.find("main:v", namespace)
                text = "" if value is None else value.text or ""
                if cell.attrib.get("t") == "s" and text:
                    text = shared_strings[int(text)]
                values.append(text)
            rows.append(values)
    if not rows:
        return []
    headers = rows[0]
    return [dict(zip(headers, row)) for row in rows[1:]]


def dataset_fingerprint(data, split_method=TRAINING_SPLIT_METHOD):
    """Return a stable fingerprint for the labelled data and split contract.

    Rows are normalized and sorted so equivalent workbook/CSV loads produce the
    same identity regardless of source row order. The split contract is part of
    the digest because the same records evaluated under a different methodology
    are not the same experiment.
    """
    columns = config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    canonical_data = data[columns].copy()
    for column in columns:
        canonical_data[column] = pd.to_numeric(
            canonical_data[column],
            errors="raise",
        ).astype(np.float64)
    canonical_data = canonical_data.sort_values(
        columns,
        kind="mergesort",
    ).reset_index(drop=True)
    payload = {
        "columns": columns,
        "rows": json.loads(
            canonical_data.to_json(
                orient="records",
                double_precision=15,
            )
        ),
        "split_method": split_method,
        "random_state": config.RANDOM_STATE,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def split_indices(labels):
    """Return disjoint train, validation, and test indices.

    Exact duplicate feature/label rows share a group, preventing identical
    records from crossing a split. No synthetic grouping identifier is added.
    """
    labels = np.asarray(labels)
    frame = pd.DataFrame({"label": labels})
    groups = np.arange(len(labels))
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_val, test = next(splitter.split(frame, labels, groups))
    train_val_labels = labels[train_val]
    second_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_relative, validation_relative = next(
        second_splitter.split(
            frame.iloc[train_val],
            train_val_labels,
            groups[train_val],
        )
    )
    return (
        train_val[train_relative],
        train_val[validation_relative],
        test,
    )


def split_dataframe(data):
    labels = data[config.TARGET_COLUMN].astype(int).to_numpy()
    row_values = data[
        config.FEATURE_COLUMNS + [config.TARGET_COLUMN]
    ].to_numpy()
    groups = np.array(
        [
            hashlib.sha256(
                json.dumps(row.tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            for row in row_values
        ]
    )
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_val, test = next(splitter.split(data, labels, groups))
    second_splitter = StratifiedGroupKFold(
        n_splits=4,
        shuffle=True,
        random_state=config.RANDOM_STATE,
    )
    train_relative, validation_relative = next(
        second_splitter.split(
            data.iloc[train_val],
            labels[train_val],
            groups[train_val],
        )
    )
    return (
        train_val[train_relative],
        train_val[validation_relative],
        test,
    )
