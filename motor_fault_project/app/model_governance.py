"""Background retraining and versioned model governance services."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib
import numpy as np
import torch

import config
from prediction_service import MotorFaultNN


STATUS_FILE = config.RUNTIME_DIR / "retraining_status.json"
REGISTRY_FILE = config.RUNTIME_DIR / "model_registry.json"
HISTORY_FILE = config.RUNTIME_DIR / "training_history.json"
AUDIT_FILE = config.RUNTIME_DIR / "model_audit_log.csv"
CANDIDATE_REPORT_FILE = config.RUNTIME_DIR / "candidate_comparison_report_leakage_safe.txt"
CANDIDATE_METADATA_FILE = config.RUNTIME_DIR / "candidate_metadata.json"
MODEL_COMPARISON_REPORT_FILE = config.RUNTIME_DIR / "model_comparison_leakage_safe.txt"
PRODUCTION_REPORT_FILE = config.RUNTIME_DIR / "evaluation_report_leakage_safe.txt"
BUFFER_FILE = config.RUNTIME_DIR / "verified_training_buffer.csv"
OPERATOR_HEADER = "X-Model-Operator"

_lock = threading.RLock()
_scheduler_thread = None
_training_process = None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_json(path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _write_json(path, value):
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _append_audit(event, actor, details):
    config.RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "event", "actor", "details"]
    exists = AUDIT_FILE.exists()
    with AUDIT_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": _now(),
            "event": event,
            "actor": actor,
            "details": json.dumps(details, sort_keys=True),
        })


def _unique_labelled_fingerprint():
    if not BUFFER_FILE.exists():
        return 0, None
    try:
        data = pd.read_csv(BUFFER_FILE)
        required = config.FEATURE_COLUMNS + ["Actual_Label"]
        if any(column not in data.columns for column in required):
            return 0, None
        data = data.dropna(subset=required)[required].copy()
        data = data.drop_duplicates().sort_values(required).reset_index(drop=True)
        digest = hashlib.sha256(
            data.to_json(orient="records", double_precision=15).encode("utf-8")
        ).hexdigest()
        return len(data), f"sha256:{digest}"
    except (OSError, ValueError, TypeError):
        return 0, None


def _default_status():
    return {
        "state": "idle",
        "last_trigger": None,
        "last_started": None,
        "last_completed": None,
        "last_error": None,
        "pid": None,
        "eligible_labelled_records": 0,
        "last_labelled_fingerprint": None,
        "manual_requests": 0,
    }


def _status():
    value = _read_json(STATUS_FILE, _default_status())
    result = _default_status()
    result.update(value)
    return result


def _registry():
    value = _read_json(REGISTRY_FILE, {})
    if not value:
        value = {
            "active_model": {
                "version": "production",
                "artifact": str(config.MODEL_FILE),
                "activated_at": None,
            },
            "candidate": None,
            "versions": [],
        }
    return value


def _history():
    value = _read_json(HISTORY_FILE, [])
    return value if isinstance(value, list) else []


def _candidate_summary():
    metadata = _read_json(CANDIDATE_METADATA_FILE, None)
    if metadata:
        return metadata
    if not CANDIDATE_REPORT_FILE.exists():
        return None
    return {
        "report": str(CANDIDATE_REPORT_FILE),
        "decision": "REVIEW_REQUIRED",
        "metrics_source": "candidate_comparison_report_leakage_safe.txt",
    }


def _active_production_metrics():
    metrics = {
        "accuracy": None,
        "macro_f1": None,
        "source": str(PRODUCTION_REPORT_FILE),
    }
    if PRODUCTION_REPORT_FILE.exists():
        report = PRODUCTION_REPORT_FILE.read_text(encoding="utf-8")
        patterns = {
            "accuracy": r"^Accuracy:\s*([0-9.]+)%",
            "macro_f1": r"^Macro F1:\s*([0-9.]+)%",
        }
        for name, pattern in patterns.items():
            match = re.search(pattern, report, flags=re.MULTILINE)
            if match:
                metrics[name] = float(match.group(1)) / 100
    if metrics["accuracy"] is None or metrics["macro_f1"] is None:
        checkpoint = _read_json(config.MODEL_DIR / "production_metadata.json", {})
        metrics["accuracy"] = metrics["accuracy"] or checkpoint.get(
            "final_test_accuracy"
        )
        metrics["macro_f1"] = metrics["macro_f1"] or checkpoint.get(
            "final_test_macro_f1"
        )
    if metrics["accuracy"] is None or metrics["macro_f1"] is None:
        raise RuntimeError(
            "Active production metrics are unavailable; promotion is blocked"
        )
    return metrics


def _launch_training(trigger, actor):
    global _training_process
    with _lock:
        status = _status()
        if status["state"] == "running":
            return False
        command = [
            sys.executable,
            str(Path(__file__).with_name("10_candidate_retraining.py")),
        ]
        try:
            _training_process = subprocess.Popen(
                command,
                cwd=str(config.PROJECT_ROOT),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as error:
            status.update({
                "state": "failed",
                "last_error": str(error),
                "last_completed": _now(),
            })
            _write_json(STATUS_FILE, status)
            _append_audit("retraining_launch_failed", actor, {"error": str(error)})
            return False
        status.update({
            "state": "running",
            "last_trigger": trigger,
            "last_started": _now(),
            "last_error": None,
            "pid": _training_process.pid,
        })
        _write_json(STATUS_FILE, status)
        _append_audit("retraining_started", actor, {"trigger": trigger})
        threading.Thread(
            target=_wait_for_training,
            args=(_training_process, trigger, actor),
            daemon=True,
        ).start()
        return True


def _wait_for_training(process, trigger, actor):
    global _training_process
    exit_code = process.wait()
    with _lock:
        status = _status()
        count, fingerprint = _unique_labelled_fingerprint()
        if exit_code == 0:
            candidate = _candidate_summary() or {}
            history = _history()
            history.append({
                "timestamp": _now(),
                "trigger": trigger,
                "status": "completed",
                "candidate": candidate,
                "eligible_labelled_records": count,
                "labelled_fingerprint": fingerprint,
            })
            _write_json(HISTORY_FILE, history[-100:])
            status.update({
                "state": "completed",
                "last_completed": _now(),
                "last_error": None,
                "pid": None,
                "eligible_labelled_records": count,
                "last_labelled_fingerprint": fingerprint,
            })
            _append_audit("retraining_completed", actor, {
                "trigger": trigger,
                "exit_code": exit_code,
                "candidate": candidate,
            })
        else:
            status.update({
                "state": "failed",
                "last_completed": _now(),
                "last_error": f"Candidate retraining exited with code {exit_code}",
                "pid": None,
            })
            _append_audit("retraining_failed", actor, {
                "trigger": trigger,
                "exit_code": exit_code,
            })
        _write_json(STATUS_FILE, status)
        _training_process = None


def _scheduler_loop():
    while True:
        time.sleep(config.RETRAINING_POLL_SECONDS)
        with _lock:
            status = _status()
            count, fingerprint = _unique_labelled_fingerprint()
            status["eligible_labelled_records"] = count
            if status["state"] == "running":
                _write_json(STATUS_FILE, status)
                continue
            last = status.get("last_started")
            elapsed = True
            if last:
                try:
                    elapsed = (
                        time.time()
                        - datetime.fromisoformat(last).timestamp()
                        >= config.RETRAINING_INTERVAL_SECONDS
                    )
                except ValueError:
                    elapsed = True
            new_labels = fingerprint and fingerprint != status.get("last_labelled_fingerprint")
            count_trigger = bool(
                new_labels and count >= config.RETRAINING_MIN_LABELLED_RECORDS
            )
            elapsed_trigger = bool(new_labels and elapsed)
            if count_trigger or elapsed_trigger:
                _write_json(STATUS_FILE, status)
                _launch_training(
                    "scheduled_label_count" if count_trigger else "scheduled_elapsed",
                    "scheduler",
                )
            else:
                _write_json(STATUS_FILE, status)


def start_scheduler():
    global _scheduler_thread
    with _lock:
        if _scheduler_thread and _scheduler_thread.is_alive():
            return
        _scheduler_thread = threading.Thread(
            target=_scheduler_loop,
            name="model-retraining-scheduler",
            daemon=True,
        )
        _scheduler_thread.start()


def status_payload():
    with _lock:
        status = _status()
        registry = _registry()
        count, fingerprint = _unique_labelled_fingerprint()
        status["eligible_labelled_records"] = count
        return {
            "status": "ok",
            "retraining": status,
            "registry": registry,
            "candidate": _candidate_summary(),
            "model_comparison": (
                MODEL_COMPARISON_REPORT_FILE.read_text(encoding="utf-8")
                if MODEL_COMPARISON_REPORT_FILE.exists()
                else None
            ),
            "history": _history()[-20:][::-1],
            "criteria": {
                "interval_seconds": config.RETRAINING_INTERVAL_SECONDS,
                "minimum_labelled_records": config.RETRAINING_MIN_LABELLED_RECORDS,
                "approval_required": config.RETRAINING_APPROVAL_REQUIRED,
                "minimum_accuracy": config.RETRAINING_MIN_CANDIDATE_ACCURACY,
                "minimum_class_recall": config.RETRAINING_MIN_CLASS_RECALL,
                "maximum_accuracy_drop": config.RETRAINING_MAX_ACCURACY_DROP,
                "maximum_macro_f1_drop": config.RETRAINING_MAX_MACRO_F1_DROP,
            },
            "labelled_fingerprint": fingerprint,
        }


def request_retraining(actor, trigger="manual"):
    if not actor:
        raise PermissionError("Authorized operator identity is required")
    with _lock:
        status = _status()
        status["manual_requests"] = int(status.get("manual_requests", 0)) + 1
        _write_json(STATUS_FILE, status)
    started = _launch_training(trigger, actor)
    if not started:
        raise RuntimeError("Retraining is already running")
    result = status_payload()
    return result


def _artifact_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote_candidate(actor, notes):
    if not actor:
        raise PermissionError("Authorized reviewer identity is required")
    if not notes:
        raise ValueError("Approval notes are required")
    candidate_files = [
        config.MODEL_DIR / "candidate_motor_model.pth",
        config.MODEL_DIR / "candidate_scaler.pkl",
        config.MODEL_DIR / "candidate_classes.json",
    ]
    if any(not path.exists() for path in candidate_files):
        raise FileNotFoundError("Candidate artifacts are incomplete")
    if config.RETRAINING_APPROVAL_REQUIRED is not True:
        raise PermissionError("Promotion approval policy is disabled")
    candidate = _candidate_summary() or {}
    production = _active_production_metrics()
    candidate_accuracy = float(candidate.get("accuracy", 0))
    candidate_macro_f1 = float(candidate.get("macro_f1", 0))
    accuracy_drop = float(production["accuracy"]) - candidate_accuracy
    macro_f1_drop = float(production["macro_f1"]) - candidate_macro_f1
    minimum_class_recall = float(candidate.get("minimum_class_recall", 0))
    if (
        not candidate.get("eligible_for_promotion", False)
        or candidate_accuracy < config.RETRAINING_MIN_CANDIDATE_ACCURACY
        or accuracy_drop > config.RETRAINING_MAX_ACCURACY_DROP
        or macro_f1_drop > config.RETRAINING_MAX_MACRO_F1_DROP
        or minimum_class_recall < config.RETRAINING_MIN_CLASS_RECALL
    ):
        raise PermissionError(
            "Candidate does not meet the configured acceptance criteria"
        )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = config.MODEL_DIR / "backups" / f"production_{timestamp}"
    backup.mkdir(parents=True, exist_ok=False)
    production_files = [
        config.MODEL_FILE,
        config.SCALER_FILE,
        config.CLASSES_FILE,
    ]
    for path in production_files:
        if not path.exists():
            raise FileNotFoundError(f"Production artifact is missing: {path}")
        shutil.copy2(path, backup / path.name)

    for source, destination in zip(candidate_files, production_files):
        temporary = destination.with_suffix(destination.suffix + ".candidate")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)

    version = f"production-{timestamp}"
    registry = _registry()
    registry.setdefault("versions", []).append({
        "version": version,
        "previous_version": registry.get("active_model", {}).get(
            "version",
            "production",
        ),
        "activated_at": _now(),
        "backup_path": str(backup),
        "actor": actor,
        "notes": notes,
        "artifact_hashes": {path.name: _artifact_hash(path) for path in production_files},
    })
    registry["active_model"] = {
        "version": version,
        "artifact": str(config.MODEL_FILE),
        "activated_at": _now(),
    }
    registry["candidate"] = None
    _write_json(REGISTRY_FILE, registry)
    _append_audit("model_promoted", actor, {
        "version": version,
        "backup_path": str(backup),
        "notes": notes,
    })
    return status_payload()


def rollback(actor, version):
    if not actor:
        raise PermissionError("Authorized reviewer identity is required")
    registry = _registry()
    selected = next(
        (item for item in registry.get("versions", []) if item.get("version") == version),
        None,
    )
    if not selected:
        raise ValueError("Requested production version was not found")
    backup = Path(selected["backup_path"])
    if not backup.exists():
        raise FileNotFoundError("Version backup is missing")
    for name, destination in (
        ("motor_model.pth", config.MODEL_FILE),
        ("scaler.pkl", config.SCALER_FILE),
        ("classes.json", config.CLASSES_FILE),
    ):
        temporary = destination.with_suffix(destination.suffix + ".rollback")
        shutil.copy2(backup / name, temporary)
        os.replace(temporary, destination)
    registry["active_model"] = {
        "version": selected.get("previous_version", "production"),
        "artifact": str(config.MODEL_FILE),
        "activated_at": _now(),
    }
    _write_json(REGISTRY_FILE, registry)
    _append_audit("model_rolled_back", actor, {"version": version})
    return status_payload()


def test_single_record(model_name, sensor_data):
    """Test a record against an isolated artifact set without changing inference."""
    if model_name not in {"production", "candidate"}:
        raise ValueError("model must be production or candidate")
    if any(field not in sensor_data for field in config.FEATURE_COLUMNS):
        raise ValueError("All model features are required")
    if model_name == "production":
        model_file, scaler_file, classes_file = (
            config.MODEL_FILE,
            config.SCALER_FILE,
            config.CLASSES_FILE,
        )
    else:
        model_file = config.MODEL_DIR / "candidate_motor_model.pth"
        scaler_file = config.MODEL_DIR / "candidate_scaler.pkl"
        classes_file = config.MODEL_DIR / "candidate_classes.json"
    if any(not path.exists() for path in (model_file, scaler_file, classes_file)):
        raise FileNotFoundError(f"{model_name} model artifacts are incomplete")
    checkpoint = torch.load(model_file, map_location="cpu", weights_only=False)
    scaler = joblib.load(scaler_file)
    classes = json.loads(classes_file.read_text(encoding="utf-8"))
    model = MotorFaultNN(checkpoint["input_size"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    values = np.array(
        [[float(sensor_data[field]) for field in config.FEATURE_COLUMNS]],
        dtype=np.float32,
    )
    with torch.no_grad():
        probabilities = torch.softmax(
            model(torch.tensor(scaler.transform(values), dtype=torch.float32)),
            dim=1,
        )[0]
    index = int(probabilities.argmax().item())
    return {
        "model": model_name,
        "predicted_fault": int(classes[index]),
        "confidence": float(probabilities[index].item()),
        "model_version": checkpoint.get(
            "model_version",
            f"{model_name}-artifact",
        ),
    }
