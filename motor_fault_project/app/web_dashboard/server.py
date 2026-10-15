"""
Flask API server for the Motor Intelligence Dashboard.
Reads operational CSV data produced by the simulator
and serves it as JSON for the web frontend.
"""

import json
import sys
import threading
import time
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS


# ── Path setup ───────────────────────────────────────────────

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1]),
)

import config
import firebase_manager
from prediction_service import ProductionPredictor
import model_governance


app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates",
)

CORS(app)


SIMULATION_DATA_FILE = (
    config.RUNTIME_DIR / "operational_log.csv"
)

REAL_MOTOR_DATA_FILE = (
    config.RUNTIME_DIR / "real_motor_operational_log.csv"
)

REVIEW_LOG_FILE = (
    config.RUNTIME_DIR / "model_review_log.csv"
)

REQUIRED_CONSECUTIVE_ABNORMAL = 3
SIMULATION_INTERVAL_SECONDS = 1.5
SIMULATION_RECORD_LIMIT = 250

SIMULATION_THRESHOLDS = {
    "Temp1": {"warning": 55.0, "critical": 75.0},
    "Temp2": {"warning": 55.0, "critical": 75.0},
    "Current": {"warning": 16.0, "critical": 22.0},
    "RPM": {"warning": 1750.0, "critical": 2200.0},
    # X/Y/Z are stored on the same scale as the training workbook.  The
    # observed normal magnitude is about 9.9, so the previous 1.8/2.8
    # limits made nearly every row critical.
    "Vibration": {"warning": 16.0, "critical": 22.0},
}

SENSOR_UNITS = {
    "Voltage": "V",
    "Current": "A",
    "RPM": "rpm",
    "Temp1": "°C",
    "Temp2": "°C",
    "X": "g",
    "Y": "g",
    "Z": "g",
}

SIMULATION_COLUMNS = [
    *config.FEATURE_COLUMNS,
    config.TARGET_COLUMN,
]

simulation_state = {
    "status": "stopped",
    "selected_fault": None,
    "row_index": 0,
    "dataset_rows": [],
    "records": deque(maxlen=SIMULATION_RECORD_LIMIT),
    "thread": None,
}
simulation_lock = threading.RLock()
production_predictor = ProductionPredictor()
model_governance.start_scheduler()

FAULT_LABEL_MAP = {
    0: "Normal",
    1: "Bearing Fault",
    2: "Stator Fault",
    3: "Rotor Fault",
    4: "Overload",
    5: "Misalignment",
}


# ── Helpers ──────────────────────────────────────────────────

def get_data_file(mode):
    if mode == "real_motor":
        return REAL_MOTOR_DATA_FILE
    if mode == "simulation":
        return SIMULATION_DATA_FILE
    raise ValueError(f"Unsupported dashboard mode: {mode}")


def get_requested_mode():
    mode = request.args.get("mode", "simulation")
    if mode not in {"simulation", "real_motor", "firebase"}:
        return None
    return mode


def safe_read_csv(filepath):
    if not filepath.exists():
        return None
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def load_simulation_dataset():
    """Load the source without writing to it and support XLSX, CSV, and TSV."""
    filepath = config.DATA_FILE
    if not filepath.exists():
        raise FileNotFoundError(f"Simulation dataset not found: {filepath}")

    if filepath.suffix.lower() in {".xlsx", ".xls"}:
        try:
            data = pd.read_excel(filepath)
        except ImportError:
            data = pd.DataFrame(_read_xlsx_rows(filepath))
    else:
        sample = filepath.read_text(encoding="utf-8-sig")[:4096]
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        data = pd.read_csv(filepath, sep=delimiter)

    missing = [column for column in SIMULATION_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Simulation dataset is missing columns: {missing}")

    data = data[SIMULATION_COLUMNS].copy()
    for column in SIMULATION_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if data.isna().any().any():
        raise ValueError("Simulation dataset contains missing or non-numeric values")
    return data.to_dict("records")


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
        worksheet_path = (
            target if target.startswith("xl/") else f"xl/{target}"
        )
        worksheet = ElementTree.fromstring(
            workbook.read(worksheet_path)
        )
        rows = []
        for row in worksheet.findall(
            ".//main:sheetData/main:row",
            namespace,
        ):
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


def simulation_fault_label(fault_status):
    if fault_status is None:
        return "Unavailable"
    return FAULT_LABEL_MAP.get(
        int(fault_status),
        f"Fault {int(fault_status)}",
    )


def simulation_alert(sensor_data):
    warnings = []
    critical = False
    for field in ("Temp1", "Temp2", "Current", "RPM"):
        value = sensor_data.get(field)
        if value is None:
            continue
        limits = SIMULATION_THRESHOLDS[field]
        if value >= limits["critical"]:
            warnings.append(f"{field} critical")
            critical = True
        elif value >= limits["warning"]:
            warnings.append(f"{field} warning")

    vibration_values = [
        sensor_data.get(axis)
        for axis in ("X", "Y", "Z")
    ]
    if any(value is None or not np.isfinite(value)
           for value in vibration_values):
        vibration = None
    else:
        vibration = float(np.sqrt(sum(value ** 2 for value in vibration_values)))
    vibration_limits = SIMULATION_THRESHOLDS["Vibration"]
    if vibration is not None and vibration >= vibration_limits["critical"]:
        warnings.append("Vibration critical")
        critical = True
    elif vibration is not None and vibration >= vibration_limits["warning"]:
        warnings.append("Vibration warning")

    if critical:
        return "Critical", warnings
    if warnings:
        return "Warning", warnings
    return "Normal", warnings


def _threshold_status(value, limits):
    if value is None or not np.isfinite(value):
        return "Unavailable"
    if value >= limits["critical"]:
        return "Critical"
    if value >= limits["warning"]:
        return "Warning"
    return "Normal"


def diagnostic_warning_status(severity, predicted_fault):
    if severity == "Critical":
        return "CRITICAL ALERT"
    if severity == "Warning":
        return "SENSOR THRESHOLD WARNING"
    if predicted_fault not in (None, 0):
        return "POSSIBLE ABNORMALITY"
    return "NORMAL"


def build_diagnostic_snapshot(
    sensors,
    predicted_fault,
    confidence,
    *,
    expected_fault=None,
    phase=None,
    source="unknown",
    record_id=None,
    timestamp=None,
):
    validated = {}
    invalid_fields = []
    for field in config.FEATURE_COLUMNS:
        value = sensors.get(field)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        if value is None or not np.isfinite(value):
            invalid_fields.append(field)
        validated[field] = value

    if invalid_fields:
        raise ValueError(
            "Invalid sensor values: " + ", ".join(invalid_fields)
        )

    severity, warnings = simulation_alert(validated)
    warning_status = diagnostic_warning_status(severity, predicted_fault)
    diagnostic = compute_health_and_diagnostics(
        validated,
        predicted_fault,
        confidence or 0.0,
        warning_status,
    )
    sensor_condition = {
        field: _threshold_status(
            validated[field],
            SIMULATION_THRESHOLDS[field],
        )
        for field in ("Temp1", "Temp2", "Current", "RPM")
    }
    vibration = diagnostic["vib_magnitude"]
    sensor_condition["Vibration"] = _threshold_status(
        vibration,
        SIMULATION_THRESHOLDS["Vibration"],
    )
    return {
        "record_id": record_id,
        "timestamp": timestamp,
        "source": source,
        "phase": phase,
        "expected_fault": expected_fault,
        "predicted_fault": predicted_fault,
        "confidence": confidence,
        "sensor_condition": sensor_condition,
        "motor_operating_state": (
            "STOPPED" if validated["RPM"] == 0 else "OPERATING"
        ),
        "alert_severity": severity,
        "threshold_warnings": warnings,
        "warning_status": warning_status,
        "sensor_units": SENSOR_UNITS,
        "thresholds": SIMULATION_THRESHOLDS,
        **diagnostic,
    }


def simulation_record(row):
    sensors = {
        column: float(row[column])
        for column in config.FEATURE_COLUMNS
    }
    fault_status = int(row[config.TARGET_COLUMN])
    prediction = production_predictor.predict(sensors)
    predicted_fault = prediction["Predicted_Fault"]
    timestamp = datetime.now().isoformat(timespec="seconds")
    record_id = f"simulation-{simulation_state['row_index']}"
    diagnostic = build_diagnostic_snapshot(
        sensors,
        predicted_fault,
        prediction["Confidence"],
        expected_fault=fault_status,
        phase="DATASET_PLAYBACK",
        source=prediction["prediction_source"],
        record_id=record_id,
        timestamp=timestamp,
    )
    return {
        "timestamp": timestamp,
        "record_id": record_id,
        **sensors,
        "Fault Status": fault_status,
        "fault_status": fault_status,
        "expected_fault": fault_status,
        "expected_label": simulation_fault_label(fault_status),
        "fault_label": simulation_fault_label(fault_status),
        "Predicted_Fault": predicted_fault,
        "predicted_fault": predicted_fault,
        "Confidence": prediction["Confidence"],
        "ml_prediction": predicted_fault,
        "ml_label": simulation_fault_label(predicted_fault),
        "prediction_match": (
            predicted_fault == fault_status
            if predicted_fault is not None
            else None
        ),
        "prediction_source": prediction["prediction_source"],
        "model_version": prediction["model_version"],
        "prediction_error": prediction["prediction_error"],
        "alert_severity": diagnostic["alert_severity"],
        "threshold_warnings": diagnostic["threshold_warnings"],
        "sensor_condition": diagnostic["sensor_condition"],
        "motor_operating_state": diagnostic["motor_operating_state"],
        "diagnostic": diagnostic,
    }


def simulation_worker():
    while True:
        with simulation_lock:
            if simulation_state["status"] != "running":
                return
            rows = simulation_state["dataset_rows"]
            if not rows:
                simulation_state["status"] = "stopped"
                return
            row = rows[
                simulation_state["row_index"] % len(rows)
            ]
            simulation_state["row_index"] += 1
            simulation_state["records"].append(simulation_record(row))
        time.sleep(SIMULATION_INTERVAL_SECONDS)


def simulation_status():
    with simulation_lock:
        if not simulation_state["dataset_rows"]:
            try:
                simulation_state["dataset_rows"] = load_simulation_dataset()
            except (FileNotFoundError, ImportError, ValueError):
                simulation_state["dataset_rows"] = []
        latest = (
            simulation_state["records"][-1]
            if simulation_state["records"]
            else None
        )
        return {
            "status": simulation_state["status"],
            "selected_fault": simulation_state["selected_fault"],
            "record_count": len(simulation_state["records"]),
            "latest": latest,
            "available_faults": sorted({
                int(row[config.TARGET_COLUMN])
                for row in simulation_state["dataset_rows"]
            }),
            "interval_seconds": SIMULATION_INTERVAL_SECONDS,
            "record_limit": SIMULATION_RECORD_LIMIT,
            "thresholds": SIMULATION_THRESHOLDS,
        }


def start_simulation(selected_fault=None):
    rows = load_simulation_dataset()
    if selected_fault is not None:
        rows = [
            row for row in rows
            if int(row[config.TARGET_COLUMN]) == selected_fault
        ]
        if not rows:
            raise ValueError("Selected fault status is not present in dataset")

    with simulation_lock:
        simulation_state["dataset_rows"] = rows
        simulation_state["selected_fault"] = selected_fault
        simulation_state["row_index"] = 0
        simulation_state["records"].clear()
        simulation_state["status"] = "running"
        worker = threading.Thread(
            target=simulation_worker,
            name="motor-dataset-simulator",
            daemon=True,
        )
        simulation_state["thread"] = worker
        worker.start()


def set_simulation_status(status):
    with simulation_lock:
        if status == "resume":
            if not simulation_state["dataset_rows"]:
                raise ValueError("Simulation has not been started")
            simulation_state["status"] = "running"
            worker = threading.Thread(
                target=simulation_worker,
                name="motor-dataset-simulator",
                daemon=True,
            )
            simulation_state["thread"] = worker
            worker.start()
        elif status == "pause":
            if simulation_state["status"] == "running":
                simulation_state["status"] = "paused"
        elif status == "stop":
            simulation_state["status"] = "stopped"
        else:
            raise ValueError(f"Unsupported simulation action: {status}")


def compute_warning(data, predicted_fault):
    abnormal = (data["Predicted_Fault"] != 0).astype(int)
    consecutive = 0
    for v in reversed(abnormal.tolist()):
        if v == 1:
            consecutive += 1
        else:
            break

    if consecutive >= REQUIRED_CONSECUTIVE_ABNORMAL:
        return "EARLY WARNING", consecutive
    if predicted_fault != 0:
        return "POSSIBLE ABNORMALITY", consecutive
    return "NORMAL", consecutive


def classify_severity(fault, confidence):
    if confidence > 0.85 and fault >= 3:
        return "Critical"
    if confidence > 0.7:
        return "Major"
    return "Minor"


FAULT_GUIDE_DATA = {
    0: {
        "title": "Normal Operation (Healthy)",
        "badge_color": "#00E396",
        "icon": "✅",
        "component": "All Subsystems Nominal",
        "physical_mechanism": "Motor operates under balanced electromagnetic flux and mechanical equilibrium. Rotor spins centrally without eccentricity, bearings glide smoothly on unbroken grease hydrodynamic film, and field coils remain well below thermal breakdown limits.",
        "simple_analogy": "Like a car cruising smoothly on a flat highway with fresh engine oil and perfect wheel alignment.",
        "key_symptoms": "Stable RPM (1400-1600), Low vibration (< 1.5g), Cool windings (< 50°C), Current proportional to nominal load.",
        "engineering_standard": "ISO 10816-1 Zone A/B: Unrestricted long-term continuous industrial operation.",
        "action_required": "None. Continuous automatic condition monitoring active.",
    },
    1: {
        "title": "Bearing Fault (Raceway / Ball Wear)",
        "badge_color": "#FF8C00",
        "icon": "⚙️",
        "component": "Drive-End & Non-Drive-End Bearings",
        "physical_mechanism": "Fatigue spalling, pitting, or loss of lubrication on inner/outer bearing raceways or rolling balls. Each time a ball rolls across a micro-pit, it strikes the race, producing high-frequency shock pulses and localized friction.",
        "simple_analogy": "Like roller skates with grit or rust inside the wheel bearings, grinding and violently vibrating as they spin.",
        "key_symptoms": "Sharp spike in high-frequency radial vibration (X and Y axes), localized bearing temperature rise (Temp 2).",
        "engineering_standard": "ISO 10816-1 Zone C/D: Unacceptable vibration; high risk of sudden bearing seizure.",
        "action_required": "1. Re-grease bearing housing. 2. Verify bearing housing clearance. 3. Schedule deep-groove ball bearing replacement.",
    },
    2: {
        "title": "Stator Winding Fault (Insulation Degradation)",
        "badge_color": "#FF3547",
        "icon": "⚡",
        "component": "Stator Field Coils",
        "physical_mechanism": "Enamel insulation between copper wire turns in stationary field coils degrades from thermal stress or electrical aging. Micro-arcs form between adjacent turns, causing localized short-circuit heating and distorted magnetic flux.",
        "simple_analogy": "Like two frayed electrical wires touching each other inside a wall, sparking and getting dangerously hot.",
        "key_symptoms": "High winding temperature (Temp 1 > 60°C), abnormal current fluctuation, localized electrical hum.",
        "engineering_standard": "IEEE 522 / NEMA MG-1: Thermal insulation Class F limit exceeded.",
        "action_required": "1. Perform Megger insulation resistance test (500V). 2. Check coil resistance balance. 3. Rewind or replace defective stator coil.",
    },
    3: {
        "title": "Rotor Fault (Armature / Commutator Defect)",
        "badge_color": "#FF3547",
        "icon": "🔄",
        "component": "Armature Core & Commutator Bars",
        "physical_mechanism": "Broken armature copper conductors, loose rotor laminations, or spark erosion on commutator copper segments. Creates asymmetric rotating inertia, magnetic pull imbalance, and intermittent carbon brush contact.",
        "simple_analogy": "Like an unbalanced ceiling fan or car wheel that wobbles violently when spinning at high speed.",
        "key_symptoms": "Rotational speed fluctuation (hunting RPM), cyclical 1X rotational vibration, visible sparking at carbon brushes.",
        "engineering_standard": "ISO 1940-1 Balance Quality Grade G2.5 exceeded.",
        "action_required": "1. Inspect commutator copper surface for burning. 2. Replace worn carbon brushes and adjust spring tension. 3. Re-balance armature assembly.",
    },
    4: {
        "title": "Thermal Overload (Excessive Mechanical Load)",
        "badge_color": "#FF8C00",
        "icon": "🔥",
        "component": "Total Motor Electromagnetic Circuit",
        "physical_mechanism": "Mechanical resisting torque required by driven equipment exceeds motor's continuous rated horsepower. To deliver torque, motor draws excessive armature current, creating severe I²R resistive heating across all copper coils.",
        "simple_analogy": "Like pedaling a bicycle in highest gear up a steep mountain until legs overheat and stall.",
        "key_symptoms": "High current draw (> 18A), motor RPM pulls down below nominal, both Temp 1 and Temp 2 climb rapidly.",
        "engineering_standard": "IEC 60034-1: Exceeds maximum continuous duty rating (S1).",
        "action_required": "1. Immediately reduce driven mechanical load. 2. Check driven machinery for binding or jams. 3. Allow cooling fan to circulate until winding drops below 50°C.",
    },
    5: {
        "title": "Shaft Misalignment (Coupling Angular/Parallel Offset)",
        "badge_color": "#FFB800",
        "icon": "📐",
        "component": "Drive Shaft & Flexible Coupling",
        "physical_mechanism": "Motor centerline does not coincide precisely with driven pump/fan shaft centerline. As shafts rotate, the flexible coupling flexes cyclically, transmitting heavy axial (Z-axis) and radial reaction forces into motor bearings.",
        "simple_analogy": "Like connecting two pipes with a bent joint; the joint fights and flexes with every turn.",
        "key_symptoms": "Elevated axial vibration (Z-axis) at 1X and 2X rotational frequency, premature coupling elastomeric wear.",
        "engineering_standard": "ANSI/ASA S2.75: Angular and offset alignment tolerances exceeded.",
        "action_required": "1. Perform laser shaft alignment check on drive coupling. 2. Inspect soft-foot condition with feeler gauges. 3. Re-torque base mounting bolts.",
    },
}


def compute_health_and_diagnostics(sensors, predicted_fault, confidence, warning_status):
    x = sensors.get("X", 0.0)
    y = sensors.get("Y", 0.0)
    z = sensors.get("Z", 0.0)
    vib_mag = float(np.sqrt(x**2 + y**2 + z**2))
    t1 = sensors.get("Temp1", 35.0)
    t2 = sensors.get("Temp2", 30.0)
    avg_temp = (t1 + t2) / 2.0
    rpm = sensors.get("RPM", 1500.0)
    current = sensors.get("Current", 12.0)
    voltage = sensors.get("Voltage", 220.0)

    # Health Index (0-100)
    if predicted_fault == 0:
        health = 98.5
        if vib_mag > 1.6:
            health -= min(15.0, (vib_mag - 1.6) * 8.0)
        if avg_temp > 50.0:
            health -= min(15.0, (avg_temp - 50.0) * 1.2)
        if current > 17.0:
            health -= min(10.0, (current - 17.0) * 2.0)
    else:
        severity_penalty = {
            1: 45.0,  # Bearing
            2: 50.0,  # Stator
            3: 55.0,  # Rotor
            4: 60.0,  # Overload
            5: 48.0,  # Misalignment
        }.get(predicted_fault, 45.0)
        conf_factor = max(0.5, confidence)
        health = max(12.0, 100.0 - (severity_penalty * conf_factor))
        if warning_status == "EARLY WARNING":
            health = max(10.0, health - 12.0)

    health_score = round(float(np.clip(health, 5.0, 100.0)), 1)

    if health_score >= 88.0:
        condition_label = "OPTIMAL"
        condition_color = "#00E396"
    elif health_score >= 70.0:
        condition_label = "ACCEPTABLE"
        condition_color = "#00D4FF"
    elif health_score >= 50.0:
        condition_label = "CAUTION REQUIRED"
        condition_color = "#FFB800"
    else:
        condition_label = "CRITICAL HAZARD"
        condition_color = "#FF3547"

    # Component Physical Health Assessment
    de_vib = max(abs(x), abs(y))
    de_status = "CRITICAL" if predicted_fault == 1 or de_vib > 2.8 else ("WARNING" if de_vib > 1.6 or t2 > 55 else "HEALTHY")
    nde_status = "CRITICAL" if predicted_fault == 1 else ("WARNING" if t2 > 60 else "HEALTHY")
    stator_status = "CRITICAL" if predicted_fault == 2 or t1 > 75 else ("WARNING" if predicted_fault == 4 or t1 > 55 else "HEALTHY")
    rotor_status = "CRITICAL" if predicted_fault == 3 else ("WARNING" if rpm < 1100 or rpm > 1800 else "HEALTHY")
    coupling_status = "CRITICAL" if predicted_fault == 5 else ("WARNING" if abs(z) > 10.5 or abs(z) < 8.0 else "HEALTHY")

    components = {
        "de_bearing": {
            "name": "Drive-End (DE) Bearing",
            "status": de_status,
            "metric": f"{de_vib:.2f} g vibration",
            "desc": "Supports shaft at drive output end against radial and dynamic loads.",
        },
        "nde_bearing": {
            "name": "Non-Drive-End (NDE) Bearing",
            "status": nde_status,
            "metric": f"{t2:.1f} °C housing temp",
            "desc": "Guides commutator-end shaft journals and prevents axial drift.",
        },
        "stator": {
            "name": "Stator Field Coils",
            "status": stator_status,
            "metric": f"{t1:.1f} °C coil temp",
            "desc": "Produces stationary magnetic field connected in series with the armature.",
        },
        "rotor": {
            "name": "Armature & Commutator",
            "status": rotor_status,
            "metric": f"{rpm:.0f} RPM rotation",
            "desc": "Rotating assembly carrying copper conductors, commutator bars, and brushes.",
        },
        "coupling": {
            "name": "Drive Shaft & Coupling",
            "status": coupling_status,
            "metric": f"{abs(z):.2f} g axial load",
            "desc": "Transfers rotational torque directly to connected industrial load.",
        },
    }

    # Plain-English Physical Summaries for Students & Operators
    fault_explanations = {
        0: f"Motor M-01 is operating smoothly at {rpm:.0f} RPM drawing {current:.1f} A. Temperatures ({t1:.1f}°C / {t2:.1f}°C) and vibration ({vib_mag:.2f}g) are within safe baseline operating limits.",
        1: f"Bearing raceway friction or surface fatigue spalling detected. Abnormal radial vibration ({vib_mag:.2f}g) indicates metal-to-metal impact inside the ball bearings.",
        2: f"Stator field coil insulation degradation or micro-short detected. Coil temperature is {t1:.1f}°C with current draw at {current:.1f} A, creating magnetic field asymmetry.",
        3: f"Rotor dynamic eccentricity or commutator bar wear detected. Armature rotation is experiencing torque ripple and balance oscillations at {rpm:.0f} RPM.",
        4: f"Thermal Overload condition active! Resisting mechanical load exceeds torque capacity, pulling heavy current ({current:.1f} A) and heating windings to {t1:.1f}°C.",
        5: f"Mechanical shaft misalignment detected between motor and coupling. Angular/parallel offset creates cyclic vibration stress ({vib_mag:.2f}g) and axial thrust.",
    }

    maintenance_actions = {
        0: [
            "Routine telemetry check normal.",
            "Maintain standard inspection interval (ISO 10816 Class II).",
        ],
        1: [
            "Inspect bearing grease for contamination or dehydration.",
            "Verify bearing housing clearance with dial gauge.",
            "Schedule SKF/NSK deep-groove ball bearing replacement.",
        ],
        2: [
            "Perform insulation resistance (Megger 500V) check on field coils.",
            "Check for thermal hot-spots using infrared thermography.",
            "Inspect lead terminal connections for high resistance.",
        ],
        3: [
            "Inspect commutator copper segments for pitting or spark carbonization.",
            "Check carbon brush spring tension and brush wear length.",
            "Perform dynamic rotor balancing check.",
        ],
        4: [
            "Reduce driven mechanical load immediately.",
            "Check driven machinery for binding or jams.",
            "Allow cooling fan to circulate until winding drops below 50°C.",
        ],
        5: [
            "Perform laser shaft alignment check on drive coupling.",
            "Inspect soft-foot condition with feeler gauges.",
            "Torque base mounting bolts to specified manufacturer foot-lbs.",
        ],
    }

    return {
        "health_score": health_score,
        "condition_label": condition_label,
        "condition_color": condition_color,
        "vib_magnitude": round(vib_mag, 3),
        "plain_english_summary": fault_explanations.get(predicted_fault, "Diagnostic telemetry analyzing..."),
        "maintenance_actions": maintenance_actions.get(predicted_fault, []),
        "components": components,
    }


# ── Routes: static frontend ─────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


def simulation_payload(record):
    if record is None:
        return {
            "status": "simulation_stopped",
            "simulation": simulation_status(),
        }

    sensor_values = {
        column: record.get(column)
        for column in config.FEATURE_COLUMNS
    }
    severity = record["alert_severity"]
    warning_status = (
        "SENSOR THRESHOLD WARNING"
        if severity != "Normal"
        else "DATASET LABEL"
    )
    expected_fault = record["expected_fault"]
    predicted_fault = record["predicted_fault"] if "predicted_fault" in record else record["Predicted_Fault"]
    diagnostic_code = predicted_fault
    prediction_confidence = record["Confidence"]
    if prediction_confidence is None:
        prediction_confidence = None
    diagnostic = compute_health_and_diagnostics(
        sensor_values,
        diagnostic_code,
        prediction_confidence or 0.0,
        warning_status,
    )
    with simulation_lock:
        records = list(simulation_state["records"])

    return {
        "status": "ok",
        "mode": "simulation",
        "simulation_mode": True,
        "simulation_status": simulation_state["status"],
        "timestamp": record["timestamp"],
        "record_id": record.get("record_id"),
        "sensors": sensor_values,
        "fault_status": expected_fault,
        "fault_label": record["expected_label"],
        "expected_fault": expected_fault,
        "expected_label": record["expected_label"],
        "predicted_fault": predicted_fault,
        "predicted_label": record["ml_label"],
        "ml_prediction": predicted_fault,
        "ml_label": record["ml_label"],
        "confidence": prediction_confidence,
        "prediction_source": record["prediction_source"],
        "model_version": record["model_version"],
        "prediction_error": record["prediction_error"],
        "target_fault": expected_fault,
        "actual_label": expected_fault,
        "is_correct": int(
            predicted_fault == expected_fault
        ) if predicted_fault is not None else None,
        "prediction_match": record["prediction_match"],
        "diagnostic_fault": diagnostic_code,
        "diagnostic_label": record["ml_label"],
        "alert_severity": severity,
        "threshold_warnings": record["threshold_warnings"],
        "warning_status": warning_status,
        "active_alarms": sum(
            1 for item in records
            if item["alert_severity"] != "Normal"
        ),
        "total_records": len(records),
        "uptime_pct": round(
            sum(item["alert_severity"] == "Normal" for item in records)
            / len(records)
            * 100,
            1,
        ) if records else 0,
        "last_fault_time": record["timestamp"] if predicted_fault else "None",
        "health_score": diagnostic["health_score"],
        "condition_label": diagnostic["condition_label"],
        "condition_color": diagnostic["condition_color"],
        "vib_magnitude": diagnostic["vib_magnitude"],
        "plain_english_summary": diagnostic["plain_english_summary"],
        "maintenance_actions": diagnostic["maintenance_actions"],
        "components": diagnostic["components"],
        "phase": "SIMULATION MODE",
        "thresholds": SIMULATION_THRESHOLDS,
        "simulation": simulation_status(),
    }


@app.route("/api/simulation/status")
def api_simulation_status():
    return jsonify({
        "status": "ok",
        "simulation": simulation_status(),
    })


@app.route("/api/simulation/start", methods=["POST"])
def api_simulation_start():
    payload = request.get_json(silent=True) or {}
    selected_fault = payload.get("fault_status")
    try:
        if selected_fault not in (None, "", "all"):
            selected_fault = int(selected_fault)
        else:
            selected_fault = None
        start_simulation(selected_fault)
        return jsonify({"status": "ok", "simulation": simulation_status()})
    except (FileNotFoundError, ValueError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400


@app.route("/api/simulation/<action>", methods=["POST"])
def api_simulation_action(action):
    try:
        if action == "clear":
            with simulation_lock:
                simulation_state["records"].clear()
        else:
            set_simulation_status(action)
        return jsonify({"status": "ok", "simulation": simulation_status()})
    except ValueError as error:
        return jsonify({"status": "error", "message": str(error)}), 400


# ── API: latest reading ─────────────────────────────────────

@app.route("/api/latest")
def api_latest():
    mode = get_requested_mode()
    if mode is None:
        return jsonify({"status": "error", "message": "Unsupported dashboard mode"}), 400

    if mode == "simulation":
        with simulation_lock:
            latest = (
                simulation_state["records"][-1]
                if simulation_state["records"]
                else None
            )
        if latest is not None:
            return jsonify(simulation_payload(latest))
        return jsonify(simulation_payload(None))

    # ── Firebase Realtime Cloud Mode ──
    if mode == "firebase":
        fb_sensor = firebase_manager.fetch_latest_sensor_reading()
        if fb_sensor.get("status") == "unauthorized":
            return jsonify({
                "status": "firebase_unauthorized",
                "message": fb_sensor.get("message"),
                "database_url": config.FIREBASE_DATABASE_URL,
            })
        if fb_sensor.get("status") != "ok" or not fb_sensor.get("data"):
            return jsonify({
                "status": "firebase_waiting",
                "message": "Waiting for sensor readings from Firebase Realtime Database...",
                "database_url": config.FIREBASE_DATABASE_URL,
            })

        latest_data = fb_sensor["data"]
        sensors = {}
        for col in config.FEATURE_COLUMNS:
            sensors[col] = float(latest_data.get(col, 0.0))

        predicted_fault = int(latest_data.get("predicted_fault", latest_data.get("fault_simulation_target", 0)))
        confidence = float(latest_data.get("confidence", 0.96))
        warning_status = "NORMAL" if predicted_fault == 0 else ("EARLY WARNING" if confidence > 0.8 else "POSSIBLE ABNORMALITY")

        diag = compute_health_and_diagnostics(
            sensors, predicted_fault, confidence, warning_status
        )

        return jsonify({
            "status": "ok",
            "mode": "firebase",
            "timestamp": str(latest_data.get("timestamp", "Live Cloud Stream")),
            "predicted_fault": predicted_fault,
            "fault_label": FAULT_LABEL_MAP.get(predicted_fault, f"Fault {predicted_fault}"),
            "confidence": confidence,
            "warning_status": warning_status,
            "consecutive_abnormal": 1 if predicted_fault != 0 else 0,
            "sensors": sensors,
            "uptime_pct": 99.8,
            "active_alarms": 1 if predicted_fault != 0 else 0,
            "total_records": 1,
            "last_fault_time": str(latest_data.get("timestamp", "None")) if predicted_fault != 0 else "None",
            "health_score": diag["health_score"],
            "condition_label": diag["condition_label"],
            "condition_color": diag["condition_color"],
            "vib_magnitude": diag["vib_magnitude"],
            "plain_english_summary": diag["plain_english_summary"],
            "maintenance_actions": diag["maintenance_actions"],
            "components": diag["components"],
            "phase": "FIREBASE CLOUD REALTIME",
        })

    data = safe_read_csv(get_data_file(mode))
    if data is None:
        return jsonify({"status": "no_data"})

    latest = data.iloc[-1]
    predicted_fault = int(latest["Predicted_Fault"])
    confidence = float(latest["Confidence"])
    warning_status, consecutive = compute_warning(
        data, predicted_fault
    )

    total = len(data)
    normal_count = int((data["Predicted_Fault"] == 0).sum())
    uptime_pct = normal_count / total * 100 if total > 0 else 0

    recent = data.tail(20)
    active_alarms = int((recent["Predicted_Fault"] != 0).sum())

    faults = data[data["Predicted_Fault"] != 0]
    last_fault_time = (
        str(faults.iloc[-1]["timestamp"])
        if not faults.empty
        else "None"
    )

    sensors = {}
    for col in config.FEATURE_COLUMNS:
        sensors[col] = float(latest[col])

    diag = compute_health_and_diagnostics(
        sensors, predicted_fault, confidence, warning_status
    )

    result = {
        "status": "ok",
        "timestamp": str(latest["timestamp"]),
        "predicted_fault": predicted_fault,
        "fault_label": FAULT_LABEL_MAP.get(
            predicted_fault, f"Fault {predicted_fault}"
        ),
        "confidence": confidence,
        "warning_status": warning_status,
        "consecutive_abnormal": consecutive,
        "sensors": sensors,
        "uptime_pct": round(uptime_pct, 1),
        "active_alarms": active_alarms,
        "total_records": total,
        "last_fault_time": last_fault_time,
        "health_score": diag["health_score"],
        "condition_label": diag["condition_label"],
        "condition_color": diag["condition_color"],
        "vib_magnitude": diag["vib_magnitude"],
        "plain_english_summary": diag["plain_english_summary"],
        "maintenance_actions": diag["maintenance_actions"],
        "components": diag["components"],
    }

    # Simulation-specific fields
    if "Phase" in latest.index:
        result["phase"] = str(latest["Phase"])
        result["target_fault"] = int(latest["Target_Fault"])
        result["actual_label"] = int(latest["Actual_Label"])
        result["is_correct"] = int(latest["Is_Correct"])
    else:
        result["mode"] = mode

    return jsonify(result)


# ── API: fault guide / knowledge base ───────────────────────

@app.route("/api/fault_guide")
def api_fault_guide():
    return jsonify({
        "status": "ok",
        "classes": FAULT_GUIDE_DATA,
    })


# ── API: firebase cloud connection status ───────────────────

@app.route("/api/firebase_status")
def api_firebase_status():
    return jsonify(firebase_manager.check_firebase_connection())


# ── API: trend data ──────────────────────────────────────────

@app.route("/api/trends/<int:n>")
def api_trends(n):
    mode = get_requested_mode()
    if mode is None:
        return jsonify({"status": "error", "message": "Unsupported dashboard mode"}), 400
    if mode == "simulation":
        with simulation_lock:
            subset = list(simulation_state["records"])[-min(n, SIMULATION_RECORD_LIMIT):]
        subset = list(reversed(subset))
        if not subset:
            return jsonify({"status": "no_data"})
        return jsonify({
            "status": "ok",
            "timestamps": [row["timestamp"] for row in subset],
            **{
                column: [row[column] for row in subset]
                for column in ("Temp1", "Temp2", "RPM", "Current", "Voltage", "X", "Y", "Z")
            },
            "Confidence": [None for _ in subset],
            "Predicted_Fault": [row["fault_status"] for row in subset],
        })

    data = safe_read_csv(get_data_file(mode))
    if data is None:
        return jsonify({"status": "no_data"})

    subset = data.tail(n)

    result = {
        "status": "ok",
        "timestamps": subset["timestamp"].tolist(),
        "Temp1": subset["Temp1"].tolist(),
        "Temp2": subset["Temp2"].tolist(),
        "RPM": subset["RPM"].tolist(),
        "Current": subset["Current"].tolist(),
        "Voltage": subset["Voltage"].tolist(),
        "X": subset["X"].tolist(),
        "Y": subset["Y"].tolist(),
        "Z": subset["Z"].tolist(),
        "Confidence": subset["Confidence"].tolist(),
        "Predicted_Fault": subset["Predicted_Fault"].tolist(),
    }

    return jsonify(result)


# ── API: alarm log ───────────────────────────────────────────

@app.route("/api/alarms")
def api_alarms():
    mode = get_requested_mode()
    if mode is None:
        return jsonify({"status": "error", "message": "Unsupported dashboard mode"}), 400
    data = safe_read_csv(get_data_file(mode))
    if data is None:
        return jsonify({"status": "no_data", "alarms": []})

    alarms = data[data["Predicted_Fault"] != 0].copy()
    if alarms.empty:
        return jsonify({"status": "ok", "alarms": [], "summary": {
            "total": 0, "critical": 0, "major": 0, "minor": 0
        }})

    alarm_list = []
    for _, row in alarms.tail(100).iloc[::-1].iterrows():
        fault = int(row["Predicted_Fault"])
        conf = float(row["Confidence"])
        sev = classify_severity(fault, conf)
        alarm_list.append({
            "timestamp": str(row["timestamp"]),
            "severity": sev,
            "fault": fault,
            "fault_label": FAULT_LABEL_MAP.get(fault, f"Fault {fault}"),
            "confidence": round(conf * 100, 1),
            "phase": str(row.get("Phase", "")),
        })

    summary = {
        "total": len(alarms),
        "critical": sum(1 for a in alarm_list if a["severity"] == "Critical"),
        "major": sum(1 for a in alarm_list if a["severity"] == "Major"),
        "minor": sum(1 for a in alarm_list if a["severity"] == "Minor"),
    }

    return jsonify({
        "status": "ok",
        "alarms": alarm_list,
        "summary": summary,
    })


# ── API: system health ───────────────────────────────────────

@app.route("/api/health")
def api_health():
    mode = get_requested_mode()
    if mode is None:
        return jsonify({"status": "error", "message": "Unsupported dashboard mode"}), 400
    data = safe_read_csv(get_data_file(mode))
    if data is None:
        return jsonify({"status": "no_data"})

    total = len(data)
    missing = int(
        data[config.FEATURE_COLUMNS].isna().sum().sum()
    )
    duplicates = int(
        data.duplicated(subset=config.FEATURE_COLUMNS).sum()
    )
    normal_pct = (
        (data["Predicted_Fault"] == 0).sum() / total * 100
        if total > 0
        else 0
    )

    counts = data["Predicted_Fault"].value_counts().sort_index()
    distribution = {
        FAULT_LABEL_MAP.get(int(k), f"Fault {k}"): int(v)
        for k, v in counts.items()
    }

    return jsonify({
        "status": "ok",
        "total_records": total,
        "missing_values": missing,
        "duplicate_rows": duplicates,
        "normal_pct": round(normal_pct, 1),
        "distribution": distribution,
    })


# ── API: model governance ────────────────────────────────────

@app.route("/api/governance")
def api_governance():
    payload = model_governance.status_payload()
    review_data = safe_read_csv(REVIEW_LOG_FILE)
    latest = review_data.iloc[-1] if review_data is not None else None

    def safe_get(key, default=""):
        if latest is None:
            return default
        val = latest.get(key, default)
        if pd.isna(val):
            return str(default)
        return str(val)

    def safe_float(key, default=0):
        if latest is None:
            return default
        val = latest.get(key, default)
        try:
            return round(float(val), 2)
        except (ValueError, TypeError):
            return default

    payload.update({
        "reviewer_decision": safe_get("Reviewer_Decision", "PENDING"),
        "system_decision": safe_get("System_Decision", "UNKNOWN"),
        "reviewer_name": safe_get("Reviewer_Name", "Pending"),
        "production_accuracy": safe_float("Production_Accuracy"),
        "candidate_accuracy": safe_float("Candidate_Accuracy"),
        "accuracy_change": safe_float("Accuracy_Change_Percentage_Points"),
        "candidate_fault4_recall": safe_float("Candidate_Fault_4_Recall"),
        "fault4_recall_change": safe_float("Fault_4_Recall_Change_Percentage_Points"),
        "review_notes": safe_get("Review_Notes", ""),
    })
    return jsonify(payload)


def authorized_operator():
    token = request.headers.get(model_governance.OPERATOR_HEADER)
    if token != config.MODEL_OPERATOR_TOKEN:
        raise PermissionError("Authorized model operator token is required")
    return token


@app.route("/api/governance/retrain", methods=["POST"])
def api_governance_retrain():
    try:
        actor = authorized_operator()
        payload = request.get_json(silent=True) or {}
        trigger = payload.get("trigger", "manual")
        return jsonify(model_governance.request_retraining(actor, trigger))
    except (PermissionError, RuntimeError, ValueError) as error:
        return jsonify({"status": "error", "message": str(error)}), 403


@app.route("/api/governance/promote", methods=["POST"])
def api_governance_promote():
    try:
        actor = authorized_operator()
        payload = request.get_json(silent=True) or {}
        result = model_governance.promote_candidate(
            actor,
            str(payload.get("notes", "")).strip(),
        )
        if not production_predictor.reload():
            return jsonify({
                "status": "error",
                "message": "Promotion completed but production reload failed",
            }), 500
        return jsonify(result)
    except (PermissionError, RuntimeError, ValueError, FileNotFoundError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400


@app.route("/api/governance/rollback", methods=["POST"])
def api_governance_rollback():
    try:
        actor = authorized_operator()
        payload = request.get_json(silent=True) or {}
        result = model_governance.rollback(
            actor,
            str(payload.get("version", "")).strip(),
        )
        if not production_predictor.reload():
            return jsonify({
                "status": "error",
                "message": "Rollback completed but production reload failed",
            }), 500
        return jsonify(result)
    except (PermissionError, RuntimeError, ValueError, FileNotFoundError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400


@app.route("/api/governance/test", methods=["POST"])
def api_governance_test():
    try:
        payload = request.get_json(silent=True) or {}
        return jsonify({
            "status": "ok",
            "result": model_governance.test_single_record(
                payload.get("model", "production"),
                payload.get("sensors", {}),
            ),
        })
    except (ValueError, TypeError, FileNotFoundError, RuntimeError) as error:
        return jsonify({"status": "error", "message": str(error)}), 400


# ── API: recent records ─────────────────────────────────────

@app.route("/api/records/<int:n>")
def api_records(n):
    mode = get_requested_mode()
    if mode is None:
        return jsonify({"status": "error", "message": "Unsupported dashboard mode"}), 400
    if mode == "simulation":
        with simulation_lock:
            records = list(simulation_state["records"])[-min(n, SIMULATION_RECORD_LIMIT):]
        return jsonify({
            "status": "ok",
            "records": list(reversed(records)),
            "simulation": simulation_status(),
        })

    data = safe_read_csv(get_data_file(mode))
    if data is None:
        return jsonify({"status": "no_data", "records": []})

    subset = data.tail(n).iloc[::-1]

    cols = ["timestamp", "Predicted_Fault", "Confidence"]
    optional = ["Phase", "Target_Fault", "Actual_Label", "Is_Correct"]
    for c in optional:
        if c in subset.columns:
            cols.append(c)

    records = []
    for _, row in subset.iterrows():
        record = {}
        for c in cols:
            val = row[c]
            if isinstance(val, (np.integer,)):
                record[c] = int(val)
            elif isinstance(val, (np.floating, float)):
                record[c] = round(float(val), 4)
            else:
                record[c] = str(val)
        record["fault_label"] = FAULT_LABEL_MAP.get(
            int(row["Predicted_Fault"]),
            f"Fault {int(row['Predicted_Fault'])}"
        )
        records.append(record)

    return jsonify({
        "status": "ok",
        "records": records,
    })


# ── Run ──────────────────────────────────────────────────────

if __name__ == "__main__":
    print()
    print("=" * 56)
    print("  MOTOR INTELLIGENCE CENTER")
    print("  Dashboard: http://localhost:5000")
    print("=" * 56)
    print()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
