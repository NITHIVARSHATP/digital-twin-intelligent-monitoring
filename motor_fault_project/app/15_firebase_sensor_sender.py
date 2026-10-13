"""
Firebase Realtime Sensor Telemetry Publisher.
Simulates an IoT microcontroller (e.g. ESP32, Arduino) streaming physical
motor vibration, current, voltage, speed, and temperature directly to the
Firebase Cloud Realtime Database.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import firebase_manager

SAMPLE_INTERVAL_SECONDS = 1.0

# Nominal Baseline Profile
NOMINAL_VALUES = {
    "Voltage": 224.5,
    "Current": 12.4,
    "RPM": 1520.0,
    "Temp1": 36.5,
    "Temp2": 32.0,
    "X": 0.22,
    "Y": -1.05,
    "Z": 9.10,
}

NOISE_STDS = {
    "Voltage": 1.5,
    "Current": 0.35,
    "RPM": 18.0,
    "Temp1": 0.4,
    "Temp2": 0.3,
    "X": 0.04,
    "Y": 0.06,
    "Z": 0.05,
}

# Fault Injection Profiles
FAULT_PROFILES = {
    0: ("Normal Steady State", {}),
    1: ("Bearing Fault (High Radial Vibration + Temp2 Rise)", {"X": 2.4, "Y": 2.6, "Temp2": 58.0}),
    2: ("Stator Winding Fault (Coil Overheat + Current Surge)", {"Temp1": 74.0, "Current": 19.5}),
    3: ("Rotor Eccentricity (RPM Fluctuation + Commutator Sparks)", {"RPM": 1380.0, "X": 1.8, "Y": 1.9}),
    4: ("Thermal Overload (Massive Current + Thermal Buildup)", {"Current": 24.0, "Temp1": 82.0, "Temp2": 68.0, "RPM": 1250.0}),
    5: ("Shaft Misalignment (Coupling Axial Stress)", {"Z": 12.2, "X": 1.9, "Y": 1.8}),
}


def generate_reading(fault_code: int = 0) -> dict:
    fault_name, overrides = FAULT_PROFILES.get(fault_code, ("Normal", {}))
    reading = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "fault_simulation_target": fault_code,
        "fault_simulation_label": fault_name,
    }

    rng = np.random.default_rng()

    for feat in config.FEATURE_COLUMNS:
        base = overrides.get(feat, NOMINAL_VALUES[feat])
        std = NOISE_STDS[feat]
        reading[feat] = round(float(rng.normal(loc=base, scale=std)), 3)

    return reading


def main():
    print("=" * 64)
    print("  FIREBASE CLOUD MOTOR TELEMETRY SENDER")
    print(f"  Target Database: {config.FIREBASE_DATABASE_URL}")
    print("=" * 64)
    print()

    # Initial connection check
    print("Checking Firebase connection...")
    conn = firebase_manager.check_firebase_connection()
    if conn["status"] == "unauthorized":
        print()
        print("⚠️  ATTENTION: Firebase Realtime Database returned 401 Unauthorized.")
        print("   To enable public test access:")
        print("   1. Open: https://console.firebase.google.com")
        print("   2. Go to: Realtime Database -> Rules tab")
        print("   3. Set rules to:")
        print("      {\n        \".read\": true,\n        \".write\": true\n      }")
        print("   4. Click 'Publish'.")
        print()

    # Cycle pattern: Normal (10s), Bearing (8s), Normal (6s), Stator (8s), Overload (8s)...
    cycle_schedule = [
        (0, 10),
        (1, 8),
        (0, 6),
        (2, 8),
        (0, 6),
        (3, 8),
        (0, 6),
        (4, 8),
        (0, 6),
        (5, 8),
    ]

    current_idx = 0
    phase_timer = 0

    print("Beginning continuous streaming to /motor_telemetry/latest ...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            fault_code, duration = cycle_schedule[current_idx]
            reading = generate_reading(fault_code)

            res = firebase_manager.push_sensor_reading(reading)

            if res.get("status") == "ok":
                print(
                    f"[{reading['timestamp']}] Pushed to Firebase -> "
                    f"State: {reading['fault_simulation_label']} | "
                    f"RPM: {reading['RPM']:.0f} | Current: {reading['Current']:.1f}A | "
                    f"T1: {reading['Temp1']:.1f}°C | Vib Z: {reading['Z']:.2f}g"
                )
            elif res.get("status") == "unauthorized":
                print(f"[{reading['timestamp']}] ⚠️ 401 Unauthorized: {res.get('message')}")
            else:
                print(f"[{reading['timestamp']}] Error: {res.get('message')}")

            phase_timer += SAMPLE_INTERVAL_SECONDS
            if phase_timer >= duration:
                phase_timer = 0
                current_idx = (current_idx + 1) % len(cycle_schedule)

            time.sleep(SAMPLE_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStreaming halted by operator.")


if __name__ == "__main__":
    main()
