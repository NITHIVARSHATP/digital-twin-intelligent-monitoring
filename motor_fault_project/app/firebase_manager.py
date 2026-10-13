"""
Firebase Cloud Integration Manager for Motor Digital Twin.
Handles real-time telemetry streaming, model prediction synchronization,
and connection health checks with Firebase Realtime Database.
"""

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import config

logger = logging.getLogger("FirebaseManager")

# Track whether firebase_admin is initialized
_FIREBASE_ADMIN_INITIALIZED = False
_FIREBASE_DB_REF = None


def init_firebase_admin() -> bool:
    """
    Attempts to initialize firebase_admin if a service account credentials
    file exists at config.FIREBASE_CREDENTIALS_FILE.
    """
    global _FIREBASE_ADMIN_INITIALIZED, _FIREBASE_DB_REF
    if _FIREBASE_ADMIN_INITIALIZED:
        return True

    cred_file = config.FIREBASE_CREDENTIALS_FILE
    if not cred_file.exists():
        return False

    try:
        import firebase_admin
        from firebase_admin import credentials, db

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_file))
            firebase_admin.initialize_app(
                cred,
                {"databaseURL": config.FIREBASE_DATABASE_URL},
            )
        _FIREBASE_ADMIN_INITIALIZED = True
        _FIREBASE_DB_REF = db.reference()
        logger.info("Initialized firebase-admin with service account.")
        return True
    except Exception as exc:
        logger.warning(f"Could not initialize firebase_admin: {exc}")
        return False


def rest_request(
    path: str,
    method: str = "GET",
    data: Optional[Dict[str, Any]] = None,
    timeout: float = 4.0,
) -> Dict[str, Any]:
    """
    Direct REST API client for Firebase Realtime Database.
    Works seamlessly without service accounts when database rules permit read/write.
    """
    base_url = config.FIREBASE_DATABASE_URL.rstrip("/")
    url = f"{base_url}/{path.lstrip('/')}.json"

    headers = {"Content-Type": "application/json"}
    body = json.dumps(data).encode("utf-8") if data is not None else None

    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            resp_body = response.read().decode("utf-8")
            if not resp_body or resp_body == "null":
                return {"status": "ok", "data": None}
            return {"status": "ok", "data": json.loads(resp_body)}
    except urllib.error.HTTPError as http_err:
        if http_err.code == 401:
            return {
                "status": "unauthorized",
                "code": 401,
                "message": (
                    "Firebase Realtime Database access is Unauthorized (401). "
                    "In Firebase Console -> Realtime Database -> Rules, set '.read': true and '.write': true "
                    "for test mode, or provide firebase_credentials.json in the project root."
                ),
            }
        return {"status": "error", "code": http_err.code, "message": str(http_err)}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def push_sensor_reading(reading: Dict[str, Any]) -> Dict[str, Any]:
    """
    Publishes a new raw sensor telemetry reading to Firebase:
    - Writes to /motor_telemetry/latest
    - Optionally appends to /motor_telemetry/history (capped)
    """
    if "timestamp" not in reading:
        reading["timestamp"] = datetime.now().isoformat()

    if init_firebase_admin():
        try:
            from firebase_admin import db

            ref = db.reference("motor_telemetry/latest")
            ref.set(reading)
            return {"status": "ok", "method": "firebase_admin"}
        except Exception as exc:
            logger.warning(f"firebase_admin push failed, falling back to REST: {exc}")

    # Fallback to REST
    res = rest_request("motor_telemetry/latest", method="PUT", data=reading)
    return res


def fetch_latest_sensor_reading() -> Dict[str, Any]:
    """
    Retrieves the latest raw sensor readings from Firebase.
    """
    if init_firebase_admin():
        try:
            from firebase_admin import db

            ref = db.reference("motor_telemetry/latest")
            val = ref.get()
            return {"status": "ok", "data": val, "method": "firebase_admin"}
        except Exception as exc:
            logger.warning(f"firebase_admin fetch failed, falling back to REST: {exc}")

    return rest_request("motor_telemetry/latest", method="GET")


def push_prediction(prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Publishes the latest ML fault diagnosis and health score to Firebase.
    """
    if "timestamp" not in prediction:
        prediction["timestamp"] = datetime.now().isoformat()

    if init_firebase_admin():
        try:
            from firebase_admin import db

            ref = db.reference("motor_predictions/latest")
            ref.set(prediction)
            return {"status": "ok", "method": "firebase_admin"}
        except Exception as exc:
            logger.warning(f"firebase_admin prediction push failed: {exc}")

    return rest_request("motor_predictions/latest", method="PUT", data=prediction)


def fetch_latest_prediction() -> Dict[str, Any]:
    """
    Retrieves the latest prediction and health metrics from Firebase.
    """
    if init_firebase_admin():
        try:
            from firebase_admin import db

            ref = db.reference("motor_predictions/latest")
            val = ref.get()
            return {"status": "ok", "data": val, "method": "firebase_admin"}
        except Exception as exc:
            logger.warning(f"firebase_admin fetch prediction failed: {exc}")

    return rest_request("motor_predictions/latest", method="GET")


def check_firebase_connection() -> Dict[str, Any]:
    """
    Tests the connection status of the Firebase Realtime Database.
    Uses firebase_admin if initialized, otherwise tests via REST.
    """
    if init_firebase_admin():
        try:
            from firebase_admin import db

            ref = db.reference("test_connection")
            ref.get()
            return {
                "status": "connected",
                "url": config.FIREBASE_DATABASE_URL,
                "method": "firebase_admin",
                "has_credentials_file": True,
            }
        except Exception as exc:
            logger.warning(f"firebase_admin check failed, falling back to REST: {exc}")

    res = rest_request("motor_telemetry", method="GET", timeout=4.0)
    if res.get("status") == "ok":
        return {
            "status": "connected",
            "url": config.FIREBASE_DATABASE_URL,
            "has_credentials_file": config.FIREBASE_CREDENTIALS_FILE.exists(),
        }
    if res.get("status") == "unauthorized":
        return {
            "status": "unauthorized",
            "url": config.FIREBASE_DATABASE_URL,
            "message": res.get("message"),
            "has_credentials_file": config.FIREBASE_CREDENTIALS_FILE.exists(),
        }
    return {
        "status": "disconnected",
        "url": config.FIREBASE_DATABASE_URL,
        "message": res.get("message"),
        "has_credentials_file": config.FIREBASE_CREDENTIALS_FILE.exists(),
    }
