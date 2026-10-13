import sys
from pathlib import Path

# Add app directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

import firebase_admin
from firebase_admin import credentials, db
import config

print("=" * 60)
print("  FIREBASE ADMIN SDK CONNECTION TEST")
print(f"  Database URL : {config.FIREBASE_DATABASE_URL}")
print(f"  Credentials  : {config.FIREBASE_CREDENTIALS_FILE}")
print("=" * 60)

if not config.FIREBASE_CREDENTIALS_FILE.exists():
    print(f"[!] Error: Credentials file not found at {config.FIREBASE_CREDENTIALS_FILE}")
    sys.exit(1)

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(config.FIREBASE_CREDENTIALS_FILE))
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": config.FIREBASE_DATABASE_URL},
        )

    ref = db.reference("test_connection")
    test_payload = {
        "status": "connected",
        "source": "python",
        "timestamp": "verified",
    }
    ref.set(test_payload)

    print("\n[+] Firebase connection successful!")
    print(f"Database test_connection record: {ref.get()}")
    print("\nYour backend is now fully authenticated with Firebase Realtime Database.")

except Exception as err:
    print(f"\n[-] Firebase connection failed: {err}")
    sys.exit(1)
