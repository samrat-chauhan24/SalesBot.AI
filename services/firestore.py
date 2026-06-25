
import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime, timedelta
import pytz
from google.cloud.firestore_v1.base_query import FieldFilter

# ---------------- INIT ----------------
if not firebase_admin._apps:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

# 🔥 IST timezone (IMPORTANT)
IST = pytz.timezone("Asia/Kolkata")

# ---------------- BASE QUERY ----------------
def _get_base_query(salesperson_id: str):
    return db.collection("visits").where(
        filter=FieldFilter("salesPersonId", "==", salesperson_id.strip())
    )

# ---------------- HELPER ----------------
def to_millis(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

# ---------------- TODAY VISITS ----------------
def get_today_visits(salesperson_id: str):
    try:
        now = datetime.now(IST)

        start_of_day = IST.localize(datetime(now.year, now.month, now.day))
        end_of_day = start_of_day + timedelta(days=1)

        start_ts = to_millis(start_of_day)
        end_ts = to_millis(end_of_day)

        docs = (
            _get_base_query(salesperson_id)
            .where(filter=FieldFilter("visitDate", ">=", start_ts))
            .where(filter=FieldFilter("visitDate", "<", end_ts))
            .stream()
        )

        results = []
        for doc in docs:
            d = doc.to_dict()

            results.append({
                "customerName": d.get("customerName"),
                "summary": (d.get("meetingSummary") or "")[:200],
                "emotion": d.get("customerEmotion"),
                "date": d.get("visitDate"),
            })

        return results

    except Exception as e:
        print("Error in get_today_visits:", e)
        return []

# ---------------- PENDING FOLLOWUPS ----------------
def get_pending_followups(salesperson_id: str):
    try:
        docs = (
            _get_base_query(salesperson_id)
            .where(filter=FieldFilter("outcomeStatus", "==", "Completed"))
            .stream()
        )

        followups = []

        for doc in docs:
            d = doc.to_dict()

            next_step = (d.get("nextStep") or "").strip().lower()

            # 🔥 remove junk
            if not next_step or next_step in ["none", "n/a", "not mentioned"]:
                continue

            followups.append({
                "customerName": d.get("customerName"),
                "contact": d.get("contactPerson"),
                "nextStep": d.get("nextStep"),
                "action": d.get("actionItems", ""),
            })

        return followups[:10]

    except Exception as e:
        print("Error in get_pending_followups:", e)
        return []

# ---------------- WEEKLY PERFORMANCE ----------------
def get_weekly_performance_data(salesperson_id: str):
    try:
        now = datetime.now(IST)
        start = now - timedelta(days=7)

        start_ts = to_millis(start)
        end_ts = to_millis(now)

        docs = (
            _get_base_query(salesperson_id)
            .where(filter=FieldFilter("visitDate", ">=", start_ts))
            .where(filter=FieldFilter("visitDate", "<", end_ts))
            .stream()
        )

        results = []
        for doc in docs:
            d = doc.to_dict()

            results.append({
                "customerName": d.get("customerName"),
                "probability": d.get("dealProbability"),
                "status": d.get("outcomeStatus"),
                "painPoints": d.get("painPoints", ""),
            })

        return results

    except Exception as e:
        print("Error in get_weekly_performance_data:", e)
        return []

