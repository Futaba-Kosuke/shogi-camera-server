import datetime
import os

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials, db, firestore

load_dotenv()

cred = credentials.Certificate("firebase/cred.json")

firebase_admin.initialize_app(
    cred,
    {
        "databaseURL": os.getenv("FIREBASE_URL"),
        "databaseAuthVariableOverride": {"uid": "my-service-worker"},
    },
)

firestore = firestore.client()


def create_game(sente: str, gote: str) -> str:
    # databaseに初期データを追加する
    users_ref = db.reference()

    realtime_db_ref = users_ref.push(["平手"])

    doc_ref = firestore.collection("games").document(realtime_db_ref.key)
    doc_ref.set(
        {
            "sente": sente,
            "gote": gote,
            "startTime": datetime.datetime.now(),
            "endTime": None,
            "id": realtime_db_ref.key,
            "status": "対局中",
            "handicap": "平手",
        }
    )
    return realtime_db_ref.key
