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

fireDB = firestore.client()


def create_item(sente, gote) -> None:
    # databaseに初期データを追加する
    users_ref = db.reference()

    realID = users_ref.push(["平手"])

    doc_ref = fireDB.collection("games")
    doc_ref.add(
        {
            "sente": sente,
            "gote": gote,
            "startTime": datetime.datetime.now(),
            "endTime": None,
            "id": realID.key,
            "status": "対局中",
            "handicap": "平手",
        }
    )
