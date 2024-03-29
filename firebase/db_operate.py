import datetime
import os
from typing import List

import firebase_admin
from dotenv import load_dotenv
from firebase_admin import credentials
from firebase_admin import db as realtime_db
from firebase_admin import firestore

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
    realtime_db_ref = realtime_db.reference()
    game_realtime_db_ref = realtime_db_ref.push(["平手"])

    game_firestore_ref = firestore.collection("games").document(
        game_realtime_db_ref.key
    )
    game_firestore_ref.set(
        {
            "sente": sente,
            "gote": gote,
            "startTime": datetime.datetime.now(),
            "endTime": None,
            "id": game_realtime_db_ref.key,
            "status": "対局中",
            "handicap": "平手",
        }
    )
    return game_realtime_db_ref.key


def get_kifu_list(id: str) -> List[str]:
    game_realtime_db_ref = realtime_db.reference(id)
    return game_realtime_db_ref.get()


def move_piece(id: str, kifu: str) -> List[str]:
    game_realtime_db_ref = realtime_db.reference(id)
    kifu_list: List[str] = game_realtime_db_ref.get()
    kifu_list.append(kifu)
    game_realtime_db_ref.set(kifu_list)
    return kifu_list


def end_game(id: str) -> None:
    # realtime_db
    game_realtime_db_ref = realtime_db.reference(id)
    kifu_list: List[str] = game_realtime_db_ref.get()
    kifu_list.append("終局")
    game_realtime_db_ref.set(kifu_list)
    # firestore
    status: str = "先手勝ち" if len(kifu_list) % 2 == 1 else "後手勝ち"
    game_firestore_ref = firestore.collection("games").document(id)
    game_firestore_ref.update(
        {"endTime": datetime.datetime.now(), "status": status}
    )
    return
