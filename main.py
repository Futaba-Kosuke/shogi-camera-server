import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from firebase import db_operate
from my_types import HelloWorldType

app = FastAPI()


class ReturnResponseKey(BaseModel):
    id: str


class StartRequestType(BaseModel):
    sente: str
    gote: str


@app.get("/")
def root() -> HelloWorldType:
    return {"Hello": "Shogi"}


@app.post("/start", response_model=ReturnResponseKey)
def create_battle(item: StartRequestType):
    sente = item.sente
    gote = item.gote
    realtime_key = db_operate.create_item(sente, gote)
    return {"id": realtime_key}


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
