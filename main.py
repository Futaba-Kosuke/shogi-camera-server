import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from firebase import db_operate
from my_types import HelloWorldType

app = FastAPI()


class Item(BaseModel):
    sente: str
    gote: str


@app.get("/")
def root() -> HelloWorldType:
    return {"Hello": "Shogi"}


@app.post("/start")
def create_item(item: Item) -> int:
    sente = item.sente
    gote = item.gote
    db_operate.create_item(sente, gote)
    return 200


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
