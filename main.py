import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from classify_shogi_piece import mock_classify_pieces
from firebase import db_operate
from generate_kifu import mock_generate_kifu
from my_types import HelloWorldType, IntegerArrayType
from predict_board import mock_predict_board

app = FastAPI()


class StartRequestModel(BaseModel):
    sente: str
    gote: str


class StartResponseModel(BaseModel):
    id: str


class MoveResponseModel(BaseModel):
    kifu: str


@app.get("/")
def root() -> HelloWorldType:
    return {"Hello": "Shogi"}


@app.post("/start", response_model=StartResponseModel)
def create_game(properties: StartRequestModel):
    sente, gote = properties.sente, properties.gote
    realtime_key: str = db_operate.create_game(sente=sente, gote=gote)
    return {"id": realtime_key}


@app.post("/move", response_model=MoveResponseModel)
async def move_piece(
    id: str = Form(...),
    is_sente: bool = Form(...),
    image_file: UploadFile = File(...),
):

    # 画像をnumpyとして読み込み
    contents: bytes = await image_file.read()
    array: IntegerArrayType = np.fromstring(contents, np.uint8)
    image: IntegerArrayType = cv2.imdecode(array, cv2.IMREAD_COLOR)

    # 盤面検出
    board: IntegerArrayType = mock_predict_board(image=image)
    # コマ検出
    pieces: IntegerArrayType = mock_classify_pieces(
        image=board, model_path="./models/shogi_model.pth"
    )
    # 棋譜生成
    kifu: str = mock_generate_kifu(pieces, is_sente)
    # DB登録

    return {"kifu": kifu}


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
