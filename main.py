from typing import Final, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from classify_shogi_piece import mock_classify_pieces
from firebase import db_operate
from generate_kifu import generate_kifu
from my_types import HelloWorldType, IntegerArrayType
from predict_board import predict_board

DEFAULT_BOARD: Final[IntegerArrayType] = np.array(
    [
        [-5, -4, -3, -2, -1, -2, -3, -4, -5],
        [0, -6, 0, 0, 0, 0, 0, -7, 0],
        [-8, -8, -8, -8, -8, -8, -8, -8, -8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [8, 8, 8, 8, 8, 8, 8, 8, 8],
        [0, 6, 0, 0, 0, 0, 0, 7, 0],
        [5, 4, 3, 2, 1, 2, 3, 4, 5],
    ]
)


app = FastAPI()


class StartRequestModel(BaseModel):
    sente: str
    gote: str


class StartResponseModel(BaseModel):
    id: str


class MoveResponseModel(BaseModel):
    kifu_list: List[str]


@app.get("/")
def root() -> HelloWorldType:
    return {"Hello": "Shogi"}


@app.post("/start", response_model=StartResponseModel)
def create_game(properties: StartRequestModel):
    sente, gote = properties.sente, properties.gote
    id: str = db_operate.create_game(sente=sente, gote=gote)
    np.savetxt(f"./data/csv/{id}.csv", DEFAULT_BOARD, delimiter=",", fmt="%d")
    return {"id": id}


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

    # 各マス目画像を取得, shape: (マス目の数, マス目の縦幅, マス目の横幅, 色) = (81, 98, 91, 3)
    square_images: IntegerArrayType = predict_board(image=image)
    # コマ検出
    pieces: IntegerArrayType = mock_classify_pieces(
        image=square_images, model_path="./models/shogi_model.pth"
    )
    prev_kifu: str = db_operate.get_kifu_list(id=id)[-1]
    # 棋譜生成
    kifu: str = generate_kifu(
        pieces=pieces,
        is_sente=is_sente,
        csv_path=f"./data/csv/{id}.csv",
        prev_kifu=prev_kifu,
    )
    # DB登録
    kifu_list: List[str] = db_operate.move_piece(id=id, kifu=kifu)

    return {"kifu_list": kifu_list}


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
