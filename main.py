import os
from typing import Final, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from classify_shogi_piece import mock_classify_pieces
from firebase import db_operate
from generate_kifu import mock_generate_kifu
from my_types import HelloWorldType, IntegerArrayType
from predict_board import predict_board

app = FastAPI()

DATASET_DIR: Final[str] = "./data/dataset"


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

    # 各マス目画像を取得, shape: (マス目の数, マス目の縦幅, マス目の横幅, 色) = (81, 98, 91, 3)
    square_images: IntegerArrayType = predict_board(image=image)
    print(square_images.shape)
    # コマ検出
    pieces: IntegerArrayType = mock_classify_pieces(
        image=square_images, model_path="./models/shogi_model.pth"
    )
    # 棋譜生成
    kifu: str = mock_generate_kifu(pieces, is_sente)
    # DB登録

    return {"kifu": kifu}


@app.post("/add_dataset")
async def add_dataset(image_files: List[UploadFile] = File(...)):

    for image_index, image_file in enumerate(image_files):
        # 画像をnumpyとして読み込み
        contents: bytes = await image_file.read()
        array: IntegerArrayType = np.fromstring(contents, np.uint8)
        image: IntegerArrayType = cv2.imdecode(array, cv2.IMREAD_COLOR)

        # 各マス目画像を取得, shape: (マス目の数, マス目の縦幅, マス目の横幅, 色) = (81, 98, 91, 3)
        square_images: IntegerArrayType = predict_board(image=image)

        # マス目画像保存
        for square_index, square_image in enumerate(square_images):
            path: str = os.path.join(
                DATASET_DIR,
                "uncategorized",
                f"{image_index}-{square_index}.jpg",
            )
            print(path)
            cv2.imwrite(path, square_image)


def main() -> None:
    print("===== main() =====")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, workers=2)


if __name__ == "__main__":
    main()
