import glob
import os
from typing import Final, List

import cv2

from my_types import IntegerArrayType
from predict_board import predict_board

DATASET_DIR: Final[str] = "./data/dataset"
RAW_DATA_PATH: Final[str] = "./data/raw_data/*.JPG"

if __name__ == "__main__":
    image_files: List[str] = sorted(glob.glob(RAW_DATA_PATH))
    for image_index, image_file in enumerate(image_files):
        # 画像をnumpyとして読み込み
        image: IntegerArrayType = cv2.imread(image_file)
        print(image_file)

        # 各マス目画像を取得, shape: (マス目の数, マス目の縦幅, マス目の横幅, 色) = (81, 98, 91, 3)
        square_images: IntegerArrayType = predict_board(image=image)

        # マス目画像保存
        for square_index, square_image in enumerate(square_images):
            path: str = os.path.join(
                DATASET_DIR,
                "uncategorized",
                f"{image_index}-{square_index}.jpg",
            )
            cv2.imwrite(path, square_image)
