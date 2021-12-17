from typing import Any, Final, List

import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]

SHOGI_PIECES: List[str] = [
    "玉",
    "金",
    "銀",
    "桂",
    "香",
    "角",
    "飛",
    "歩",
    "成銀",
    "成桂",
    "成香",
    "馬",
    "竜",
    "と金",
]


def mock_classify_pieces(
    image: IntegerArrayType, model_path: str
) -> IntegerArrayType:
    return np.array(
        [
            [-5, -4, -3, -2, -1, -2, -3, -4, -5],
            [0, -6, 0, 0, 0, 0, 0, -7, 0],
            [-8, -8, -8, -8, -8, -8, -8, -8, -8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 8, 0, 0],
            [8, 8, 8, 8, 8, 8, 0, 8, 8],
            [0, 6, 0, 0, 0, 0, 0, 7, 0],
            [5, 4, 3, 2, 1, 2, 3, 4, 5],
        ]
    )
