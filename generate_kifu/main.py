from typing import Any, Final, List

import numpy as np
import numpy.typing as npt

# from numpy.lib.function_base import diff

# from classify_shogi_piece.main import SHOGI_PIECES

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

KAN_SUJI: List[str] = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]


def mock_generate_kifu(pieces: IntegerArrayType, is_sente: bool) -> str:
    before_pieces = np.loadtxt(
        "../data/prev_board.csv", delimiter=",", dtype="int64"
    )
    if is_sente:
        before_move = list(
            zip(*np.where((pieces != before_pieces) & (before_pieces > 0)))
        )
        after_move = list(
            zip(*np.where((pieces != before_pieces) & (before_pieces <= 0)))
        )
    else:
        before_move = list(
            zip(*np.where((pieces != before_pieces) & (before_pieces < 0)))
        )
        after_move = list(
            zip(*np.where((pieces != before_pieces) & (before_pieces >= 0)))
        )
    kifu: str = ""
    if len(after_move) == 0:
        print("変更ナシ")
    elif len(before_move) == 0:
        diff_coords = list(zip(*(before_pieces != pieces)))
        kifu = (
            str(9 - diff_coords[0][1])
            + KAN_SUJI[diff_coords[0][0]]
            + SHOGI_PIECES[abs(pieces[diff_coords[0]]) - 1]
            + "打"
        )
    else:
        kifu = (
            str(9 - after_move[0][1])
            + KAN_SUJI[after_move[0][0]]
            + SHOGI_PIECES[abs(pieces[after_move[0]]) - 1]
        )
        if before_pieces[before_move[0]] != pieces[after_move[0]]:
            kifu += "成"
        kifu += (
            "(" + str(9 - before_move[0][1]) + str(before_move[0][0] + 1) + ")"
        )
    print(kifu)
    np.savetxt("../data/prev_board.csv", pieces, delimiter=",", fmt="%d")
    return kifu


piece = np.array(
    [
        [-5, -4, -3, -2, 1, -2, -3, -4, -5],
        [0, -6, 0, 0, 0, 0, 0, -7, 0],
        [-8, -8, -8, -8, -8, 0, -8, -8, -8],
        [0, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 8, 6, 0, 0, 0, 8, 0],
        [8, 8, 0, 8, 8, 8, 0, 0, 8],
        [0, 0, 0, 0, 0, 0, 0, 7, 0],
        [5, 4, 3, 2, 1, 2, 3, 4, 5],
    ]
)
mock_generate_kifu(piece, True)
