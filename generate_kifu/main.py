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
    diff_pieces = np.where(pieces != before_pieces)
    kifu: str = ""
    if pieces[diff_pieces].size == 0:
        print("変更ナシ")
    elif pieces[diff_pieces].size == 2:
        diff_coords = [
            list(diff_pieces[np.where(pieces[diff_pieces] != 0)[0][0]]),
            list(diff_pieces[np.where(pieces[diff_pieces] == 0)[0][0]]),
        ]
        before_move = list(before_pieces[diff_coords])
        after_move = list(
            [
                pieces[diff_coords[0][0]][diff_coords[0][1]],
                pieces[diff_coords[1][0]][diff_coords[1][1]],
            ]
        )
        print(
            [
                [diff_coords[0][0]],
                [diff_coords[0][1]],
                [diff_coords[1][0]],
                [diff_coords[1][1]],
            ]
        )
        print(before_move)
        print(after_move)

        print(diff_coords[0])
        print(diff_coords)
        print(pieces[5][6])
        kifu = (
            str(9 - diff_coords[0][1])
            + KAN_SUJI[diff_coords[0][0]]
            + SHOGI_PIECES[abs(after_move[0]) - 1]
        )
        if before_move[1] != after_move[0]:
            kifu += "成"
        kifu += (
            "("
            + str(9 - diff_coords[1][1] + 1)
            + str(diff_coords[1][0] + 1)
            + ")"
        )
        print(kifu)
    else:
        diff_coords = list(zip(*diff_pieces))
        kifu = (
            str(9 - diff_coords[0][1])
            + KAN_SUJI[diff_coords[0][0]]
            # + SHOGI_PIECES[abs(pieces[diff_coords[0]]) - 1]
            + "打"
        )
        print(kifu)
    # np.savetxt("../data/prev_board.csv", pieces, delimiter=",", fmt='%d')
    return kifu


piece = np.array(
    [
        [-5, -4, -3, -2, 1, -2, -3, -4, -5],
        [0, -6, 0, 0, 0, 0, 0, -7, 0],
        [-8, -8, -8, -8, -8, 0, -8, -8, -8],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 8, 0, 0],
        [0, 0, 8, 0, 8, 8, 0, 0, 0],
        [8, 8, 6, 8, 0, 0, 0, 8, 8],
        [0, 0, 0, 0, 8, 0, 0, 7, 0],
        [5, 4, 3, 2, 1, 2, 3, 4, 5],
    ]
)
mock_generate_kifu(piece, True)
