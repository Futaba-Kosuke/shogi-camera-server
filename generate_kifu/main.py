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
    "龍",
    "と",
]

KAN_SUJI: List[str] = ["一", "二", "三", "四", "五", "六", "七", "八", "九"]


def generate_kifu(
    pieces: IntegerArrayType, is_sente: bool, csv_path: str, prev_kifu: str
) -> str:
    before_pieces = np.loadtxt(csv_path, delimiter=",", dtype=np.int32)
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
    if len(after_move) == 0:
        pass
    elif len(before_move) == 0:
        diff_coords = list(zip(*(np.where(before_pieces != pieces))))
        x, y = 9 - diff_coords[0][1], KAN_SUJI[diff_coords[0][0]]
        kifu = f"{x}{y}{SHOGI_PIECES[abs(pieces[diff_coords[0]]) - 1]}打"
    else:
        x, y = 9 - after_move[0][1], KAN_SUJI[after_move[0][0]]
        destination = "同" if prev_kifu.startswith(f"{x}{y}") else f"{x}{y}"
        if before_pieces[before_move[0]] != pieces[after_move[0]]:
            piece = f"{SHOGI_PIECES[abs(before_pieces[before_move[0]]) - 1]}成"
        else:
            piece = f"{SHOGI_PIECES[abs(pieces[after_move[0]]) - 1]}"
        prev_x, prev_y = 9 - before_move[0][1], before_move[0][0] + 1
        kifu = f"{destination}{piece}({prev_x}{prev_y})"
    np.savetxt(csv_path, pieces, delimiter=",", fmt="%d")
    return kifu


if __name__ == "__main__":
    pieces = np.array(
        [
            [-5, -4, -3, -2, -1, -2, -3, -4, -5],
            [0, -7, 0, 0, 0, 0, 0, -6, 0],
            [-8, -8, -8, -8, -8, -8, -8, -8, -8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [8, -6, 0, 0, 0, 0, 0, -7, 0],
            [5, 4, 3, 2, 1, 2, 3, 4, 5],
        ]
    )
    kifu: str = generate_kifu(
        pieces=pieces,
        is_sente=True,
        csv_path="./data/csv/TEST.csv",
        prev_kifu="平手",
    )
    print(kifu)
