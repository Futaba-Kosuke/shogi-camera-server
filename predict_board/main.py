from typing import Any, Final

import numpy as np
import numpy.typing as npt

from .clip_squares import clip_squares
from .corners import detect_corners

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]


def predict_board(image: IntegerArrayType) -> IntegerArrayType:
    corners: IntegerArrayType = detect_corners(image=image)
    square_image: IntegerArrayType = clip_squares(image=image, corners=corners)
    return square_image
