from typing import Any, Final

import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]


def mock_predict_board(image: IntegerArrayType) -> IntegerArrayType:
    return image
