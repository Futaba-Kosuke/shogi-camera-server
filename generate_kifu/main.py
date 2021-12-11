from typing import Any, Final

import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]


def mock_generate_kifu(pieces: IntegerArrayType, is_sente: bool) -> str:
    return "7六歩(77)"
