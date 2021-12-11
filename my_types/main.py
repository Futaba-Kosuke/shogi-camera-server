from typing import Any, Final, TypedDict

import numpy as np
import numpy.typing as npt

HelloWorldType: Final[Any] = TypedDict("HelloWorldType", {"Hello": str})

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]
