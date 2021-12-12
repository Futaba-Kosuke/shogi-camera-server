from typing import Any, Final

import cv2
import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]
FloatArrayType: Final[Any] = npt.NDArray[np.float_]

BASE_SIZE: Final[int] = 63


def clip_squares(
    image: IntegerArrayType, corners: IntegerArrayType
) -> IntegerArrayType:
    board_image = get_board_image(image=image, corners=corners)
    print(board_image)


def get_board_image(
    image: IntegerArrayType, corners: IntegerArrayType
) -> IntegerArrayType:
    width: int = BASE_SIZE * 13
    height: int = BASE_SIZE * 14
    transform: FloatArrayType = cv2.getPerspectiveTransform(
        np.float32(corners),
        np.float32([[0, 0], [width, 0], [width, height], [0, height]]),
    )
    result: IntegerArrayType = cv2.warpPerspective(
        image, transform, dsize=(width, height)
    )
    return result


if __name__ == "__main__":
    image: IntegerArrayType = cv2.imread("./data/test.jpg")
    board_image: IntegerArrayType = get_board_image(
        image=image,
        corners=np.array(
            [[[758, 606]], [[30, 690]], [[118, 136]], [[610, 110]]]
        ),
    )

    print(board_image.shape, board_image.dtype)

    cv2.imshow("corners", board_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
