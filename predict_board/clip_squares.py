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
    board_image: IntegerArrayType = get_board_image(
        image=image, corners=corners
    )
    squares: IntegerArrayType = get_squares(image=board_image)
    return squares


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


def get_squares(image: IntegerArrayType) -> IntegerArrayType:
    width, height, _ = image.shape
    square_width: int = int(width / 9)
    square_height: int = int(height / 9)

    squares: IntegerArrayType = np.array(
        [
            image[
                j * square_width : (j + 1) * square_width,
                i * square_height : (i + 1) * square_height,
                :,
            ]
            for j in range(9)
            for i in range(9)
        ]
    )

    return squares


if __name__ == "__main__":
    image: IntegerArrayType = cv2.imread("./data/test.jpg")
    # cornersで検出されたものを登録する
    corners: IntegerArrayType = np.array(
        [[[118, 136]], [[610, 110]], [[758, 606]], [[30, 690]]]
    )
    board_image: IntegerArrayType = get_board_image(
        image=image, corners=corners
    )
    square_images: IntegerArrayType = clip_squares(
        image=image, corners=corners
    )

    print(square_images.shape, square_images.dtype)

    cv2.imshow("board", board_image)
    cv2.imshow("squares", np.vstack(square_images))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("./data/board.jpg", board_image)
    cv2.imwrite("./data/square.jpg", np.vstack(square_images))
