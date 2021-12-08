from typing import Any, Final, Iterable, List, Tuple

import cv2
import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]


# 頂点検出
def detect_corners(image: IntegerArrayType) -> Tuple[IntegerArrayType, float]:
    # 画像サイズ圧縮
    image_processed: IntegerArrayType = compression_image(
        image=image, max_length=500
    )
    # 将棋盤の候補領域を抽出
    candidacy_areas: List[IntegerArrayType] = get_candidacy_areas(
        image=image_processed
    )
    return candidacy_areas, 0.1

    # return corners, score


# 画像サイズ圧縮
def compression_image(
    image: IntegerArrayType, max_length: int
) -> IntegerArrayType:
    height, width, _ = image.shape
    f = min(max_length / height, max_length / width)
    return cv2.resize(
        image, dsize=None, fx=f, fy=f, interpolation=cv2.INTER_AREA
    )


# 将棋盤の候補領域を抽出
def get_candidacy_areas(image: IntegerArrayType) -> List[IntegerArrayType]:
    # 線分検出
    def get_edge_image(image: IntegerArrayType) -> IntegerArrayType:
        # グレースケール化
        gray_image: IntegerArrayType = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 線分検出・画像生成
        edge_image: IntegerArrayType = cv2.Canny(gray_image, 50, 150)
        return edge_image

    # 輪郭検出 / 一定以上の領域の輪郭を検出
    def get_contours(image: IntegerArrayType) -> List[IntegerArrayType]:
        # 画像中の全ての輪郭を抽出
        contours: Iterable[IntegerArrayType] = cv2.findContours(
            image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )[0]
        # 画像中2割以上の範囲を持つ輪郭のみ抽出
        threshold: float = image.shape[0] * image.shape[1] * 0.2
        return [
            contour
            for contour in contours
            if cv2.contourArea(contour) > threshold
        ]

    # 凸包み処理 / 凸性欠陥を補完
    def get_convexes(
        contours: List[IntegerArrayType],
    ) -> List[IntegerArrayType]:
        convexes: List[IntegerArrayType] = [
            cv2.convexHull(contour) for contour in contours
        ]
        return convexes

    # 輪郭の近似 / 複雑な形状の輪郭を単純化
    def get_polygons(
        convexes: List[IntegerArrayType],
    ) -> List[IntegerArrayType]:
        polygons: List[IntegerArrayType] = [
            cv2.approxPolyDP(convex, 0.02 * cv2.arcLength(convex, True), True)
            for convex in convexes
        ]
        return polygons

    edge_image: IntegerArrayType = get_edge_image(image=image)
    contours: List[IntegerArrayType] = get_contours(image=edge_image)
    convexes: List[IntegerArrayType] = get_convexes(contours=contours)
    polygons: List[IntegerArrayType] = get_polygons(convexes=convexes)
    return polygons


# 候補領域を選定
# def get_selected_area(areas: List[IntegerArrayType]) -> IntegerArrayType:


if __name__ == "__main__":
    image: IntegerArrayType = cv2.imread("./data/test.jpg")
    results = detect_corners(image)
    print(results[0][0])
