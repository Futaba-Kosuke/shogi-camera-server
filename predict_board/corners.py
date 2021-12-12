from typing import Any, Final, Iterable, List

import cv2
import numpy as np
import numpy.typing as npt

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]
FloatArrayType: Final[Any] = npt.NDArray[np.float_]


# 頂点検出
def detect_corners(image: IntegerArrayType) -> IntegerArrayType:
    # 画像サイズ圧縮
    image_processed: IntegerArrayType = compression_image(
        image=image, max_length=500
    )

    # 将棋盤の候補領域を抽出
    candidacy_areas: List[IntegerArrayType] = get_candidacy_areas(
        image=image_processed
    )
    # 候補領域を選定
    selected_area: IntegerArrayType = get_selected_area(areas=candidacy_areas)

    # 四隅座標とスコアを取得
    corners: IntegerArrayType = np.int32(
        selected_area * image.shape[0] / image_processed.shape[0]
    )

    return corners


# 画像サイズ圧縮
def compression_image(
    image: IntegerArrayType, max_length: int
) -> IntegerArrayType:
    height, width, _ = image.shape
    f = min(max_length / height, max_length / width)
    return cv2.resize(
        image, dsize=None, fx=f, fy=f, interpolation=cv2.INTER_AREA
    )


# 線分画像取得
def get_edge_image(image: IntegerArrayType) -> IntegerArrayType:
    # グレースケール化
    gray_image: IntegerArrayType = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 線分検出・画像生成
    edge_image: IntegerArrayType = cv2.Canny(gray_image, 50, 150)
    return edge_image


# 将棋盤の候補領域を抽出
def get_candidacy_areas(image: IntegerArrayType) -> List[IntegerArrayType]:
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
            cv2.approxPolyDP(
                curve=convex,
                epsilon=0.02 * cv2.arcLength(curve=convex, closed=True),
                closed=True,
            )
            for convex in convexes
        ]
        return polygons

    edge_image: IntegerArrayType = get_edge_image(image=image)
    contours: List[IntegerArrayType] = get_contours(image=edge_image)
    convexes: List[IntegerArrayType] = get_convexes(contours=contours)
    polygons: List[IntegerArrayType] = get_polygons(convexes=convexes)
    return polygons


# 候補領域を選定
def get_selected_area(areas: List[IntegerArrayType]) -> IntegerArrayType:
    euclidean_distances: List[float] = []
    for area in areas:
        # 四隅の座標から辺の長さを計算
        line_lengths: List[float] = [
            np.linalg.norm(area[(i + 1) % 4] - area[i]) for i in range(4)
        ]
        # 四隅の座標から領域の占める面積を計算し、正方形と仮定して一辺の長さを計算
        estimated_line_length: float = cv2.contourArea(area) ** 0.5
        # 正方形との誤差をユークリッド距離で計算
        euclidean_distance = sum(
            [
                (line_length - estimated_line_length) ** 2
                for line_length in line_lengths
            ]
        )
        euclidean_distances.append(euclidean_distance)
    return areas[np.argmin(euclidean_distances)]


if __name__ == "__main__":
    image: IntegerArrayType = cv2.imread("./data/test.jpg")
    corners = detect_corners(image)

    print(corners)

    # 画像表示
    cv2.drawContours(image, [corners], -1, (0, 255, 0), 2)
    cv2.imshow("corners", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
