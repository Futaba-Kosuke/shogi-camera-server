from typing import Any, Final, Iterable, List, Tuple, Callable

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import basinhopping

IntegerArrayType: Final[Any] = npt.NDArray[np.int_]
FloatArrayType: Final[Any] = npt.NDArray[np.float_]


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
    # 候補領域を選定
    selected_area: IntegerArrayType = get_selected_area(areas=candidacy_areas)

    # 最適化する目的関数を取得
    cost_function: Callable[[FloatArrayType], float] = get_cost_function(
        image=image_processed, area=selected_area
    )
    # 目的関数に沿って四隅座標を最適化
    results: Any = basinhopping(
        func=cost_function,
        x0=selected_area.flatten(),
        T=0.1,
        niter=250,
        stepsize=3,
    )

    # 四隅座標とスコアを取得
    corners: IntegerArrayType = np.int32(
        results.x.reshape(4, 2) * image.shape[0] / image_processed.shape[0]
    )
    score: float = results.fun * (-1 / 255)

    return corners, score


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
    squared_errors: List[float] = []
    for area in areas:
        # 四隅の座標から辺の長さを計算
        line_lengths: List[float] = [
            np.linalg.norm(area[(i + 1) % 4] - area[i]) for i in range(4)
        ]
        # 四隅の座標から領域の占める面積を計算し、正方形と仮定して一辺の長さを計算
        estimated_line_length: float = cv2.contourArea(area) ** 0.5
        # 正方形との誤差をユークリッド距離で計算
        squared_error = sum(
            [
                (line_length - estimated_line_length) ** 2
                for line_length in line_lengths
            ]
        )
        squared_errors.append(squared_error)
    return areas[np.argmin(squared_errors)]


def get_cost_function(
        image: IntegerArrayType, area: IntegerArrayType
) -> Callable[[FloatArrayType], float]:
    def get_line_image(
            image: IntegerArrayType, area: IntegerArrayType
    ) -> IntegerArrayType:
        line_image: IntegerArrayType = np.zeros(image.shape, np.uint8)

        # 線分画像取得
        edge_image: IntegerArrayType = get_edge_image(image=image)
        # 周囲長取得
        perimeter_length: float = cv2.arcLength(curve=area, closed=True)
        # 直線取得
        lines: IntegerArrayType = cv2.HoughLinesP(
            image=edge_image,
            rho=1,
            theta=np.pi / 180,
            threshold=int(perimeter_length / 12),
            minLineLength=int(perimeter_length / 200),
        )
        # 直線を全て描画（不要な直線含む）
        for x1, y1, x2, y2 in lines[:, 0]:
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 1)
        # グレースケール化
        line_image = line_image[:, :, 0]

        # 領域外を描画範囲から除外
        image_size: float = (image.shape[0] * image.shape[1]) ** 0.5
        # マスク画像の生成
        mask: IntegerArrayType = np.zeros(line_image.shape, np.uint8)
        cv2.fillConvexPoly(mask, area, 1)
        kernel = np.ones(
            (int(image_size / 10), int(image_size / 10)), np.uint8
        )
        mask = cv2.erode(mask, kernel, iterations=1)
        # マスクがけ
        line_image[np.where(mask == 0)] = 0

        return line_image

    def get_board_image() -> FloatArrayType:
        half_a = np.fromfunction(
            lambda i, j: ((10 - i) ** 2) / 100.0, (10, 20), dtype=np.float32
        )
        half_b = np.rot90(half_a, 2)
        cell_a = np.r_[half_a, half_b]
        cell_b = np.rot90(cell_a)
        cell = np.maximum(cell_a, cell_b)
        return np.tile(cell, (9, 9))

    # リサイズ
    image = cv2.resize(
        image, dsize=None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA
    )
    area = np.int32(area * 0.7)

    # 直線画像
    line_image: IntegerArrayType = get_line_image(image=image, area=area)
    # 最適化用画像
    board_image: FloatArrayType = get_board_image()

    def cost_function(corners: FloatArrayType) -> float:
        image_corners: FloatArrayType = np.float32(corners).reshape(4, 2) * 0.7
        board_size: int = board_image.shape[0]
        board_corners: FloatArrayType = np.float32(
            [
                [0, 0],
                [0, board_size],
                [board_size, board_size],
                [board_size, 0],
            ]
        )

        # 射影変換の変換行列を生成／正面向きの盤面画像を予測された盤面の座標に合わせて射影
        transform: FloatArrayType = cv2.getPerspectiveTransform(
            src=board_corners, dst=image_corners
        )
        # 射影変換の実行
        board_inclined: FloatArrayType = cv2.warpPerspective(
            src=board_image,
            M=transform,
            dsize=(image.shape[1], image.shape[0]),
        )

        result = line_image * board_inclined
        return -np.average(result[np.where(result > 255 * 0.1)])

    return cost_function


if __name__ == "__main__":
    image: IntegerArrayType = cv2.imread("./data/test_01.jpg")
    corners, score = detect_corners(image)

    # 画像表示
    cv2.drawContours(image, [corners], -1, (0, 255, 0), 2)
    cv2.imshow('corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
