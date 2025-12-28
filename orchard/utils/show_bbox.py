from matplotlib import pyplot as plt
from matplotlib.axes import Axes as Axis
import numpy as np
from typing import ArrayLike


def show_bbox(
    *,
    box: list[int],
    ax: Axis,
    image_width: int | None = None,
    image_height: int | None = None,
) -> None:
    """用法，在显示 plt.imshow(image) 之后，把对应的 box 传入，再把对应的轴（plt.gca()）传入，即可绘出边框。

    Args:
        box (list[int]): 4 个元素
        ax (plt.Axis): 通常使用 plt.gca() 传入
    """
    if image_width is not None and image_height is not None:
        box = [
            box[0] / 1000 * image_width,
            box[1] / 1000 * image_height,
            box[2] / 1000 * image_width,
            box[3] / 1000 * image_height,
        ]
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_mask(mask, ax: Axis, random_color=False, borders=True):
    """
    ax 通常是 plt.gca()
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    # if borders:
    #     import cv2

    #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     # Try to smooth contours
    #     contours = [
    #         cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
    #     ]
    #     mask_image = cv2.drawContours(
    #         mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
    #     )
    ax.imshow(mask_image)


def show_image_with_bbox(*, image: ArrayLike, box: list[int]) -> None:
    plt.imshow(image)
    show_bbox(box, plt.gca())
    plt.show()


def show_image_with_mask(*, image: ArrayLike, mask: ArrayLike) -> None:
    plt.imshow(image)
    show_mask(mask, plt.gca())
    plt.show()
