import numpy as np
from PIL import Image
from io import BytesIO
import base64
from pathlib import Path
import imageio.v3 as iio
import cv2
from orchard.utils.logging_utils import get_logger

logger = get_logger(__name__)


def image_to_base64(*, image: np.ndarray | Image.Image | str | Path) -> str:
    """
    将 numpy 数组转换为 base64 编码的 JPEG 格式

    Args:
        image: 支持多种图像输入类型
    Returns:
        str: base64 编码的 JPEG 格式
    """
    match image:
        case str() | Path() as file_path if file_path.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
        ):
            pil_image = Image.open(file_path)
            buffer = BytesIO()
            pil_image.save(buffer, format="JPEG")
            encoded_data = buffer.getvalue()

        case Image.Image() as pil_img:
            buffer = BytesIO()
            pil_img.save(buffer, format="JPEG")
            encoded_data = buffer.getvalue()

        case np.ndarray() as np_image:
            # 处理 numpy 数组
            if np_image.dtype != np.uint8:
                np_image = np_image.astype(np.uint8)

            if np_image.shape[-1] not in [3, 4]:
                np_image = np.transpose(np_image, (1, 2, 0))

            success, encoded_array = cv2.imencode(".jpg", np_image)
            if not success:
                raise ValueError("图像编码失败")
            encoded_data = encoded_array.tobytes()

        case _:
            raise ValueError(f"不支持的图像类型: {type(image)}")

    return base64.b64encode(encoded_data).decode("utf-8")


def base64_to_image(*, base64_string: str) -> np.ndarray:
    padding = 4 - (len(base64_string) % 4)
    if padding != 4:
        base64_string += "=" * padding
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    img_data = base64.b64decode(base64_string)
    bytes_io = BytesIO(img_data)
    image = Image.open(bytes_io)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def calculate_entropy(*, image: np.ndarray) -> float:
    """
    计算图像的熵

    Args:
        image (ArrayLike): 输入的图像，必须是 BGR 格式
    """

    # 读取图像并转换为灰度图。计算直方图，即每个像素值的频率
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 归一化直方图以得到概率分布
    hist /= hist.sum()

    # 计算信息熵
    entropy = -np.sum(hist * np.log2(hist + (hist == 0)))
    print(f"entropy is {entropy:.2f}")
    return entropy


def video_to_gif(
    *, video_path: str | Path, gif_path: str | Path | None = None, duration: int = 100
):
    """
    转换视频转换为 GIF

    Args:

        video_path (str): 视频路径
        gif_path (str): GIF 路径，默认在原路径，作为同名文件并改后缀名为 .gif
        duration (int): GIF 帧间隔，单位毫秒，直接控制 FPS
    """
    video_path = Path(video_path)
    assert video_path.exists(), f"Video file not found: {video_path}"
    gif_path = gif_path or video_path.with_suffix(".gif")
    gif_path = Path(gif_path)
    gif_path.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for frame in iio.imiter(video_path):
        frames.append(frame)
    path = Path(gif_path)
    path_dir = path.parent
    path_dir.mkdir(parents=True, exist_ok=True)
    iio.imwrite(uri=gif_path, image=frames, duration=duration)


def depth_to_bgr_image(*, depth_image: np.ndarray) -> np.ndarray:
    """
    将深度图转换为 BGR 图像
    """
    # 归一化深度图到0-255范围以便可视化
    depth_normalized = cv2.normalize(
        depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )

    # 将灰度深度图转换为彩色图像
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_MAGMA)
    return depth_colormap
