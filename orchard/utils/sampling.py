from typing import Any
import numpy as np
import random


def uniform_sampling(frames: list[Any], n_samples: int = 16) -> list[Any]:
    """
    均匀采样
    frames: 帧列表或数组

    示例使用:
        frames = [...]  # 你的帧列表
        sampled_frames = uniform_sampling(frames, 16)   n_samples: 需要采样的数量
    """
    n_frames = len(frames)

    if n_frames <= n_samples:
        # 如果帧数不足，直接返回所有帧（或根据需要处理）
        return frames

    # 计算采样间隔
    step = n_frames / n_samples

    # 生成均匀分布的索引
    indices = [int(i * step) for i in range(n_samples)]

    # 确保索引不超出范围。实际上并不会出现超出的情况。
    indices = [min(idx, n_frames - 1) for idx in indices]

    return [frames[i] for i in indices]


def equally_spaced_sampling(frames: list[Any], n_samples: int = 16) -> list[Any]:
    """
    等间距采样（包含首尾帧）
    """
    n_frames = len(frames)

    if n_frames <= n_samples:
        return frames

    # 计算等间距索引（包含0和n_frames-1）
    indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)

    # 确保索引唯一且有序
    indices = sorted(set(indices))

    return [frames[i] for i in indices]


def center_weighted_sampling(
    frames: list[Any], n_samples: int = 16, center_ratio: float = 0.6
) -> list[Any]:
    """
    中间区域重点采样
    center_ratio: 中间部分占总采样数的比例
    """
    n_frames = len(frames)

    if n_frames <= n_samples:
        return frames

    num_center = int(n_samples * center_ratio)
    num_edges = n_samples - num_center

    # 中间部分均匀采样
    center_start = n_frames // 4
    center_end = 3 * n_frames // 4
    center_indices = np.linspace(center_start, center_end, num_center, dtype=int)

    # 边缘部分均匀采样
    edge_indices = []
    if num_edges > 0:
        left_indices = np.linspace(
            0, center_start - 1, num_edges // 2, dtype=int, endpoint=False
        )
        right_indices = np.linspace(
            center_end + 1, n_frames - 1, num_edges - len(left_indices), dtype=int
        )
        edge_indices = list(left_indices) + list(right_indices)

    # 合并并排序
    indices = sorted(set(list(center_indices) + edge_indices))

    # 如果采样数不足，补充
    while len(indices) < n_samples and len(indices) < n_frames:
        for i in range(n_frames):
            if i not in indices:
                indices.append(i)
                break

    return [frames[i] for i in indices[:n_samples]]


def random_sampling(frames: list[Any], n_samples: int = 16, seed=None) -> list[Any]:
    """
    随机采样
    seed: 随机种子，确保可重复性
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n_frames = len(frames)

    if n_frames <= n_samples:
        return frames

    indices = random.sample(range(n_frames), n_samples)
    indices.sort()  # 保持时间顺序

    return [frames[i] for i in indices]


def adaptive_sampling_by_difference(frames: list[Any], n_samples: int = 16) -> list[Any]:
    """
    基于帧间差异的自适应采样。
    TODO: 支持 metric
    """
    n_frames = len(frames)

    if n_frames <= n_samples:
        return frames

    # 计算帧间差异（这里需要根据实际情况计算）
    # 假设frames是图像数组，可以计算像素差异
    differences = []
    for i in range(1, n_frames):
        # 这里需要根据你的帧数据类型实现差异计算
        # 例如，对于图像可以使用MSE、SSIM等
        diff = np.mean(np.abs(frames[i] - frames[i - 1]))  # 简化示例
        differences.append(diff)

    # 归一化差异作为权重
    weights = (
        np.array(differences) / sum(differences)
        if sum(differences) > 0
        else np.ones(len(differences)) / len(differences)
    )

    # 根据权重采样
    indices = [0]  # 总是包含第一帧
    cumulative_weights = np.cumsum(weights)

    for _ in range(1, n_samples):
        target = np.random.rand() * cumulative_weights[-1]
        idx = np.searchsorted(cumulative_weights, target) + 1
        if idx < n_frames and idx not in indices:
            indices.append(idx)

    indices.sort()
    return [frames[i] for i in indices]
