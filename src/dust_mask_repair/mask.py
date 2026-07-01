from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NormalizedMask:
    values: np.ndarray
    channel_used: str


@dataclass(frozen=True)
class Component:
    label: int
    area: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class ComponentFilterResult:
    mask: np.ndarray
    kept_components: list[Component]
    removed_small: list[Component]
    removed_large: list[Component]


def normalize_mask(mask: np.ndarray, channel: str = "auto") -> NormalizedMask:
    arr = np.asarray(mask)
    if arr.ndim not in (2, 3):
        raise ValueError(f"Unsupported mask shape: {arr.shape}")
    if channel == "auto":
        return _normalize_auto(arr)
    if channel == "grayscale":
        return NormalizedMask(_normalize_grayscale(arr), "grayscale")
    if channel == "alpha":
        if arr.ndim != 3 or arr.shape[2] not in (2, 4):
            raise ValueError("alpha mask channel requires a 2-channel or 4-channel mask")
        return NormalizedMask(_normalize_channel(arr[:, :, -1]), "alpha")
    if channel == "red":
        if arr.ndim != 3 or arr.shape[2] < 3:
            raise ValueError("red mask channel requires an RGB/RGBA mask")
        return NormalizedMask(_normalize_channel(arr[:, :, 0]), "red")
    if channel == "max_rgb":
        if arr.ndim == 2:
            return NormalizedMask(_normalize_channel(arr), "grayscale")
        if arr.shape[2] < 3:
            raise ValueError("max_rgb mask channel requires an RGB/RGBA mask")
        return NormalizedMask(_normalize_channel(np.max(arr[:, :, :3], axis=2)), "max_rgb")
    raise ValueError(f"Unsupported mask channel: {channel}")


def threshold_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32) >= threshold


def connected_components(binary_mask: np.ndarray) -> tuple[np.ndarray, list[Component]]:
    binary = np.asarray(binary_mask, dtype=bool)
    height, width = binary.shape
    labels = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros((height, width), dtype=bool)
    components: list[Component] = []
    label = 0

    true_pixels = np.argwhere(binary)
    for start_y, start_x in true_pixels:
        if visited[start_y, start_x]:
            continue
        label += 1
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        labels[start_y, start_x] = label
        area = 0
        min_y = max_y = int(start_y)
        min_x = max_x = int(start_x)

        while stack:
            y, x = stack.pop()
            area += 1
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny = y + dy
                    nx = x + dx
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not binary[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    labels[ny, nx] = label
                    stack.append((ny, nx))

        components.append(Component(label=label, area=area, bbox=(min_x, min_y, max_x + 1, max_y + 1)))

    return labels, components


def filter_components(
    binary_mask: np.ndarray,
    min_area: int = 1,
    max_area: int | None = None,
) -> ComponentFilterResult:
    labels, components = connected_components(binary_mask)
    kept: list[Component] = []
    removed_small: list[Component] = []
    removed_large: list[Component] = []
    filtered = np.zeros_like(np.asarray(binary_mask, dtype=bool))

    for component in components:
        if component.area < min_area:
            removed_small.append(component)
            continue
        if max_area is not None and component.area > max_area:
            removed_large.append(component)
            continue
        kept.append(component)
        filtered[labels == component.label] = True

    return ComponentFilterResult(
        mask=filtered,
        kept_components=kept,
        removed_small=removed_small,
        removed_large=removed_large,
    )


def dilate_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    mask = np.asarray(binary_mask, dtype=bool)
    if radius <= 0 or not mask.any():
        return mask.copy()
    height, width = mask.shape
    result = np.zeros_like(mask)
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dy * dy + dx * dx <= radius * radius
    ]
    for dy, dx in offsets:
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)
        if src_y0 >= src_y1 or src_x0 >= src_x1 or dst_y0 >= dst_y1 or dst_x0 >= dst_x1:
            continue
        result[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]
    return result


def feather_mask(binary_mask: np.ndarray, radius: int) -> np.ndarray:
    core = np.asarray(binary_mask, dtype=bool)
    if not core.any():
        return np.zeros(core.shape, dtype=np.float32)
    if radius <= 0:
        return core.astype(np.float32)
    expanded = dilate_mask(core, radius)
    blurred = _box_blur(core.astype(np.float32), radius)
    soft = np.where(expanded, blurred, 0.0).astype(np.float32)
    soft[core] = 1.0
    return np.clip(soft, 0.0, 1.0)


def bbox_with_padding(
    bbox: tuple[int, int, int, int],
    padding: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def bboxes_from_mask(binary_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    _labels, components = connected_components(binary_mask)
    return [component.bbox for component in components]


def _normalize_auto(arr: np.ndarray) -> NormalizedMask:
    if arr.ndim == 2:
        return NormalizedMask(_normalize_channel(arr), "grayscale")

    channels = arr.shape[2]
    if channels in (2, 4):
        alpha = arr[:, :, -1]
        alpha_norm = _normalize_channel(alpha)
        rgb_has_signal = channels == 4 and np.max(arr[:, :, :3]) > 0
        alpha_is_full = np.all(alpha_norm >= 1.0)
        if alpha_norm.max() > 0.0 and not (alpha_is_full and rgb_has_signal):
            return NormalizedMask(alpha_norm, "alpha")

    if channels >= 3:
        rgb = arr[:, :, :3]
        if np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2]):
            return NormalizedMask(_normalize_channel(rgb[:, :, 0]), "grayscale")
        red = _normalize_channel(rgb[:, :, 0])
        green = _normalize_channel(rgb[:, :, 1])
        blue = _normalize_channel(rgb[:, :, 2])
        if red.max() > 0.0 and red.max() >= max(green.max(), blue.max()) * 1.5:
            return NormalizedMask(red, "red")
        return NormalizedMask(np.maximum.reduce([red, green, blue]).astype(np.float32), "max_rgb")

    return NormalizedMask(_normalize_channel(arr[:, :, 0]), "grayscale")


def _normalize_channel(channel: np.ndarray) -> np.ndarray:
    arr = np.asarray(channel)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float32) / float(info.max)
    raise ValueError(f"Unsupported mask dtype: {arr.dtype}")


def _normalize_grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return _normalize_channel(arr)
    if arr.shape[2] == 1:
        return _normalize_channel(arr[:, :, 0])
    if arr.shape[2] < 3:
        return _normalize_channel(arr[:, :, 0])
    red = _normalize_channel(arr[:, :, 0])
    green = _normalize_channel(arr[:, :, 1])
    blue = _normalize_channel(arr[:, :, 2])
    return ((0.2126 * red) + (0.7152 * green) + (0.0722 * blue)).astype(np.float32)


def _box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32)
    padded = np.pad(values, radius, mode="edge")
    result = np.zeros_like(values, dtype=np.float32)
    count = 0
    size = radius * 2 + 1
    for dy in range(size):
        for dx in range(size):
            result += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
            count += 1
    return result / float(count)
