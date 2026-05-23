from __future__ import annotations

import binascii
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_COLOR_CHANNELS = {
    0: 1,
    2: 3,
    4: 2,
    6: 4,
}


@dataclass(frozen=True)
class ImageData:
    pixels: np.ndarray
    bit_depth: int
    color_mode: str
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def read_image(path: str | Path) -> ImageData:
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        pixels = read_png(image_path)
        return ImageData(
            pixels=pixels,
            bit_depth=bit_depth_for_array(pixels),
            color_mode=color_mode_for_array(pixels),
            path=image_path,
            metadata={"format": "PNG"},
        )
    if suffix in (".tif", ".tiff"):
        pixels, metadata = read_tiff(image_path)
        return ImageData(
            pixels=pixels,
            bit_depth=bit_depth_for_array(pixels),
            color_mode=color_mode_for_array(pixels),
            path=image_path,
            metadata=metadata,
        )
    if suffix in (".jpg", ".jpeg"):
        pixels = read_jpeg(image_path)
        return ImageData(
            pixels=pixels,
            bit_depth=8,
            color_mode=color_mode_for_array(pixels),
            path=image_path,
            metadata={"format": "JPEG", "reader": "Pillow"},
        )
    raise ValueError(f"Unsupported image extension: {suffix}")


def write_image(path: str | Path, pixels: np.ndarray) -> None:
    image_path = Path(path)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        write_png(image_path, pixels)
        return
    if suffix in (".tif", ".tiff"):
        write_tiff(image_path, pixels)
        return
    if suffix in (".jpg", ".jpeg"):
        write_jpeg(image_path, pixels)
        return
    raise ValueError(f"Unsupported image extension: {suffix}")


def bit_depth_for_array(pixels: np.ndarray) -> int:
    if pixels.dtype == np.uint8:
        return 8
    if pixels.dtype == np.uint16:
        return 16
    if np.issubdtype(pixels.dtype, np.floating):
        return 32
    raise ValueError(f"Unsupported image dtype: {pixels.dtype}")


def color_mode_for_array(pixels: np.ndarray) -> str:
    if pixels.ndim == 2:
        return "grayscale"
    if pixels.ndim == 3:
        channels = pixels.shape[2]
        if channels == 1:
            return "grayscale"
        if channels == 2:
            return "grayscale_alpha"
        if channels == 3:
            return "rgb"
        if channels == 4:
            return "rgba"
    raise ValueError(f"Unsupported image shape: {pixels.shape}")


def as_float32(pixels: np.ndarray) -> np.ndarray:
    arr = np.asarray(pixels)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if np.issubdtype(arr.dtype, np.floating):
        return np.clip(arr.astype(np.float32), 0.0, 1.0)
    raise ValueError(f"Unsupported image dtype: {arr.dtype}")


def restore_dtype(pixels: np.ndarray, dtype: np.dtype) -> np.ndarray:
    clipped = np.clip(pixels, 0.0, 1.0)
    if dtype == np.uint8:
        return np.rint(clipped * 255.0).astype(np.uint8)
    if dtype == np.uint16:
        return np.rint(clipped * 65535.0).astype(np.uint16)
    if np.issubdtype(dtype, np.floating):
        return clipped.astype(dtype)
    raise ValueError(f"Unsupported output dtype: {dtype}")


def read_png(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("Not a PNG file")

    offset = len(PNG_SIGNATURE)
    width = height = bit_depth = color_type = interlace = None
    idat_parts: list[bytes] = []

    while offset < len(data):
        if offset + 8 > len(data):
            raise ValueError("Truncated PNG chunk header")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _compression, _filter, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
        elif chunk_type == b"IDAT":
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None or bit_depth is None or color_type is None:
        raise ValueError("PNG is missing IHDR")
    if interlace != 0:
        raise ValueError("Interlaced PNG is not supported")
    if bit_depth not in (8, 16):
        raise ValueError(f"Unsupported PNG bit depth: {bit_depth}")
    if color_type not in PNG_COLOR_CHANNELS:
        raise ValueError(f"Unsupported PNG color type: {color_type}")

    channels = PNG_COLOR_CHANNELS[color_type]
    bytes_per_sample = bit_depth // 8
    bytes_per_pixel = channels * bytes_per_sample
    row_bytes = width * bytes_per_pixel
    raw = zlib.decompress(b"".join(idat_parts))
    expected = height * (1 + row_bytes)
    if len(raw) != expected:
        raise ValueError("Unexpected PNG decompressed data length")

    rows: list[bytes] = []
    previous = bytearray(row_bytes)
    pos = 0
    for _y in range(height):
        filter_type = raw[pos]
        row = bytearray(raw[pos + 1 : pos + 1 + row_bytes])
        pos += 1 + row_bytes
        _unfilter_png_row(filter_type, row, previous, bytes_per_pixel)
        rows.append(bytes(row))
        previous = row

    dtype = np.uint8 if bit_depth == 8 else ">u2"
    pixels = np.frombuffer(b"".join(rows), dtype=dtype)
    if bit_depth == 16:
        pixels = pixels.astype(np.uint16)
    pixels = pixels.reshape((height, width, channels))
    if channels == 1:
        return pixels[:, :, 0]
    return pixels


def write_png(path: str | Path, pixels: np.ndarray) -> None:
    arr = np.asarray(pixels)
    if arr.dtype not in (np.uint8, np.uint16):
        raise ValueError("PNG output requires uint8 or uint16 pixels")
    if arr.ndim == 2:
        height, width = arr.shape
        channels = 1
        color_type = 0
    elif arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        height, width, channels = arr.shape
        color_type = {1: 0, 3: 2, 4: 6}[channels]
        if channels == 1:
            arr = arr[:, :, 0]
    else:
        raise ValueError(f"Unsupported PNG output shape: {arr.shape}")

    bit_depth = 8 if arr.dtype == np.uint8 else 16
    if bit_depth == 16:
        arr_bytes = arr.astype(">u2", copy=False).tobytes()
    else:
        arr_bytes = arr.tobytes()

    row_bytes = width * channels * (bit_depth // 8)
    scanlines = bytearray()
    for y in range(height):
        start = y * row_bytes
        scanlines.append(0)
        scanlines.extend(arr_bytes[start : start + row_bytes])

    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    compressed = zlib.compress(bytes(scanlines))
    out = bytearray(PNG_SIGNATURE)
    out.extend(_png_chunk(b"IHDR", ihdr))
    out.extend(_png_chunk(b"IDAT", compressed))
    out.extend(_png_chunk(b"IEND", b""))
    Path(path).write_bytes(bytes(out))


def read_tiff(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError:
        tifffile = None

    if tifffile is not None:
        pixels = tifffile.imread(path)
        return np.asarray(pixels), {"format": "TIFF", "reader": "tifffile"}

    with Image.open(path) as image:
        pixels = np.asarray(image)
        metadata = {"format": "TIFF", "reader": "Pillow", "pillow_mode": image.mode}
    return pixels, metadata


def read_jpeg(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def write_tiff(path: str | Path, pixels: np.ndarray) -> None:
    arr = np.asarray(pixels)
    try:
        import tifffile  # type: ignore
    except ModuleNotFoundError:
        tifffile = None

    if tifffile is not None:
        tifffile.imwrite(path, arr)
        return

    if arr.dtype == np.uint16 and arr.ndim == 3 and arr.shape[2] in (3, 4):
        raise ValueError("Writing 16-bit RGB/RGBA TIFF requires optional dependency: tifffile")
    Image.fromarray(arr).save(path)


def write_jpeg(path: str | Path, pixels: np.ndarray) -> None:
    arr = np.asarray(pixels)
    if arr.dtype != np.uint8:
        arr = restore_dtype(as_float32(arr), np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("JPEG output requires RGB pixels")
    Image.fromarray(arr).save(path, quality=95)


def _unfilter_png_row(filter_type: int, row: bytearray, previous: bytearray, bpp: int) -> None:
    if filter_type == 0:
        return
    if filter_type == 1:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + left) & 0xFF
        return
    if filter_type == 2:
        for i in range(len(row)):
            row[i] = (row[i] + previous[i]) & 0xFF
        return
    if filter_type == 3:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = previous[i]
            row[i] = (row[i] + ((left + up) // 2)) & 0xFF
        return
    if filter_type == 4:
        for i in range(len(row)):
            left = row[i - bpp] if i >= bpp else 0
            up = previous[i]
            up_left = previous[i - bpp] if i >= bpp else 0
            row[i] = (row[i] + _paeth(left, up, up_left)) & 0xFF
        return
    raise ValueError(f"Unsupported PNG filter type: {filter_type}")


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _png_chunk(chunk_type: bytes, chunk_data: bytes) -> bytes:
    crc = binascii.crc32(chunk_type)
    crc = binascii.crc32(chunk_data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(chunk_data)) + chunk_type + chunk_data + struct.pack(">I", crc)
