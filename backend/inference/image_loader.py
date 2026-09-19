"""
Image Loader and Input Validator Module
Handles secure loading, decoding, format normalization, and validation of uploaded component images.
"""
import io
import base64
import hashlib
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image
import cv2

# Maximum allowable upload payload: 25 Megabytes
MAX_IMAGE_SIZE_BYTES = 25 * 1024 * 1024
SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "BMP", "WEBP"}


def validate_and_decode_image(raw_bytes: bytes) -> Dict[str, Any]:
    """
    Validates and decodes image raw bytes.
    Ensures safe handling of RGBA, Grayscale, and standard RGB.
    Returns normalized 3-channel RGB numpy array and metadata.
    """
    if not raw_bytes or len(raw_bytes) == 0:
        raise ValueError("Image data is empty. Please select or upload a valid component image.")

    if len(raw_bytes) > MAX_IMAGE_SIZE_BYTES:
        raise ValueError(
            f"Image file size ({round(len(raw_bytes)/(1024*1024), 2)} MB) exceeds the 25 MB maximum limit."
        )

    # 1. Attempt PIL decoding and verify format
    try:
        pil_img = Image.open(io.BytesIO(raw_bytes))
        img_format = pil_img.format.upper() if pil_img.format else "PNG"
    except Exception as e:
        raise ValueError(f"Unable to decode image file. File may be corrupted or not a valid image format. ({str(e)})")

    if img_format not in SUPPORTED_FORMATS and img_format != "MPO":
        raise ValueError(
            f"Unsupported image format: '{img_format}'. Supported formats are: PNG, JPG, JPEG, BMP, WEBP."
        )

    # 2. Normalize color space to RGB uint8
    try:
        if pil_img.mode == "RGBA":
            # Composite transparent areas over neutral industrial background (white/gray)
            background = Image.new("RGBA", pil_img.size, (255, 255, 255, 255))
            alpha_composite = Image.alpha_composite(background, pil_img)
            pil_rgb = alpha_composite.convert("RGB")
        elif pil_img.mode == "L":
            # Grayscale to 3-channel RGB
            pil_rgb = pil_img.convert("RGB")
        elif pil_img.mode == "CMYK":
            pil_rgb = pil_img.convert("RGB")
        elif pil_img.mode != "RGB":
            pil_rgb = pil_img.convert("RGB")
        else:
            pil_rgb = pil_img

        img_rgb = np.array(pil_rgb, dtype=np.uint8)
    except Exception as e:
        raise ValueError(f"Color space normalization failed: {str(e)}")

    h, w = img_rgb.shape[:2]
    if h < 16 or w < 16:
        raise ValueError(f"Image dimensions ({w}x{h}) are too small for optical metrology. Minimum size is 16x16 px.")

    # 3. Compute integrity SHA-256 and metadata
    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    file_size_kb = round(len(raw_bytes) / 1024.0, 1)

    # Standardize data URL for web transport
    mime_type = "image/jpeg" if img_format in ("JPEG", "JPG") else "image/png"
    clean_b64 = f"data:{mime_type};base64,{base64.b64encode(raw_bytes).decode('utf-8')}"

    return {
        "image_rgb": img_rgb,
        "width": w,
        "height": h,
        "dimensions": (w, h),
        "format": img_format,
        "file_size_kb": file_size_kb,
        "sha256_hash": sha256_hash,
        "clean_base64": clean_b64,
        "raw_bytes": raw_bytes
    }


def load_image_from_base64(image_base64: str) -> Dict[str, Any]:
    """
    Decodes base64 string (with or without data URI header) into image array and metadata.
    """
    if not image_base64 or not isinstance(image_base64, str):
        raise ValueError("Invalid image upload: No image data provided.")

    clean_str = image_base64.strip()
    if "," in clean_str:
        header, encoded = clean_str.split(",", 1)
    else:
        encoded = clean_str

    try:
        raw_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image data: {str(e)}")

    return validate_and_decode_image(raw_bytes)
