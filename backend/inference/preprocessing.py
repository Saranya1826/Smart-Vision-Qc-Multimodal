"""
Image Preprocessing and Optical Enhancement Module
Implements edge-preserving noise filtering, LAB-space CLAHE enhancement,
and calibrated industrial image quality assurance metrics.
"""
import io
import base64
from typing import Dict, Any, Tuple
import numpy as np
import cv2
from PIL import Image


def ndarray_to_base64(arr: np.ndarray, format: str = "PNG") -> str:
    """Converts RGB/RGBA numpy array into a base64 data URI string."""
    mode = "RGBA" if (arr.ndim == 3 and arr.shape[2] == 4) else "RGB"
    pil_img = Image.fromarray(arr, mode=mode)
    buf = io.BytesIO()
    save_format = "PNG" if mode == "RGBA" else ("JPEG" if format.upper() in ("JPG", "JPEG") else "PNG")
    pil_img.save(buf, format=save_format, quality=92 if save_format == "JPEG" else None)
    mime = "image/png" if save_format == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def preprocess_and_enhance(
    img_rgb: np.ndarray,
    enable_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
    max_dimension: int = 1536
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Preprocesses the component image:
    1. Resizes with aspect ratio preservation if dimension exceeds max_dimension.
    2. Computes optical QA metrics (Laplacian focus, luminance, RMS contrast, SNR).
    3. Applies edge-preserving bilateral filtering to reduce sensor noise without destroying crack tips.
    4. Applies CLAHE in LAB color space if enabled.
    """
    h, w = img_rgb.shape[:2]
    
    # 1. Aspect-ratio preserving resize if needed
    scale = 1.0
    if max(h, w) > max_dimension:
        scale = max_dimension / float(max(h, w))
        new_w = max(16, int(round(w * scale)))
        new_h = max(16, int(round(h * scale)))
        processed_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        processed_rgb = img_rgb.copy()

    # 2. Quality Metrics computation on grayscale
    gray = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2GRAY)
    
    # Laplacian Focus Score (sharpness variance)
    focus_val = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    focus_score = round(focus_val, 2)
    
    # Mean Brightness [0, 100]
    mean_lum = float(np.mean(gray))
    brightness_score = round((mean_lum / 255.0) * 100.0, 1)
    
    # RMS Contrast (standard deviation of gray levels)
    contrast_score = round(float(np.std(gray)), 2)
    
    # Estimated Signal-to-Noise Ratio (dB)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = np.abs(gray.astype(np.float64) - blurred.astype(np.float64))
    noise_sigma = np.std(noise)
    snr_val = float(20.0 * np.log10((mean_lum + 1e-5) / (noise_sigma + 1e-5))) if noise_sigma > 0 else 45.0
    snr_score = round(min(55.0, max(10.0, snr_val)), 1)
    
    # Determine overall QA status
    if focus_score < 40.0:
        quality_check = "WARNING: BLURRED / OUT-OF-FOCUS"
    elif brightness_score < 18.0:
        quality_check = "WARNING: UNDEREXPOSED"
    elif brightness_score > 88.0:
        quality_check = "WARNING: OVEREXPOSED"
    elif contrast_score < 18.0:
        quality_check = "MARGINAL: LOW CONTRAST"
    else:
        quality_check = "PASS - OPTIMAL METROLOGY GRADE"

    # 3. Edge-preserving bilateral denoising
    # Small diameter (d=5) and subtle color sigma (25) preserves hairline cracks & scratches
    denoised = cv2.bilateralFilter(processed_rgb, d=5, sigmaColor=25, sigmaSpace=25)

    # 4. Optional CLAHE enhancement in LAB color space
    if enable_clahe and clip_limit > 0:
        lab = cv2.cvtColor(denoised, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
        enhanced_l = clahe.apply(l_channel)
        merged_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        enhanced_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    else:
        enhanced_rgb = denoised

    enhanced_b64 = ndarray_to_base64(enhanced_rgb, "PNG")

    quality_metrics = {
        "quality_check": quality_check,
        "focus_score": focus_score,
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "noise_level_snr": snr_score,
        "processed_dimensions": (enhanced_rgb.shape[1], enhanced_rgb.shape[0])
    }

    return enhanced_rgb, enhanced_b64, quality_metrics
