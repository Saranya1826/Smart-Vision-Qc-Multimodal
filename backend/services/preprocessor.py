"""
Image Preprocessing and Quality Assessment Service (Stage 2)
Implements CLAHE contrast enhancement, bilateral filtering, and image quality metrics.
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple
from backend.services.mock_data import ndarray_to_base64

def evaluate_image_quality(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Computes objective industrial optical quality metrics:
    - Focus (Laplacian variance)
    - Brightness (Normalized mean luminance)
    - Contrast (Root-Mean-Square intensity standard deviation)
    - Noise Level (Signal-to-Noise Ratio in dB)
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    
    # 1. Focus Metric (Laplacian variance)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    focus_var = float(laplacian.var())
    
    # 2. Brightness Metric (Mean luminance 0-100)
    mean_val = float(np.mean(gray))
    brightness_norm = round((mean_val / 255.0) * 100.0, 1)
    
    # 3. Contrast Metric (RMS standard deviation)
    contrast_rms = float(np.std(gray))
    
    # 4. Noise SNR Metric (dB)
    noise_est = np.median(np.abs(laplacian)) / 0.6745
    if noise_est > 1e-4:
        snr_db = float(20 * np.log10((mean_val + 1e-5) / noise_est))
    else:
        snr_db = 42.0
    snr_db = round(max(10.0, min(55.0, snr_db)), 1)
    
    # Overall Quality Assessment Check
    if focus_var > 120 and 30 <= brightness_norm <= 80 and contrast_rms > 35:
        quality_status = "PASSED (Optimal Industrial Inspection Grade)"
    elif focus_var <= 80:
        quality_status = "WARNING (Low Focus / Motion Blur Detected)"
    elif brightness_norm < 25 or brightness_norm > 85:
        quality_status = "WARNING (Sub-optimal Exposure / Glare)"
    else:
        quality_status = "ACCEPTABLE (Within Standard Tolerance)"
        
    return {
        "quality_check": quality_status,
        "focus_score": round(focus_var, 1),
        "brightness_score": brightness_norm,
        "contrast_score": round(contrast_rms, 1),
        "noise_level_snr": snr_db
    }

def preprocess_and_enhance(
    img_rgb: np.ndarray,
    enable_clahe: bool = True,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8
) -> Tuple[np.ndarray, str, Dict[str, Any]]:
    """
    Applies CLAHE enhancement on the L-channel of LAB color space and mild bilateral filtering,
    returning the enhanced image array, base64 data URI, and quality metrics.
    """
    # Quality metrics of raw input
    quality_metrics = evaluate_image_quality(img_rgb)
    
    if not enable_clahe:
        enhanced_b64 = ndarray_to_base64(img_rgb, "PNG")
        return img_rgb, enhanced_b64, quality_metrics
        
    # Convert to LAB color space to equalize Luminance without chromatic distortion
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_enhanced = clahe.apply(l_channel)
    
    # Re-merge channels and convert back to RGB
    lab_enhanced = cv2.merge((l_enhanced, a_channel, b_channel))
    rgb_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)
    
    # Mild bilateral filtering to suppress sensor noise while preserving sharp crack edges
    rgb_smoothed = cv2.bilateralFilter(rgb_enhanced, d=5, sigmaColor=45, sigmaSpace=45)
    
    enhanced_b64 = ndarray_to_base64(rgb_smoothed, "PNG")
    return rgb_smoothed, enhanced_b64, quality_metrics
