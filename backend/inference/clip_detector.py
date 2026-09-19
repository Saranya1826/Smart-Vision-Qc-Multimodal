"""
Zero-Shot Defect Identification and Coarse Localization Module
Provides multi-modal CLIP visual semantic matching with industrial prompt taxonomy
and an edge/gradient/chromatic fallback analyzer for environments without GPU/weights.
"""
import io
import base64
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image

# Industrial Quality Defect Taxonomy (Standardized 11-Class Spectrum)
INDUSTRIAL_DEFECT_TAXONOMY = [
    "Normal",
    "Crack",
    "Scratch",
    "Dent",
    "Corrosion",
    "Hole",
    "Surface Damage",
    "Missing Part",
    "Deformation",
    "Contamination",
    "Other Anomaly"
]

DEFAULT_PROMPTS = {
    "Normal": "a clean, pristine industrial metal component with a smooth conforming surface and no visible defect",
    "Crack": "an industrial metal component with a sharp crack or hairline fracture line",
    "Scratch": "an industrial component with linear surface scratches or machining abrasive marks",
    "Dent": "an industrial component with an impact dent, localized crater, or mechanical gouge",
    "Corrosion": "an industrial component with rust, chemical oxidation, or corrosion pitting spots",
    "Hole": "an industrial component with an unintended perforated hole or missing material void",
    "Surface Damage": "an industrial component with severe localized surface damage or rough deformation",
    "Missing Part": "an industrial component with a broken or missing section or geometry gap",
    "Deformation": "an industrial component with warped, bent, or distorted geometric profile",
    "Contamination": "an industrial component with oil stain, dark residue, or surface contamination",
    "Other Anomaly": "an industrial component with an abnormal surface irregularity"
}


def ndarray_to_base64_heatmap(heatmap_bgr: np.ndarray) -> str:
    """Encodes BGR color-mapped heatmap into PNG base64 string."""
    buf = io.BytesIO()
    rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    pil_img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


class ZeroShotDefectDetector:
    """
    Zero-Shot Visual Defect Identification Engine.
    Loads real PyTorch CLIP model if transformers/open_clip is installed,
    or smoothly transitions to a Computer Vision Fallback Analyzer.
    """
    def __init__(self):
        self.is_model_available = False
        self.model_status = "FALLBACK"
        self.model_name = "CLIP (FALLBACK - Computer Vision Engine)"
        self.clip_model = None
        self.clip_processor = None
        
        # Check if live transformers/open_clip is available
        self._init_live_clip_if_possible()

    def _init_live_clip_if_possible(self):
        try:
            from transformers import CLIPProcessor, CLIPModel
            import torch
            model_id = "openai/clip-vit-base-patch32"
            self.clip_processor = CLIPProcessor.from_pretrained(model_id)
            self.clip_model = CLIPModel.from_pretrained(model_id)
            self.clip_model.eval()
            self.is_model_available = True
            self.model_status = "REAL"
            self.model_name = "CLIP ViT-B/32 (REAL)"
        except Exception:
            self.is_model_available = False
            self.model_status = "FALLBACK"
            self.model_name = "CLIP (FALLBACK - Computer Vision Engine)"

    def identify_defects(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str] = INDUSTRIAL_DEFECT_TAXONOMY,
        prompts: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Executes defect identification on the actual preprocessed image.
        Never uses hardcoded or predetermined defect results.
        """
        if self.is_model_available and self.clip_model is not None:
            return self._run_live_clip(image_rgb, candidate_classes, prompts)
        else:
            return self._run_fallback_cv_analysis(image_rgb, candidate_classes)

    def _run_live_clip(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str],
        prompts: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Inference with real HuggingFace / PyTorch CLIP model."""
        import torch
        prompt_map = prompts or DEFAULT_PROMPTS
        text_prompts = [prompt_map.get(c, f"an industrial component with {c}") for c in candidate_classes]
        
        pil_img = Image.fromarray(image_rgb)
        inputs = self.clip_processor(
            text=text_prompts,
            images=pil_img,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image  # (1, num_classes)
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]

        scores = {c: round(float(p), 3) for c, p in zip(candidate_classes, probs)}
        predicted_defect = candidate_classes[int(np.argmax(probs))]
        confidence = float(scores[predicted_defect])

        normal_prob = float(scores.get("Normal", 0.0))
        defect_prob = round(1.0 - normal_prob, 3)
        uncertainty = round(float(1.0 - confidence), 3)

        # Generate visual heatmap
        heatmap_b64, peak_coord, anomaly_map = self._generate_visual_heatmap(image_rgb, predicted_defect, confidence)

        return {
            "predicted_defect": predicted_defect,
            "confidence": confidence,
            "clip_confidence": confidence,
            "candidate_defect_types": candidate_classes,
            "similarity_scores": scores,
            "defect_probabilities": scores,
            "normal_probability": normal_prob,
            "defect_probability": defect_prob,
            "uncertainty": uncertainty,
            "heatmap_base64": heatmap_b64,
            "peak_coordinate": peak_coord,
            "anomaly_map": anomaly_map,
            "is_fallback": False,
            "model_status": "REAL",
            "debug_info": {
                "anomaly_score": defect_prob,
                "reason_for_classification": f"Zero-shot CLIP semantic prediction: {predicted_defect} ({confidence*100:.1f}%)"
            }
        }

    def _run_fallback_cv_analysis(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str]
    ) -> Dict[str, Any]:
        """
        Algorithmic Computer Vision Defect Feature Analyzer.
        Follows optical metrology principles:
        1. Background illumination subtraction to isolate localized flaws from ambient machining textures.
        2. Multi-indicator verification: local contrast anomaly, elongated aspect ratio (cracks),
           tortuosity (meandering vs straight line), internal texture variance (surface damage vs dent),
           and chromatic oxidation (corrosion).
        3. Conservative normal thresholding: unless a genuine structural flaw exceeds the minimum
           detection threshold, the component is reliably classified as 'Normal'.
        """
    def _extract_surface_roi(self, image_rgb: np.ndarray) -> np.ndarray:
        """
        Extracts the valid component surface inspection zone (ROI).
        Excludes outer background silhouettes and intentional machined through-holes
        (bolt holes, center bores) so that geometric boundaries are not misdiagnosed as cracks.
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Detect studio/conveyor background (pure white/light > 215 or pure dark < 30)
        is_white_bg = gray > 215
        is_dark_bg = gray < 30
        is_bg = is_white_bg | is_dark_bg
        bg_ratio = np.count_nonzero(is_bg) / float(h * w)

        if bg_ratio > 0.08:
            fg_mask = (~is_bg).astype(np.uint8) * 255
            # Exclude silhouette boundary transition zone (margin of 12 px)
            kernel_bound = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
            bg_dilated = cv2.dilate(is_bg.astype(np.uint8) * 255, kernel_bound)
            
            # Detect intentional through-holes (bolt holes, bores)
            contours, hier = cv2.findContours(fg_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            hole_mask = np.zeros((h, w), dtype=np.uint8)
            if hier is not None:
                for i, c in enumerate(contours):
                    if hier[0][i][3] != -1:
                        area = cv2.contourArea(c)
                        if area >= 20:
                            peri = cv2.arcLength(c, True)
                            circ = (4.0 * np.pi * area) / (peri * peri) if peri > 0 else 0
                            if circ > 0.25:
                                cv2.drawContours(hole_mask, [c], -1, 255, -1)
                                
            hole_dilated = cv2.dilate(hole_mask, kernel_bound)
            surface_roi = (fg_mask > 0) & (bg_dilated == 0) & (hole_dilated == 0)
            return surface_roi.astype(np.uint8) * 255
        else:
            # Full frame component without separate studio background
            return np.ones((h, w), dtype=np.uint8) * 255

    def _run_fallback_cv_analysis(
        self,
        image_rgb: np.ndarray,
        candidate_classes: List[str]
    ) -> Dict[str, Any]:
        """
        Robust Optical Metrology Fallback Analyzer.
        Distinguishes nominal component geometry (outer edges, bolt holes, chamfers)
        from true anomalous surface defects (cracks, scratches, dents, corrosion).
        """
        h, w = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        # Extract valid component surface inspection zone (excludes silhouettes and bolt holes)
        surface_roi = self._extract_surface_roi(image_rgb)
        surf_count = np.count_nonzero(surface_roi)
        if surf_count < 100:
            surface_roi = np.ones((h, w), dtype=np.uint8) * 255
            surf_count = h * w

        # 1. Local Background Subtraction: Isolates localized structural breaks
        local_bg = cv2.GaussianBlur(gray, (35, 35), 0)
        diff = gray - local_bg

        # Measure ambient texture standard deviation strictly over the component surface
        surf_diff = diff[surface_roi > 0]
        std_ambient = max(1.0, float(np.std(surf_diff)))

        # Normalized localized anomaly fields masked to component surface
        z_neg = (np.maximum(0.0, -diff) / std_ambient) * (surface_roi / 255.0)  # Dark fissures
        z_pos = (np.maximum(0.0, diff) / std_ambient) * (surface_roi / 255.0)   # Bright scratch ridges

        # Candidate defect regions: localized contrast deviation > 3.2 sigma on component surface
        defect_candidate_mask = ((z_neg > 3.2) | (z_pos > 3.2)).astype(np.uint8) * 255
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        defect_candidate_mask = cv2.morphologyEx(defect_candidate_mask, cv2.MORPH_OPEN, kernel_clean)

        contours, _ = cv2.findContours(defect_candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        crack_score = 0.0
        scratch_score = 0.0
        dent_score = 0.0
        damage_score = 0.0
        hole_score = 0.0
        peak_coord = (w // 2, h // 2)

        for c in contours:
            area = cv2.contourArea(c)
            if area < 18:
                continue

            rect = cv2.minAreaRect(c)
            bw, bh = rect[1]
            length = max(bw, bh)
            width = max(1.0, min(bw, bh))
            aspect_ratio = length / width

            perimeter = cv2.arcLength(c, True)
            tortuosity = perimeter / (2.0 * (length + width)) if (length + width) > 0 else 1.0

            poly = cv2.approxPolyDP(c, 2.0, False)
            poly_points = len(poly)

            c_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(c_mask, [c], -1, 255, -1)
            mean_v, std_v = cv2.meanStdDev(gray, mask=c_mask)
            internal_std = float(std_v[0][0])
            mean_intensity = float(mean_v[0][0])

            M = cv2.moments(c)
            if M["m00"] > 0:
                cx = int(round(M["m10"] / M["m00"]))
                cy = int(round(M["m01"] / M["m00"]))
                peak_coord = (cx, cy)

            # Elongated surface anomalies (Cracks, Scratches)
            if aspect_ratio >= 3.0 and length >= 24:
                # Distinguish intentional chamfer / bevel bands from real flaws:
                # A machined chamfer/bevel is a wide (width >= 4.0), smooth (internal_std < 22) transition band
                is_smooth_bevel = (width >= 4.0 and internal_std < 22.0)
                
                if not is_smooth_bevel:
                    # Crack: Jagged, meandering fracture path (tortuosity > 1.10 or poly_points >= 6)
                    # Scratch: Highly linear furrow (poly_points <= 5 and tortuosity <= 1.10, width < 3.5 px)
                    if tortuosity > 1.10 or poly_points >= 6:
                        score = min(0.96, 0.45 + (length / 90.0) * 0.35 + (tortuosity - 1.0) * 0.25)
                        crack_score = max(crack_score, score)
                    else:
                        score = min(0.92, 0.45 + (length / 100.0) * 0.45)
                        scratch_score = max(scratch_score, score)

            elif area >= 350:
                if internal_std > 30.0:
                    # High internal roughness / jagged fractured matrix -> SURFACE DAMAGE
                    score = min(0.94, 0.48 + (area / (h * w * 0.015)) * 0.45)
                    damage_score = max(damage_score, score)
                elif 1.0 <= aspect_ratio <= 2.2 and area >= 500:
                    # Smooth depression / impact crater -> DENT
                    score = min(0.88, 0.40 + (area / 2000.0) * 0.40)
                    dent_score = max(dent_score, score)

        # 2. Corrosion / Oxidation Detection (Chromatic deviation in LAB color space)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_c, a_c, b_c = cv2.split(lab)
        rust_pixels = ((a_c > 134) & (b_c > 140) & (l_c < 210) & (surface_roi > 0)).astype(np.uint8)
        rust_count = cv2.countNonZero(rust_pixels)
        rust_ratio = rust_count / float(surf_count)
        corrosion_score = min(0.95, rust_ratio * 40.0 + 0.42) if rust_ratio > 0.003 else 0.02

        # 3. Contamination detection (low-contrast surface residue)
        contamination_score = 0.02
        if rust_ratio > 0.001 and corrosion_score < 0.35:
            contamination_score = min(0.75, rust_ratio * 25.0 + 0.30)

        # Missing Part & Deformation
        missing_part_score = 0.02
        deformation_score = 0.02
        other_anomaly_score = 0.02

        raw_defect_scores = {
            "Crack": round(float(crack_score), 3),
            "Scratch": round(float(scratch_score), 3),
            "Dent": round(float(dent_score), 3),
            "Corrosion": round(float(corrosion_score), 3),
            "Hole": round(float(hole_score), 3),
            "Surface Damage": round(float(damage_score), 3),
            "Missing Part": round(float(missing_part_score), 3),
            "Deformation": round(float(deformation_score), 3),
            "Contamination": round(float(contamination_score), 3),
            "Other Anomaly": round(float(other_anomaly_score), 3)
        }

        max_defect_val = max(raw_defect_scores.values())
        MIN_DEFECT_THRESHOLD = 0.38

        # Conservative Decision Logic:
        # Default to 'Normal' unless clear structural defect indicators exceed threshold
        if max_defect_val < MIN_DEFECT_THRESHOLD:
            predicted_defect = "Normal"
            confidence = max(0.88, round(float(1.0 - max_defect_val), 3))
            normal_prob = confidence
            defect_prob = round(1.0 - normal_prob, 3)
            
            # Distribute residual probability among defect classes
            num_defects = len(raw_defect_scores)
            norm_scores = {"Normal": normal_prob}
            rem = defect_prob / float(num_defects)
            for k in raw_defect_scores:
                norm_scores[k] = round(rem, 3)
            
            classification_reason = (
                f"Normal conforming component: No structural discontinuity detected. "
                f"Ambient surface variance is nominal (normal probability {normal_prob*100:.1f}%)."
            )
        else:
            predicted_defect = max(raw_defect_scores, key=raw_defect_scores.get)
            confidence = raw_defect_scores[predicted_defect]
            normal_prob = round(max(0.02, float(1.0 - confidence)), 3)
            defect_prob = round(1.0 - normal_prob, 3)

            norm_scores = {"Normal": normal_prob}
            # Normalize defect probabilities
            total_defect_raw = sum(raw_defect_scores.values()) + 1e-6
            for k, v in raw_defect_scores.items():
                norm_scores[k] = round(float((v / total_defect_raw) * defect_prob), 3)
            # Ensure predicted defect holds the dominant score
            norm_scores[predicted_defect] = confidence

            classification_reason = (
                f"Defect identified: {predicted_defect} detected with confidence {confidence*100:.1f}%. "
                f"Multi-feature analysis confirms localized anomaly."
            )

        # Ensure all candidate classes are present in similarity_scores
        for c in candidate_classes:
            if c not in norm_scores:
                norm_scores[c] = 0.01

        # Generate 2D Heatmap
        combined_anomaly_map = (z_neg * 0.5 + z_pos * 0.3 + (rust_pixels.astype(np.float32) / 255.0) * 0.4)
        heatmap_b64, peak_coord, anomaly_map = self._generate_visual_heatmap(
            image_rgb, predicted_defect, confidence, combined_anomaly_map
        )

        uncertainty = round(float(1.0 - confidence), 3)

        return {
            "predicted_defect": predicted_defect,
            "confidence": confidence,
            "clip_confidence": confidence,
            "candidate_defect_types": candidate_classes,
            "similarity_scores": norm_scores,
            "defect_probabilities": norm_scores,
            "normal_probability": normal_prob,
            "defect_probability": defect_prob,
            "uncertainty": uncertainty,
            "heatmap_base64": heatmap_b64,
            "peak_coordinate": peak_coord,
            "anomaly_map": anomaly_map,
            "is_fallback": True,
            "model_status": "FALLBACK",
            "debug_info": {
                "anomaly_score": defect_prob,
                "reason_for_classification": classification_reason
            }
        }

    def _generate_visual_heatmap(
        self,
        image_rgb: np.ndarray,
        defect_type: str,
        confidence: float,
        anomaly_map: Optional[np.ndarray] = None
    ) -> Tuple[str, Tuple[int, int], np.ndarray]:
        """Creates a smooth calibrated 2D anomaly heatmap."""
        h, w = image_rgb.shape[:2]

        if anomaly_map is None:
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
            grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
            anomaly_map = cv2.magnitude(grad_x, grad_y)
            anomaly_map = cv2.normalize(anomaly_map, None, 0.0, 1.0, cv2.NORM_MINMAX)

        smoothed = cv2.GaussianBlur(anomaly_map, (15, 15), 0)
        norm_map = cv2.normalize(smoothed, None, 0.0, 1.0, cv2.NORM_MINMAX)

        if defect_type == "Normal":
            # Subtle low-intensity cool blue/cyan background
            thermal_u8 = (norm_map * 30.0).astype(np.uint8)
            peak_coord = (w // 2, h // 2)
        else:
            scaled_map = np.clip(norm_map * (confidence * 1.2), 0.0, 1.0)
            thermal_u8 = (scaled_map * 255.0).astype(np.uint8)
            min_v, max_v, min_loc, max_loc = cv2.minMaxLoc(scaled_map)
            peak_coord = (int(max_loc[0]), int(max_loc[1]))

        heatmap_bgr = cv2.applyColorMap(thermal_u8, cv2.COLORMAP_JET)
        b64 = ndarray_to_base64_heatmap(heatmap_bgr)
        return b64, peak_coord, norm_map
