"""
Comprehensive Unit and Functional Test Suite for SmartVisionOC
Validates end-to-end 10-stage optical metrology pipeline, live image upload flow,
defect localization, and state isolation between consecutive runs.
"""
import os
import sys
import io
import base64
import numpy as np
import cv2
from PIL import Image

# Ensure project root is in sys.path
curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if curr_dir not in sys.path:
    sys.path.insert(0, curr_dir)

from backend.services.pipeline import InspectionPipeline
from backend.schemas import InspectionRequest, OperatorInputPayload, InspectionConfig
from backend.inference.image_loader import validate_and_decode_image, load_image_from_base64


def create_test_image_base64(img_rgb: np.ndarray, format: str = "PNG") -> str:
    """Helper to convert numpy array into base64 data URI."""
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def test_preset_crack_specimen():
    """Test 0: Preset demonstration specimen for Crack."""
    pipeline = InspectionPipeline()
    req = InspectionRequest(
        specimen_id="FLG-CRK-01",
        operator_input=OperatorInputPayload(
            notes="Abnormal machining chatter observed on turn 3",
            critical_sealing_surface=True
        )
    )
    res = pipeline.execute_inspection(req)
    
    # Assert all 10 stages exist
    assert res.section_1_acquisition is not None
    assert res.section_2_preprocessing is not None
    assert res.section_3_defect_identification is not None
    assert res.section_4_segmentation is not None
    assert res.section_5_spatial_analysis is not None
    assert res.section_6_operator_input is not None
    assert res.section_7_evidence_fusion is not None
    assert res.section_8_risk_scoring is not None
    assert res.section_9_decision_engine is not None
    assert res.section_10_inspection_report is not None

    assert res.section_3_defect_identification.predicted_defect == "Crack"
    assert res.section_9_decision_engine.disposition == "REJECT"
    assert res.section_8_risk_scoring.weighted_risk_score >= 0.65
    print("[PASS] test_preset_crack_specimen")


def test_preset_normal_specimen():
    """Test 0b: Preset demonstration specimen for Normal."""
    pipeline = InspectionPipeline()
    req = InspectionRequest(
        specimen_id="FLG-NRM-00",
        operator_input=OperatorInputPayload(notes="Conforming batch verification")
    )
    res = pipeline.execute_inspection(req)
    
    assert res.section_3_defect_identification.predicted_defect == "Normal"
    assert res.section_9_decision_engine.disposition == "PASS"
    assert res.section_8_risk_scoring.weighted_risk_score < 0.25
    print("[PASS] test_preset_normal_specimen")


def test_upload_normal_component():
    """TEST 1: Upload a pristine/normal industrial component image."""
    pipeline = InspectionPipeline()
    
    # Create synthetic smooth metallic flange surface (uniform with subtle machining rings)
    h, w = 512, 512
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - w/2)**2 + (y - h/2)**2)
    # Subtle concentric phonographic grooves without any fractures
    metal = 175 + (np.sin(r * 0.4) * 8).astype(np.uint8)
    normal_img = cv2.merge([metal, metal, metal])

    upload_b64 = create_test_image_base64(normal_img)
    req = InspectionRequest(image_base64=upload_b64)
    res = pipeline.execute_inspection(req)

    # Must classify as Normal with PASS disposition and low risk
    assert res.section_3_defect_identification.predicted_defect == "Normal", \
        f"Expected Normal but got {res.section_3_defect_identification.predicted_defect}"
    assert res.section_9_decision_engine.disposition == "PASS", \
        f"Expected PASS disposition but got {res.section_9_decision_engine.disposition}"
    assert res.section_8_risk_scoring.weighted_risk_score <= 0.25, \
        f"Expected risk <= 0.25 but got {res.section_8_risk_scoring.weighted_risk_score}"
    assert res.section_4_segmentation.defect_area_px == 0
    assert res.section_4_segmentation.bounding_box == [0, 0, 0, 0]
    print(f"[PASS] TEST 1 (Normal Upload): Decision={res.section_9_decision_engine.disposition}, Risk={res.section_8_risk_scoring.weighted_risk_score}")


def test_upload_crack_component():
    """TEST 2: Upload an industrial component containing a sharp crack."""
    pipeline = InspectionPipeline()
    
    h, w = 512, 512
    y, x = np.mgrid[0:h, 0:w]
    r = np.sqrt((x - w/2)**2 + (y - h/2)**2)
    metal = 175 + (np.sin(r * 0.4) * 8).astype(np.uint8)
    crack_img = cv2.merge([metal, metal, metal])

    # Draw a distinct, sharp meandering dark crack fissure with opposing bright edge gradient
    pts = np.array([[220, 180], [235, 215], [242, 245], [250, 290], [268, 340], [275, 380]], np.int32)
    cv2.polylines(crack_img, [pts], isClosed=False, color=(20, 20, 20), thickness=3)
    cv2.polylines(crack_img, [pts + 1], isClosed=False, color=(255, 255, 255), thickness=1)

    upload_b64 = create_test_image_base64(crack_img)
    req = InspectionRequest(
        image_base64=upload_b64,
        operator_input=OperatorInputPayload(critical_sealing_surface=True)
    )
    res = pipeline.execute_inspection(req)

    # Must classify as Crack with REJECT disposition and high risk
    assert res.section_3_defect_identification.predicted_defect == "Crack", \
        f"Expected Crack but got {res.section_3_defect_identification.predicted_defect}"
    assert res.section_9_decision_engine.disposition == "REJECT", \
        f"Expected REJECT disposition but got {res.section_9_decision_engine.disposition}"
    assert res.section_8_risk_scoring.weighted_risk_score >= 0.65, \
        f"Expected risk >= 0.65 but got {res.section_8_risk_scoring.weighted_risk_score}"
    assert res.section_4_segmentation.defect_area_px > 0, "Expected non-zero defect area"
    assert res.section_4_segmentation.bounding_box != [0, 0, 0, 0], "Expected non-zero bounding box"
    print(f"[PASS] TEST 2 (Crack Upload): Decision={res.section_9_decision_engine.disposition}, Risk={res.section_8_risk_scoring.weighted_risk_score}, Area={res.section_4_segmentation.defect_area_mm2} mm²")


def test_upload_scratch_component():
    """TEST 3: Upload an industrial component with linear scratch damage."""
    pipeline = InspectionPipeline()
    
    h, w = 512, 512
    metal = np.full((h, w, 3), 170, dtype=np.uint8)
    
    # Draw fine linear scratch marks
    cv2.line(metal, (140, 200), (380, 210), (70, 70, 70), 2)
    cv2.line(metal, (140, 202), (380, 212), (240, 240, 240), 1)

    upload_b64 = create_test_image_base64(metal)
    req = InspectionRequest(image_base64=upload_b64)
    res = pipeline.execute_inspection(req)

    assert res.section_3_defect_identification.predicted_defect == "Scratch", \
        f"Expected Scratch but got {res.section_3_defect_identification.predicted_defect}"
    assert res.section_4_segmentation.defect_area_px > 0
    assert res.section_8_risk_scoring.weighted_risk_score > 0.25
    print(f"[PASS] TEST 3 (Scratch Upload): Defect={res.section_3_defect_identification.predicted_defect}, Decision={res.section_9_decision_engine.disposition}, Risk={res.section_8_risk_scoring.weighted_risk_score}")


def test_upload_surface_damage_component():
    """TEST 3b: Upload an industrial component with rough fractured surface damage."""
    pipeline = InspectionPipeline()
    
    h, w = 512, 512
    dmg_img = np.full((h, w, 3), 175, dtype=np.uint8)
    
    # Create large localized rough surface damage (broken matrix with high internal variance)
    noise = np.random.randint(20, 240, (70, 70, 3), dtype=np.uint8)
    dmg_img[200:270, 200:270] = noise

    upload_b64 = create_test_image_base64(dmg_img)
    req = InspectionRequest(image_base64=upload_b64)
    res = pipeline.execute_inspection(req)

    assert res.section_3_defect_identification.predicted_defect == "Surface Damage", \
        f"Expected Surface Damage but got {res.section_3_defect_identification.predicted_defect}"
    assert res.section_4_segmentation.defect_area_px > 0
    assert res.section_8_risk_scoring.weighted_risk_score >= 0.50
    print(f"[PASS] TEST 3b (Surface Damage Upload): Defect={res.section_3_defect_identification.predicted_defect}, Decision={res.section_9_decision_engine.disposition}, Risk={res.section_8_risk_scoring.weighted_risk_score}")


def test_upload_invalid_file():
    """TEST 4: Upload an invalid / non-image file and verify clear error."""
    # Corrupted / text file
    corrupt_bytes = b"This is not a PNG or JPEG file! Just arbitrary text data."
    
    try:
        validate_and_decode_image(corrupt_bytes)
        assert False, "Should have raised ValueError for invalid file"
    except ValueError as e:
        assert "Unable to decode image file" in str(e) or "Unsupported image format" in str(e)
        print(f"[PASS] TEST 4 (Invalid File): Raised expected ValueError: {str(e)}")

    # Empty payload
    try:
        load_image_from_base64("")
        assert False, "Should have raised ValueError for empty data"
    except ValueError as e:
        print(f"[PASS] TEST 4 (Empty Payload): Raised expected ValueError: {str(e)}")


def test_state_isolation_image_a_then_b():
    """TEST 5: Upload image A (Crack), run inspection, then upload image B (Normal)."""
    pipeline = InspectionPipeline()

    # Image A: Severe Crack (meandering fracture)
    img_a = np.full((512, 512, 3), 170, dtype=np.uint8)
    pts = np.array([[150, 100], [210, 170], [160, 230], [225, 300], [175, 390]], np.int32)
    cv2.polylines(img_a, [pts], isClosed=False, color=(20, 20, 20), thickness=4)
    cv2.polylines(img_a, [pts + 1], isClosed=False, color=(255, 255, 255), thickness=1)
    b64_a = create_test_image_base64(img_a)

    req_a = InspectionRequest(image_base64=b64_a)
    res_a = pipeline.execute_inspection(req_a)
    assert res_a.section_3_defect_identification.predicted_defect == "Crack"
    assert res_a.section_9_decision_engine.disposition == "REJECT"

    # Image B: Completely pristine normal surface
    img_b = np.full((512, 512, 3), 180, dtype=np.uint8)
    b64_b = create_test_image_base64(img_b)

    req_b = InspectionRequest(image_base64=b64_b)
    res_b = pipeline.execute_inspection(req_b)

    # Image B must be analyzed completely on its own merits
    assert res_b.section_3_defect_identification.predicted_defect == "Normal", \
        f"Residual defect detected! Expected Normal, got {res_b.section_3_defect_identification.predicted_defect}"
    assert res_b.section_9_decision_engine.disposition == "PASS"
    assert res_b.section_8_risk_scoring.weighted_risk_score <= 0.25
    assert res_b.section_4_segmentation.defect_area_px == 0
    assert res_b.inspection_id != res_a.inspection_id
    print(f"[PASS] TEST 5 (State Isolation): Image A -> REJECT, Image B -> PASS (Clean state transition)")


def test_upload_real_multi_flange_normal():
    """TEST 6: Upload real photograph of pristine industrial metal flanges (with silhouettes, bolt holes, bores)."""
    import os
    sample_path = os.path.join(os.path.dirname(__file__), "sample_flanges.jpg")
    if not os.path.exists(sample_path):
        sample_path = r"C:\Users\Hxtreme\Desktop\metal.jpg"
    
    img = cv2.imread(sample_path)
    assert img is not None, f"Could not read test image {sample_path}"
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    b64 = create_test_image_base64(img_rgb)
    
    pipeline = InspectionPipeline()
    req = InspectionRequest(image_base64=b64)
    res = pipeline.execute_inspection(req)
    
    # Must correctly recognize as pristine Normal, NOT false Crack or Scratch
    assert res.section_3_defect_identification.predicted_defect == "Normal", \
        f"False Positive on real flange! Expected Normal, got {res.section_3_defect_identification.predicted_defect}"
    assert res.section_9_decision_engine.disposition == "PASS", \
        f"Expected PASS disposition, got {res.section_9_decision_engine.disposition}"
    assert res.section_8_risk_scoring.weighted_risk_score <= 0.25, \
        f"Expected risk <= 0.25, got {res.section_8_risk_scoring.weighted_risk_score}"
    assert res.section_4_segmentation.defect_area_px == 0, \
        f"Expected 0 px defect area on pristine flange, got {res.section_4_segmentation.defect_area_px}"
    assert res.section_4_segmentation.bounding_box == [0, 0, 0, 0], \
        f"Expected [0, 0, 0, 0] bounding box, got {res.section_4_segmentation.bounding_box}"
    print(f"[PASS] TEST 6 (Real Normal Flanges Upload): Defect={res.section_3_defect_identification.predicted_defect}, Decision={res.section_9_decision_engine.disposition}, Risk={res.section_8_risk_scoring.weighted_risk_score}, Conf={res.section_3_defect_identification.clip_confidence}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Running SmartVisionOC Verification Test Suite")
    print("=" * 60)
    test_preset_crack_specimen()
    test_preset_normal_specimen()
    test_upload_normal_component()
    test_upload_crack_component()
    test_upload_scratch_component()
    test_upload_surface_damage_component()
    test_upload_invalid_file()
    test_state_isolation_image_a_then_b()
    test_upload_real_multi_flange_normal()
    print("=" * 60)
    print(" ALL 7 TEST CASES AND PRESET SUITES PASSED SUCCESSFULLY!")
    print("=" * 60)
