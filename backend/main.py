"""
SmartVisionOC FastAPI Application Server
Provides RESTful endpoints for 10-stage zero-shot visual inspection and serves the frontend dashboard.
"""
import os
import io
import base64
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from backend.config import settings
from backend.schemas import InspectionRequest, InspectionResponse
from backend.services.pipeline import InspectionPipeline
from backend.services.mock_data import SPECIMEN_CATALOG, generate_procedural_flange, ndarray_to_base64
from backend.inference.image_loader import validate_and_decode_image

app = FastAPI(
    title=settings.APP_TITLE,
    description="Explainable Multi-Modal AI Framework for Zero-Shot Industrial Quality Inspection",
    version=settings.VERSION
)

# Enable CORS for local development and web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = InspectionPipeline()

# Cache generated specimen thumbnails
_cached_specimens = None

def get_specimens_with_thumbnails():
    global _cached_specimens
    if _cached_specimens is not None:
        return _cached_specimens
        
    spec_list = []
    for spec_id, data in SPECIMEN_CATALOG.items():
        try:
            # Generate small thumbnail
            img_rgb, _ = generate_procedural_flange(spec_id, 256, 256)
            thumb_b64 = ndarray_to_base64(img_rgb, "JPEG")
        except Exception:
            thumb_b64 = None
            
        spec_list.append({
            "id": data["id"],
            "name": data["name"],
            "defect_type": data["defect_type"],
            "ground_truth_severity": data["ground_truth_severity"],
            "description": data["description"],
            "thumbnail_base64": thumb_b64,
            "params": {
                "center": data["defect_params"]["center"],
                "delta_z_mm": data["defect_params"]["delta_z_mm"],
                "location": data["defect_params"]["location_name"]
            }
        })
    _cached_specimens = spec_list
    return _cached_specimens

@app.get("/api/v1/status")
def get_system_status():
    """System health check and modular engine status."""
    return {
        "status": "OPERATIONAL",
        "app_name": settings.APP_NAME,
        "version": settings.VERSION,
        "inference_mode": "AI_INFERENCE_READY",
        "disclaimer": settings.DISCLAIMER,
        "models": pipeline.engine.get_models_status()
    }

@app.get("/api/v1/specimens")
def list_specimens():
    """Returns the catalog of deterministic industrial flange test specimens."""
    return get_specimens_with_thumbnails()

@app.post("/api/v1/upload")
async def upload_image_file(file: UploadFile = File(...)):
    """
    Direct image file upload endpoint.
    Validates file format and size (up to 25MB), decodes and normalizes to RGB.
    """
    try:
        raw_bytes = await file.read()
        meta = validate_and_decode_image(raw_bytes)
        return {
            "status": "SUCCESS",
            "filename": file.filename,
            "format": meta["format"],
            "dimensions": meta["dimensions"],
            "file_size_kb": meta["file_size_kb"],
            "sha256_hash": meta["sha256_hash"],
            "image_base64": meta["clean_base64"]
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload validation failed: {str(e)}")

@app.post("/api/v1/inspect", response_model=InspectionResponse)
def run_inspection(req: InspectionRequest):
    """Executes the full 10-stage zero-shot visual inspection pipeline."""
    try:
        response = pipeline.execute_inspection(req)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline execution error: {str(e)}")

# Mount static frontend
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(frontend_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "SmartVisionOC Backend API Operational. Frontend index not found."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
