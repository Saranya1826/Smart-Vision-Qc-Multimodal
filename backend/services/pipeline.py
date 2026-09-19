"""
10-Stage Multi-Modal Inspection Pipeline Service
Wraps and exposes the modular backend/inference pipeline.
"""
from backend.schemas import InspectionRequest, InspectionResponse
from backend.inference.pipeline import InferencePipeline


class InspectionPipeline:
    def __init__(self):
        self.engine = InferencePipeline()
        self.clip_detector = self.engine.clip_detector
        self.segmenter = self.engine.segmenter
        self.depth_estimator = self.engine.depth_estimator

    def execute_inspection(self, req: InspectionRequest) -> InspectionResponse:
        return self.engine.execute_inspection(req)
