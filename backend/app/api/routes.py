from fastapi import APIRouter
from app.api.endpoints import detection, health, visualization, video_detection

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(detection.router, tags=["detection"])
api_router.include_router(visualization.router, tags=["visualization"])
api_router.include_router(video_detection.router, tags=["video-detection"])
