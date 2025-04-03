#!/usr/bin/env python

import sys
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
import uvicorn
import torch # For device check

# Configuration
from app.core.config import settings, ensure_configured_dirs

# Services and Utilities (Import types needed for getters first)
from app.services.search.faiss_service import FAISSService, FAISSServiceError
from app.services.video.frame_processor import FrameProcessor
from app.utils.model_utils import load_clip_model
from app.utils.error_handling import VideoProcessingError

# Configure logging (move comprehensive config elsewhere if needed)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# --- Define State and Dependency Getters FIRST --- #
app_state = {}

def get_faiss_service() -> FAISSService:
    service = app_state.get("faiss_service")
    if not service:
        raise HTTPException(status_code=503, detail="FAISS Service not available.")
    return service

def get_frame_processor() -> FrameProcessor:
    service = app_state.get("frame_processor")
    if not service:
        raise HTTPException(status_code=503, detail="Frame Processor Service not available.")
    return service

# --- Import Routers AFTER Getters are Defined --- #
from app.api.v1.endpoints import search as search_router_module

# --- Lifespan Manager --- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events: load models/services on startup."""
    logger.info("Application startup: Loading resources...")
    try:
        ensure_configured_dirs() # Ensure dirs exist on startup

        # Use device from settings for CLIP
        clip_device = settings.CLIP_DEVICE
        logger.info(f"Using device for CLIP: {clip_device}")

        # Load CLIP model
        clip_model, clip_processor = load_clip_model(settings.CLIP_MODEL_NAME, clip_device)
        app_state["clip_model"] = clip_model
        app_state["clip_processor"] = clip_processor
        logger.info(f"CLIP model '{settings.CLIP_MODEL_NAME}' loaded successfully on {clip_device}.")

        # Initialize FAISS Service
        faiss_service = FAISSService(
            index_dir=settings.FAISS_INDEX_DIR,
            clip_model_name=settings.CLIP_MODEL_NAME,
            clip_device=clip_device # Pass CLIP device here
        )
        app_state["faiss_service"] = faiss_service
        logger.info("FAISS service initialized.")

        # Initialize FrameProcessor
        frame_processor = FrameProcessor(
            faiss_service=faiss_service,
            base_url=settings.BASE_URL
        )
        app_state["frame_processor"] = frame_processor
        logger.info("Frame processor initialized.")

    except Exception as e:
        logger.exception("Fatal error during application resource initialization.")
        # Indicate failure, but let hosting environment decide on restarts/exit
        # sys.exit(1) # Avoid hard exit if possible
        raise RuntimeError("Application startup failed.") from e

    yield # Application runs here

    # Cleanup happens after yield (if needed)
    logger.info("Application shutdown: Cleaning up resources...")
    app_state.clear()

# Create FastAPI app with lifespan manager
app = FastAPI(
    title="Video Frame Search API",
    description="API for searching video frames using CLIP embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# --- Custom OpenAPI Schema to Remove 422 Responses --- #
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Modify schema to replace 422 responses with 400
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            if "responses" in openapi_schema["paths"][path][method]:
                responses = openapi_schema["paths"][path][method]["responses"]
                # If endpoint has special flag or we want to globally remove 422 validation errors
                if "422" in responses:
                    # Copy 422 response details to 400 if not already present
                    if "400" not in responses:
                        responses["400"] = responses["422"]
                        responses["400"]["description"] = "Bad Request - Invalid input parameters"
                    # Remove the 422 response
                    del responses["422"]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# --- Exception Handlers --- #

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,  # Changed from 422 to 400
        content={"detail": exc.errors()},
    )

@app.exception_handler(FAISSServiceError)
async def faiss_service_exception_handler(request: Request, exc: FAISSServiceError):
    logger.error(f"FAISS Service error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=503, # Service Unavailable
        content={"message": f"Search index operation failed: {exc}"},
    )

@app.exception_handler(VideoProcessingError)
async def video_processing_exception_handler(request: Request, exc: VideoProcessingError):
    logger.error(f"Video Processing error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, # Internal Server Error (adjust if needed)
        content={"message": f"Video processing failed: {exc}"},
    )

@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    logger.warning(f"Invalid value encountered: {exc}", exc_info=True)
    return JSONResponse(
        status_code=400, # Bad Request
        content={"message": f"Invalid input or value: {exc}"},
    )

# Generic handler for unexpected errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, # Internal Server Error
        content={"message": "An unexpected internal server error occurred."},
    )

# --- Middleware --- #

# Add CORS middleware using settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# --- Static Files --- #

# Mount static files directory using settings
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

# --- Routers --- #

# Include API router with dependency injection
app.include_router(
    search_router_module.router,
    prefix="/api/v1",
    tags=["search"],
    # Re-add dependencies argument if needed, but usually handled by Depends in endpoints
    # dependencies=[Depends(get_frame_processor)]
)

# --- API Root --- #
@app.get("/", 
    description="Welcome endpoint providing basic information about the Video Frame Search API."
)
async def read_root():
    return {"message": f"Welcome to the Video Frame Search API (Device: {settings.CLIP_DEVICE})"}

# --- Health Check --- #
@app.get("/health", 
    description="Health check endpoint to verify API services are running properly."
)
async def health_check():
    # Revert health check to only check essential services
    if ("faiss_service" in app_state and
        "frame_processor" in app_state):
        return {"status": "ok"}
    else:
        missing = [s for s in ["faiss_service", "frame_processor"] if s not in app_state]
        raise HTTPException(status_code=503, detail=f"Service not ready. Missing: {missing}")

# --- Main Execution --- #

if __name__ == "__main__":
    # Use settings for host/port
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT} using {settings.CLIP_DEVICE} for main processing")
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level=logging.INFO
    ) 