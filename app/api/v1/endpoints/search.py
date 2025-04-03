import logging
from pathlib import Path
from typing import List, Dict, Any

# Import Depends for dependency injection
from fastapi import APIRouter, HTTPException, Request, Query, Depends, status
from fastapi.responses import FileResponse

from app.core.config import settings
from app.services.video.frame_processor import FrameProcessor
from app.api.v1.models.models import ApiResponse, ApiResultItem
# Import only the necessary dependency getter
from app.main import get_frame_processor

logger = logging.getLogger(__name__)

router = APIRouter()

# Moved frame serving endpoint under API
@router.get("/frames/{frame_path:path}", include_in_schema=False)
async def get_frame(frame_path: str) -> FileResponse:
    """Serve a frame image by its relative path within the static/frames directory."""
    # Use path from settings
    base_static_frames_dir = (settings.STATIC_DIR / "frames").resolve()
    expected_path = base_static_frames_dir / frame_path

    try:
        # Resolve the path to prevent directory traversal
        full_path = expected_path.resolve()

        # Security check: Ensure the resolved path is within the intended directory
        if not str(full_path).startswith(str(base_static_frames_dir)):
             logger.warning(f"Attempted access outside of frames directory: {frame_path}")
             raise FileNotFoundError("Invalid path")

        if full_path.exists() and full_path.is_file():
            return FileResponse(str(full_path))
        else:
            raise FileNotFoundError("Frame file not found")
    except FileNotFoundError as e:
        logger.warning(f"Frame not found: {frame_path} ({e})")
        raise HTTPException(status_code=404, detail="Frame not found")
    except Exception as e:
        logger.error(f"Error serving frame {frame_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error serving frame")

@router.get(
    "/search", 
    response_model=ApiResponse,
    responses={
        400: {
            "description": "Bad Request - Invalid input parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": [
                            {
                                "type": "string_too_short",
                                "loc": ["query", "query"],
                                "msg": "String should have at least 1 character",
                                "input": "",
                                "ctx": {"min_length": 1}
                            }
                        ]
                    }
                }
            }
        },
        200: {
            "description": "Successful response with search results"
        },
        503: {
            "description": "Service Unavailable"
        }
    },
    openapi_extra={"x-no-422": True}  # Custom hint to remove 422 from schema
)
def search(
    query: str = Query(..., min_length=1, description="Text query to search for in video frames"),
    top_k: int = Query(default=settings.DEFAULT_TOP_K,
                       ge=1, le=settings.MAX_TOP_K,
                       description="Number of results to return"),
    frame_processor: FrameProcessor = Depends(get_frame_processor)
) -> ApiResponse:
    """Search for frames matching the query text."""
    
    logger.info(f"Processing search query: '{query}' with top_k={top_k}")
    search_results = frame_processor.search_frames(query, top_k=top_k)
    
    # Return results
    return ApiResponse(results=search_results["results"]) 