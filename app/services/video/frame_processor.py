from app.services.search.faiss_service import FAISSService
from typing import List, Dict, Any
import logging
import urllib.parse
from pathlib import Path # Import Path for better path handling

logger = logging.getLogger(__name__)

class FrameProcessor:
    # Accept dependencies via __init__
    def __init__(self, faiss_service: FAISSService, base_url: str):
        self.faiss_service = faiss_service # Use passed service
        self.base_url = base_url # Use passed base_url

    def search_frames(self, query: str, top_k: int) -> Dict[str, List[Dict[str, Any]]]:
        """Search for frames matching the query text."""
        try:
            # Get search results from FAISS service using single query
            search_response = self.faiss_service.search(query, top_k)
            
            # Process the results
            processed_results = []
            for result in search_response["results"]:
                # Extract frame_path relative to FRAMES_DIR (assuming it's stored this way)
                # Need to adjust if the actual metadata key is different or path is absolute
                frame_path_metadata_key = 'frame_path' # Assuming this is the key in metadata
                relative_frame_path_str = result.get(frame_path_metadata_key)

                if not relative_frame_path_str:
                    logger.warning(f"Skipping result with missing '{frame_path_metadata_key}': {result}")
                    continue
                
                # Ensure it's treated as a relative path
                relative_frame_path = Path(relative_frame_path_str)

                # Construct path relative to the base URL's static path
                # Assume the base static path for frames is /static/frames/
                static_path = Path("static") / "frames" / relative_frame_path

                # Convert to URL path format (forward slashes)
                url_path_segment = static_path.as_posix()

                # URL encode the path segment *carefully*
                # We only want to encode the filename part usually, not the slashes
                # However, full encoding is safer if filenames might have special chars
                # Let's encode the whole segment for safety, matching previous behavior
                encoded_path_segment = urllib.parse.quote(url_path_segment)

                # Create a full URL
                # Ensure no double slashes between base_url and segment
                full_url = f"{self.base_url.rstrip('/')}/{encoded_path_segment.lstrip('/')}"

                # Create result dictionary with image_url only
                processed_result = {
                    'image_url': full_url
                }
                
                processed_results.append(processed_result)
            
            return {"results": processed_results}
            
        except Exception as e:
            # Log with traceback for single query
            logger.error(f"FrameProcessor search failed for query '{query}': {str(e)}", exc_info=True)
            raise 