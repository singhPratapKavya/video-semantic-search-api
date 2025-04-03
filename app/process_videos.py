#!/usr/bin/env python

import argparse
import logging
import sys
from pathlib import Path
import torch # Needed for device check
from typing import Tuple
from transformers import CLIPModel, CLIPProcessor # Import types used in Tuple

# Remove sys.path manipulation - Run as a module or set PYTHONPATH
# project_root = str(Path(__file__).parent.parent)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# Import necessary components
from app.services.video.processor import VideoProcessor
from app.services.search.faiss_service import FAISSService
from app.utils.error_handling import VideoProcessingError, FAISSServiceError
from app.utils.model_utils import load_clip_model
# Import the centralized settings object and dir creator
from app.core.config import settings, ensure_configured_dirs

# Configure logging (basic setup)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Process videos to create frame embeddings')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--use-cpu', action='store_true',
                        help=f'Force CPU usage even if CUDA is available (uses settings.FORCE_CPU={settings.FORCE_CPU} if not set)')
    # Use settings for defaults
    parser.add_argument('--video-dir', type=str, default=str(settings.VIDEO_DIR),
                        help=f'Directory containing videos to process (default: {settings.VIDEO_DIR})')
    parser.add_argument('--extensions', nargs='+', default=settings.ALLOWED_VIDEO_EXTENSIONS,
                        help=f'List of allowed video extensions (default: {settings.ALLOWED_VIDEO_EXTENSIONS})')

    return parser.parse_args()

def setup_dependencies(device_override: str) -> Tuple[CLIPModel, CLIPProcessor, FAISSService]:
    """Load models and initialize services based on the chosen device."""
    logger.info(f"Setting up dependencies for device: {device_override}")
    try:
        # Load CLIP model using settings name and overridden device
        clip_model, clip_processor = load_clip_model(settings.CLIP_MODEL_NAME, device_override)

        # Initialize FAISS Service using settings and overridden device
        faiss_service = FAISSService(
            index_dir=settings.FAISS_INDEX_DIR,
            clip_model_name=settings.CLIP_MODEL_NAME,
            clip_device=device_override
        )
        return clip_model, clip_processor, faiss_service
    except Exception as e:
        logger.error(f"Failed to setup dependencies: {str(e)}", exc_info=True)
        raise

def run_processing(args, device_to_use: str) -> int:
    """Runs the video processing workflow with the specified device."""
    try:
        clip_model, clip_processor, faiss_service = setup_dependencies(device_to_use)

        # Initialize Video Processor with only the required dependencies
        processor = VideoProcessor(
            clip_model=clip_model,
            clip_processor=clip_processor,
            faiss_service=faiss_service
        )

        # Handle video directory override for process_all_videos
        arg_video_dir = Path(args.video_dir).resolve()
        # Use allowed extensions from args
        allowed_extensions = args.extensions

        # Pass overrides to process_all_videos
        processor.process_all_videos(
            video_dir_override=arg_video_dir,
            allowed_extensions_override=allowed_extensions
        )

        # Finalize (Save index)
        processor.finalize_processing()

        logger.info("Video processing completed successfully")
        return 0

    except (VideoProcessingError, FAISSServiceError, RuntimeError, ValueError) as e:
        logger.error(f"Processing failed on device {device_to_use}: {str(e)}", exc_info=True)
        return 1 # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing on device {device_to_use}: {str(e)}", exc_info=True)
        return 1 # Indicate failure

def main():
    args = parse_args()

    # Call dir creator function
    ensure_configured_dirs()

    # Adjust log level based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    logging.getLogger("tqdm").setLevel(logging.INFO if not args.verbose else logging.DEBUG)

    # Determine target device based on args and settings
    force_cpu_arg = args.use_cpu or settings.FORCE_CPU
    cuda_available = torch.cuda.is_available()
    target_device = "cpu" if force_cpu_arg else ("cuda" if cuda_available else "cpu")

    logger.info(f"Target device selected: {target_device} (CUDA available: {cuda_available}, Force CPU via args/settings: {force_cpu_arg})" )

    # Run processing with the target device
    exit_code = run_processing(args, target_device)

    # If CUDA was attempted (i.e., available and not forced to CPU) and failed, try CPU fallback
    if target_device == "cuda" and exit_code != 0:
        logger.warning("Processing failed on CUDA device. Attempting fallback to CPU.")
        exit_code = run_processing(args, "cpu")
        if exit_code == 0:
            logger.info("Processing completed successfully on CPU after fallback.")
        else:
            logger.error("Processing failed on CPU after fallback as well.")

    return exit_code

if __name__ == "__main__":
    sys.exit(main()) 