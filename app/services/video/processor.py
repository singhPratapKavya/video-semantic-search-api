import cv2
import numpy as np
from pathlib import Path
import torch
import logging
from typing import List, Dict, Any, Generator, Tuple, Optional
import time
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

# Import settings object
from app.core.config import settings

from app.services.search.faiss_service import FAISSService
from app.utils.error_handling import VideoProcessingError, handle_video_processing_errors
from app.utils.file_ops import ensure_directory
from app.utils.image_ops import save_frame, extract_frames
from app.utils.model_utils import (
    load_clip_model,
    generate_image_embedding
)
from app.utils.duplicate_detector import DuplicateDetector

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Process videos to extract frames, detect duplicates, and store embeddings."""
    
    @handle_video_processing_errors("Video processor initialization failed")
    def __init__(self,
                 clip_model: CLIPModel,
                 clip_processor: CLIPProcessor,
                 faiss_service: FAISSService
                 ):
        """Initialize the video processor with dependencies."""
        logger.info("Initializing Video Processor...")
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.faiss_service = faiss_service
        self.clip_model_name = settings.CLIP_MODEL_NAME
        self.device = settings.CLIP_DEVICE
        self.frames_dir = settings.FRAMES_DIR
        self.batch_size = settings.BATCH_SIZE
        self.hash_threshold = settings.HASH_THRESHOLD
        self.window_size = settings.WINDOW_SIZE
        self.faiss_index_dir = settings.FAISS_INDEX_DIR
        self.frame_extraction_fps = settings.FRAME_EXTRACTION_FPS
        self.allowed_extensions = settings.ALLOWED_VIDEO_EXTENSIONS
        self.video_dir = settings.VIDEO_DIR

        logger.info(f"Using device: {self.device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        logger.info("Models being used:")
        logger.info(f"• CLIP Model: {self.clip_model_name}")
        logger.info("Processing Steps:")
        logger.info("1. Frame Extraction")
        logger.info("2. Embedding Generation")
        logger.info("3. Duplicate Detection (using DuplicateDetector)")
        logger.info("4. FAISS Storage")
        
        try:
            # Initialize Duplicate Detector using settings
            self.duplicate_detector = DuplicateDetector(
                clip_model=self.clip_model,
                clip_processor=self.clip_processor,
                hash_threshold=self.hash_threshold,
                window_size=self.window_size
            )
            
            # Create directories
            ensure_directory(self.frames_dir)
            
            # Initialize statistics
            self.processed_videos = set()
            self.total_frames_extracted = 0
            self.total_duplicate_frames = 0
            self.total_frames_stored = 0
            
            logger.info("Video Processor initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing VideoProcessor: {str(e)}", exc_info=True)
            if "device type" in str(e):
                logger.error("CUDA device mismatch detected. Try running with --use-cpu flag.")
            raise
    
    def process_frame(self, frame: np.ndarray, timestamp: float, video_name: str, frame_index: int) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Process a single frame: generate embedding, check for duplicates, save if unique.
        
        Args:
            frame: The frame image
            timestamp: The timestamp of the frame
            video_name: Name of the video
            frame_index: Index of the frame
            
        Returns:
            Tuple (metadata: Optional[Dict], embedding: Optional[np.ndarray]).
            Both are None if duplicate, otherwise contain metadata and the embedding.
        """
        try:
            # Generate embedding
            embedding = generate_image_embedding(
                frame,
                self.clip_model,
                self.clip_processor
            )

            # Check if duplicate using the detector, which now returns embedding if not duplicate
            is_dup, embedding_if_not_dup = self.duplicate_detector.is_duplicate(frame, embedding)
            if is_dup:
                return None, None # Return None for both if duplicate

            # Save frame to disk using the utility function
            frame_path = self.frames_dir / f"{video_name}_frame_{frame_index:05d}.jpg"
            save_frame(frame, frame_path)
            
            # Create metadata - store path relative to static dir eventually
            # For now, storing relative to frames_dir
            relative_frame_path = frame_path.relative_to(self.frames_dir)
            metadata = {
                # Store path relative to frames dir for now
                "frame_path": str(relative_frame_path),
                "video_name": video_name,
                "timestamp": float(timestamp),
            }
            
            # Return metadata and the non-duplicate embedding
            return metadata, embedding_if_not_dup
        except Exception as e:
            logger.error(f"Error processing frame {frame_index} from {video_name}: {str(e)}", exc_info=True)
            return None, None
    
    def process_video(self, video_path: str) -> None:
        """
        Process a single video: extract frames, generate embeddings, detect duplicates, store in FAISS.
        """
        start_time = time.time()
        video_path_obj = Path(video_path)
        video_name = video_path_obj.name
        
        # Skip if already processed
        if video_name in self.processed_videos:
            logger.warning(f"Video {video_name} already processed, skipping")
            return
            
        logger.info(f"Processing video: {video_name}")
        self.duplicate_detector.clear()
        frames_extracted = 0
        frames_stored = 0
        duplicates_detected = 0
        frame_metadata_batch = []
        embeddings_batch = []

        try:
            # Use FPS from settings
            extracted_frames, timestamps = extract_frames(video_path_obj, fps=self.frame_extraction_fps)
            logger.info(f"Extracted {len(extracted_frames)} potential frames.")

            # TODO: Make tqdm optional based on config/log level
            for i, (frame, timestamp) in enumerate(tqdm(zip(extracted_frames, timestamps), total=len(extracted_frames), desc=f"Processing {video_name}")):
                frames_extracted += 1
                metadata, embedding = self.process_frame(frame, timestamp, video_name, i)

                if metadata is not None and embedding is not None:
                    frames_stored += 1
                    embeddings_batch.append(embedding)
                    frame_metadata_batch.append(metadata)

                    if len(embeddings_batch) >= self.batch_size:
                        self.faiss_service.store_embeddings(embeddings_batch, frame_metadata_batch)
                        embeddings_batch = []
                        frame_metadata_batch = []
                else:
                    duplicates_detected += 1

            if embeddings_batch:
                self.faiss_service.store_embeddings(embeddings_batch, frame_metadata_batch)

            end_time = time.time()
            processing_time = end_time - start_time
            
            # Update overall statistics
            self.processed_videos.add(video_name)
            self.total_frames_extracted += frames_extracted
            self.total_duplicate_frames += duplicates_detected
            self.total_frames_stored += frames_stored
            
            logger.info(f"Finished processing {video_name} in {processing_time:.2f} seconds")
            logger.info(f"  Frames extracted: {frames_extracted}")
            logger.info(f"  Duplicates detected: {duplicates_detected}")
            logger.info(f"  Frames stored: {frames_stored}")
        except ValueError as e:
            logger.error(f"Skipping video {video_name} due to error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error processing video {video_name}: {e}", exc_info=True)

    def process_all_videos(self, video_dir_override: Optional[Path] = None, allowed_extensions_override: Optional[List[str]] = None) -> None:
        """Process all videos in the specified video directory."""
        process_dir = video_dir_override if video_dir_override else self.video_dir
        allowed_ext = allowed_extensions_override if allowed_extensions_override else self.allowed_extensions

        logger.info(f"Searching for videos in: {process_dir} with extensions {allowed_ext}")
        # Use glob with '**/*' for recursive search
        video_files = [f for f in process_dir.glob("**/*") if f.is_file() and f.suffix.lower() in allowed_ext]

        if not video_files:
            logger.warning(f"No videos found in {process_dir} with specified extensions.")
            return
            
        logger.info(f"Found {len(video_files)} videos to process")
        
        for video_path in video_files:
            try:
                self.process_video(str(video_path))
            except Exception as e:
                logger.error(f"Failed processing {video_path.name}: {e}", exc_info=True)
                continue
        
        # Log final summary
        logger.info("\nProcessing Summary:")
        logger.info(f"Total frames extracted: {self.total_frames_extracted}")
        logger.info(f"Duplicate frames removed: {self.total_duplicate_frames}")
        logger.info(f"Total frames stored: {self.total_frames_stored}")

    def finalize_processing(self):
        """Save the FAISS index after processing all videos."""
        # Check if there's anything to save
        if self.total_frames_stored > 0:
            logger.info("Finalizing processing and saving FAISS index...")
            self.faiss_service.save_index()
            logger.info("FAISS index saved.")
        else:
            logger.warning("Skipping saving FAISS index as no new frames were stored.")