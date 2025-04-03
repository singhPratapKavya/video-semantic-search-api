import numpy as np
import cv2
from PIL import Image
import imagehash
from typing import List, Tuple, Optional, Dict
import faiss
from transformers import CLIPModel, CLIPProcessor
from app.core.config import settings

class DuplicateDetector:
    # Constants
    RECENT_HASH_CHECK_COUNT = 5
    LSH_SEARCH_K = 10
    LSH_UPDATE_FREQUENCY = 100
    LSH_BITS = 8
    EMBEDDING_DIM = 768
    
    def __init__(self,
                 clip_model: CLIPModel,
                 clip_processor: CLIPProcessor,
                 hash_threshold: int,
                 window_size: int):
        """
        Initialize the duplicate detector.
        
        Args:
            clip_model: CLIP model for generating embeddings
            clip_processor: CLIP processor for preprocessing images
            hash_threshold: Maximum difference for perceptual hashes to be considered similar
            window_size: Number of recent frame embeddings to check directly
        """
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.hash_threshold = hash_threshold
        self.window_size = window_size
        self.recent_frames: List[Dict[str, imagehash.ImageHash]] = []
        self.frame_embeddings: List[np.ndarray] = []
        self.lsh_index: Optional[faiss.IndexLSH] = None

    def compute_image_hash(self, image: np.ndarray) -> imagehash.ImageHash:
        """Compute perceptual hash for an image."""
        # Convert from BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL image
        pil_image = Image.fromarray(rgb_image)
        # Compute the hash
        return imagehash.phash(pil_image)

    def hash_difference(self, hash1: imagehash.ImageHash, hash2: imagehash.ImageHash) -> int:
        """Calculate the difference between two perceptual hashes."""
        # The hash objects from imagehash have a built-in comparison operator
        return hash1 - hash2

    def is_similar_to_any(self, embedding: np.ndarray, existing_embeddings: List[np.ndarray]) -> bool:
        """Check if an embedding is similar to any in the list."""
        if not existing_embeddings:
            return False
            
        # Convert to numpy array if not already
        existing_embeddings_array = np.array(existing_embeddings)
        
        # Calculate cosine similarities
        # For normalized vectors, dot product equals cosine similarity
        similarities = np.dot(existing_embeddings_array, embedding)
        
        # Scale similarity to better range (0.5-1.0 → 0.0-1.0)
        similarities = (similarities + 1) / 2
        
        return bool(np.any(similarities > settings.SIMILARITY_THRESHOLD))

    def update_lsh_index(self):
        """Update LSH index for efficient similarity search."""
        if not self.frame_embeddings:
            return
            
        # Convert embeddings to float32 for FAISS
        embeddings_array = np.array(self.frame_embeddings).astype('float32')
        
        # Create LSH index
        self.lsh_index = faiss.IndexLSH(self.EMBEDDING_DIM, self.LSH_BITS)  # Use constants
        self.lsh_index.add(embeddings_array)

    def is_duplicate(self, frame: np.ndarray, frame_embedding: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Check if a frame is a duplicate using two-stage detection.
        
        Args:
            frame: The frame to check
            frame_embedding: Pre-generated CLIP embedding for the frame
            
        Returns:
            Tuple (is_duplicate: bool, embedding: Optional[np.ndarray]).
            The embedding is returned only if is_duplicate is False.
        """
        # Stage 1: Quick perceptual hash check with recent frames
        frame_hash = self.compute_image_hash(frame)
        for recent_frame in self.recent_frames[-self.RECENT_HASH_CHECK_COUNT:]:
            if self.hash_difference(frame_hash, recent_frame['hash']) < self.hash_threshold:
                return True, None

        # Stage 2: CLIP embedding comparison
        # Check against recent frames first (temporal locality)
        recent_embeddings = self.frame_embeddings[-self.window_size:]
        if self.is_similar_to_any(frame_embedding, recent_embeddings):
            return True, None

        # If not found in recent frames, use LSH for approximate search
        if len(self.frame_embeddings) > self.window_size:
            if self.lsh_index is None or len(self.frame_embeddings) % self.LSH_UPDATE_FREQUENCY == 0:
                self.update_lsh_index()

            if self.lsh_index:
                # Search using LSH index
                distances, indices = self.lsh_index.search(
                    np.array([frame_embedding]).astype('float32'),
                    k=self.LSH_SEARCH_K
                )

                # Check potential matches
                potential_matches = [self.frame_embeddings[i] for i in indices[0] if 0 <= i < len(self.frame_embeddings)]
                if self.is_similar_to_any(frame_embedding, potential_matches):
                    return True, None

        # Not a duplicate - add to our collections
        self.recent_frames.append({'hash': frame_hash})
        self.frame_embeddings.append(frame_embedding)

        # Return False (not duplicate) and the embedding
        return False, frame_embedding

    def clear(self):
        """Clear all stored data."""
        self.recent_frames = []
        self.frame_embeddings = []
        self.lsh_index = None 