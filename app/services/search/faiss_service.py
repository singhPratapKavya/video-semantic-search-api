import faiss
import numpy as np
import json
from pathlib import Path
import logging
from typing import List, Dict, Any
from app.core.config import settings
from app.utils.error_handling import FAISSServiceError, handle_faiss_errors
from app.utils.model_utils import generate_text_embedding, load_clip_model
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

class FAISSService:
    def __init__(self,
                 index_dir: Path,
                 clip_model_name: str,
                 clip_device: str):
        try:
            logger.info("Initializing FAISSService...")
            self.index_path = index_dir
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.embedding_dim = settings.EMBEDDING_DIM
            
            # Load CLIP model for text embeddings
            self.clip_model, self.clip_processor = load_clip_model(clip_model_name, clip_device)
            
            self.index = self._initialize_index()
            self.metadata = self._load_metadata()
            logger.info(f"FAISSService initialized with {self.index.ntotal} vectors in index.")
        except Exception as e:
            logger.error(f"Failed to initialize FAISSService: {str(e)}", exc_info=True)
            raise FAISSServiceError(f"Initialization failed: {str(e)}")

    @handle_faiss_errors("Failed to initialize FAISS index")
    def _initialize_index(self) -> faiss.Index:
        """Initialize or load the FAISS index."""
        index_file = self.index_path / "index.faiss"
        if index_file.exists():
            logger.info(f"Loading existing FAISS index from {index_file}")
            return faiss.read_index(str(index_file))
        else:
            logger.info(f"Creating new FAISS index (IndexFlatIP) with dim {self.embedding_dim}")
            return faiss.IndexFlatIP(self.embedding_dim)

    @handle_faiss_errors("Failed to load metadata")
    def _load_metadata(self) -> List[Dict[str, Any]]:
        """Load metadata from file or initialize empty list."""
        metadata_path = self.index_path / "metadata.json"
        if metadata_path.exists():
            logger.info(f"Loading existing metadata from {metadata_path}")
            try:
                with open(metadata_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                         logger.info(f"Loaded {len(data)} metadata entries.")
                         return data
                    else:
                        logger.warning(f"Metadata file {metadata_path} did not contain a list. Initializing empty metadata.")
                        return []
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from metadata file {metadata_path}. Initializing empty metadata.", exc_info=True)
                return []
        logger.info("Metadata file not found. Initializing new metadata")
        return []

    @handle_faiss_errors("Failed to save FAISS index and metadata")
    def save_index(self) -> None:
        """Save the current FAISS index and metadata to disk."""
        index_file = self.index_path / "index.faiss"
        metadata_path = self.index_path / "metadata.json"

        logger.info(f"Saving FAISS index to {index_file} ({self.index.ntotal} vectors)")
        faiss.write_index(self.index, str(index_file))

        logger.info(f"Saving metadata to {metadata_path} ({len(self.metadata)} entries)")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2) # Add indent for readability

        logger.info("Index and metadata saved successfully")

    @handle_faiss_errors("Failed to store embeddings")
    def store_embeddings(self, embeddings: List[np.ndarray], metadata: List[Dict[str, Any]]) -> None:
        """Store embeddings and metadata in the FAISS index (in memory)."""
        if not embeddings:
            logger.warning("store_embeddings called with empty embeddings list.")
            return

        # Ensure embeddings are normalized (should be already from model_utils)
        embeddings_array = np.array(embeddings).astype('float32')

        # Basic shape validation
        if len(embeddings_array.shape) != 2 or embeddings_array.shape[1] != self.embedding_dim:
             logger.error(f"Invalid embeddings shape: {embeddings_array.shape}. Expected (*, {self.embedding_dim})")
             raise ValueError(f"Invalid embeddings shape: {embeddings_array.shape}")

        self.index.add(embeddings_array)
        self.metadata.extend(metadata)

        logger.debug(f"Added {len(embeddings)} embeddings to index (current total: {self.index.ntotal})")

    @handle_faiss_errors("Failed to search embeddings")
    def search(self, query: str, top_k: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        """Search for similar frames using FAISS index."""
        if self.index.ntotal == 0:
            logger.warning("Search called on an empty FAISS index.")
            return {"results": []}

        # Revert to single query embedding generation
        try:
            query_embedding = generate_text_embedding(query, self.clip_model, self.clip_processor)
        except Exception as e:
            logger.error(f"Failed to generate embedding for query '{query}': {e}", exc_info=True)
            return {"results": []} # Return empty on embedding failure
        
        query_array = np.array([query_embedding], dtype=np.float32)
        # Ensure the embedding is normalized (generate_text_embedding should already do this)
        # query_array = query_array / np.linalg.norm(query_array)

        # Search in FAISS index (IP = Inner Product/Cosine Similarity for normalized vectors)
        distances, indices = self.index.search(query_array, top_k)

        # Get metadata for matched frames
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.metadata): # Check bounds
                frame_info = self.metadata[idx].copy() # Get a copy
                # Add similarity score to the result
                frame_info['similarity'] = float(distance)
                results.append(frame_info)
            else:
                logger.warning(f"Search returned invalid index {idx}, metadata length {len(self.metadata)}")

        logger.info(f"Search for query '{query}' completed with {len(results)} results")
        return {"results": results} 