"""Machine learning model utilities."""

import torch
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

# Cache for loaded models to prevent redundant loading
_MODEL_CACHE: Dict[str, Tuple[CLIPModel, CLIPProcessor]] = {}


def load_clip_model(model_name: str, device: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Load CLIP model and processor with caching to prevent redundant loading.
    
    Args:
        model_name: Name of the CLIP model to load
        device: Device to load model on ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, processor)
    """
    cache_key = f"{model_name}_{device}"
    
    if cache_key in _MODEL_CACHE:
        logger.info(f"Using cached CLIP model: {model_name}")
        return _MODEL_CACHE[cache_key]
    
    logger.info(f"Loading CLIP model: {model_name}")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # Cache the loaded model
    _MODEL_CACHE[cache_key] = (model, processor)
    
    return model, processor


def generate_image_embedding(
    image: np.ndarray,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> np.ndarray:
    """
    Generate CLIP embedding for an image.
    
    Args:
        image: Image as numpy array (BGR format from OpenCV)
        model: CLIP model
        processor: CLIP processor
        
    Returns:
        Embedding as numpy array
    """
    # Get the actual device model is on
    model_device = next(model.parameters()).device
    
    # Convert BGR to RGB
    image_rgb = image[..., ::-1]
    
    # Prepare image for CLIP
    inputs = processor(
        images=image_rgb,
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to the same device as the model
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(model_device)
    
    # Generate embedding
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        
    # Normalize and convert to numpy
    embedding = image_features.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def generate_text_embedding(
    text: str,
    model: CLIPModel,
    processor: CLIPProcessor,
) -> np.ndarray:
    """
    Generate CLIP embedding for text.
    
    Args:
        text: Text to embed
        model: CLIP model
        processor: CLIP processor
        
    Returns:
        Embedding as numpy array
    """
    # Get the actual device model is on
    model_device = next(model.parameters()).device
    
    # Prepare text for CLIP
    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True
    )
    
    # Move inputs to the same device as the model
    for key in inputs:
        if torch.is_tensor(inputs[key]):
            inputs[key] = inputs[key].to(model_device)
    
    # Generate embedding
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        
    # Normalize and convert to numpy
    embedding = text_features.cpu().numpy()[0]
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding 