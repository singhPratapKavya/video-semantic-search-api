import torch
from pathlib import Path
# Use BaseSettings for environment variable loading
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Tuple, Optional
import os
import logging # For ensure_configured_dirs

logger = logging.getLogger(__name__)

# Define a base directory using environment variable or default
# This allows flexibility in deployment (e.g., in containers)
PROJECT_ROOT_ENV = os.getenv("PROJECT_ROOT")
BASE_DIR = Path(PROJECT_ROOT_ENV).resolve() if PROJECT_ROOT_ENV else Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """Application Configuration using Pydantic BaseSettings."""
    # Load from .env file first, then environment variables. Ignore extras.
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # --- Base Paths ---
    # Allow overriding via environment variables if needed, default relative to BASE_DIR
    # Example: Set DATA_DIR=/path/to/data in your environment
    APP_DIR: Path = BASE_DIR / "app"
    DATA_DIR: Path = BASE_DIR / "data"
    LOGS_DIR: Path = BASE_DIR / "logs"
    STATIC_DIR: Path = APP_DIR / "static"
    TEMPLATES_DIR: Path = APP_DIR / "templates"
    FRAMES_DIR: Path = STATIC_DIR / "frames"
    VIDEO_DIR: Path = DATA_DIR / "videos"
    FAISS_INDEX_DIR: Path = DATA_DIR / "faiss_index"

    # --- Model Configuration ---
    CLIP_MODEL_NAME: str = "openai/clip-vit-large-patch14"
    EMBEDDING_DIM: int = 768 # Tied to CLIP_MODEL_NAME, update if model changes

    # --- Device Configuration ---
    FORCE_CPU: bool = False
    # CLIP_DEVICE will be determined dynamically after loading other settings

    # --- Frame Processing ---
    FRAME_EXTRACTION_FPS: float = 10.0
    BATCH_SIZE: int = 32

    # --- Duplicate Detection ---
    SIMILARITY_THRESHOLD: float = 0.95  # For CLIP embedding similarity
    HASH_THRESHOLD: int = 5            # Max difference for phash
    WINDOW_SIZE: int = 10              # Recent embeddings check window
    LSH_BITS: int = 128                # FAISS LSH bits (if LSH is used)

    # --- Video Processing Script ---
    ALLOWED_VIDEO_EXTENSIONS: List[str] = ['.mp4', '.avi', '.mov']

    # --- API Configuration ---
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    # BASE_URL will be determined dynamically
    # ALLOWED_HOSTS should be set restrictively in production via env var
    # Example: ALLOWED_HOSTS='["https://yourdomain.com", "https://www.yourdomain.com"]'
    ALLOWED_HOSTS: List[str] = ["*"] # Default allows all, CHANGE FOR PROD

    # --- API Query Defaults ---
    DEFAULT_TOP_K: int = 4
    MAX_TOP_K: int = 10

    # --- Dynamic Attributes (Set after loading) ---
    CLIP_DEVICE: str = "cpu" # Initialize default
    BASE_URL: str = ""       # Initialize default

    # Pydantic v2 way to run logic after validation/loading
    def __init__(self, **values):
        super().__init__(**values)
        # Determine device
        if torch.cuda.is_available() and not self.FORCE_CPU:
            self.CLIP_DEVICE = "cuda"
        else:
            self.CLIP_DEVICE = "cpu"
        # Set Base URL
        self.BASE_URL = f"http://{self.API_HOST}:{self.API_PORT}"

# Instantiate settings - This single instance will be imported elsewhere
settings = Settings()

# Utility function to ensure directories exist (call this on app startup)
def ensure_configured_dirs():
    dirs_to_ensure = [
        settings.LOGS_DIR,
        settings.FRAMES_DIR,
        settings.VIDEO_DIR,
        settings.FAISS_INDEX_DIR
    ]
    logger.info("Ensuring configured directories exist...")
    for dir_path in dirs_to_ensure:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directory ensured: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to create directory {dir_path}: {e}", exc_info=True)
            # Decide if this is fatal depending on the directory
            if dir_path in [settings.FAISS_INDEX_DIR, settings.FRAMES_DIR]:
                 raise RuntimeError(f"Could not create critical directory: {dir_path}") from e 