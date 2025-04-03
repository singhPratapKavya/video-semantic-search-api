# Video Frame Search API

A FastAPI-based application using CLIP embeddings and FAISS for efficient semantic search across video frames.

## Features

*   **Video Processing:** Extracts frames from video files at a configurable FPS.
*   **Embedding Generation:** Uses OpenAI's CLIP model to generate image embeddings for each frame.
*   **Duplicate Detection:** Implements perceptual hashing (phash) and embedding similarity checks to filter out near-duplicate frames, storing only unique ones.
*   **Efficient Search:** Stores embeddings in a FAISS (IndexFlatIP) index for fast cosine similarity searches.
*   **API:** Provides a REST API endpoint (`/api/v1/search`) to search for frames using text queries.
*   **Static Frame Serving:** Serves the matched frame images via a static endpoint.
*   **Configuration:** Uses Pydantic's `BaseSettings` for easy configuration via environment variables or a `.env` file.
*   **GPU Acceleration:** Supports CUDA for faster model inference and processing if available.

## Test Results

| Query | Result 1 | Result 2 | Result 3 | Result 4 |
|-------|----------|----------|----------|----------|
| "hand shaped couch" | <img src="docs/test_results/hand_couch_1.jpg" width="150" /> | <img src="docs/test_results/hand_couch_2.jpg" width="150" /> | <img src="docs/test_results/hand_couch_3.jpg" width="150" /> | <img src="docs/test_results/hand_couch_4.jpg" width="150" /> |
| "bats on a wall" | <img src="docs/test_results/bats_1.jpg" width="150" /> | <img src="docs/test_results/bats_2.jpg" width="150" /> | <img src="docs/test_results/bats_3.jpg" width="150" /> | <img src="docs/test_results/bats_4.jpg" width="150" /> |
| "ember holding phone" | <img src="docs/test_results/holding_phone_1.jpg" width="150" /> | <img src="docs/test_results/holding_phone_2.jpg" width="150" /> | <img src="docs/test_results/holding_phone_3.jpg" width="150" /> | <img src="docs/test_results/holding_phone_4.jpg" width="150" /> |

*Note: The images above show example search results from our test runs. Due to the nature of CLIP embeddings and FAISS indexing, different systems might return slightly different but semantically similar results for the same query. The quality and relevance of results should remain consistent across different runs.*

## Prerequisites

*   Python 3.8+
*   Virtual environment tool (like `venv`)
*   Optional: NVIDIA GPU with CUDA installed for faster processing.

**Performance Note:**

> Highly recommended to use a **GPU** for video processing. Frame extraction, especially with the default setting of 10 FPS, is computationally intensive and can be very slow on a CPU, particularly for longer videos. Using a CPU might necessitate lowering the FPS (e.g., via the `FRAME_EXTRACTION_FPS` setting or the `--fps` command-line argument during processing), which could risk skipping important frames. A GPU significantly speeds up this process and gives better search results.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On Unix/MacOS:
    source venv/bin/activate
    ```

3.  **Install PyTorch:**
    Choose the appropriate command based on your system (CPU or CUDA). Visit the [PyTorch website](https://pytorch.org/get-started/locally/) for the latest commands.

    *   **CPU Only:**
        ```bash
        pip install torch torchvision torchaudio
        ```
    *   **GPU (CUDA - Example for 12.1):**
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        ```
    *   **Verify GPU (Optional):**
        ```bash
        python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 
        ```

4.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: The `requirements.txt` includes `faiss-cpu`. If you have a compatible CUDA setup and want GPU-accelerated FAISS, install it separately:* `pip uninstall faiss-cpu && pip install faiss-gpu`*.*

## Configuration

The application uses `app/core/config.py` with Pydantic settings.
Default configuration values are defined in the `Settings` class.

These defaults work out-of-the-box for basic local execution. However, you can override them using one of the following methods (useful for local adjustments or production deployment):

1.  **Creating a `.env` file** in the project root directory (where `README.md` is). This file is **optional** but recommended for local development overrides.
2.  **Setting environment variables** before running the application (these take precedence over `.env` values).

**Example `.env` file (Optional):**

```dotenv
# .env

# Override the video source directory
# VIDEO_DIR=/path/to/my/videos

# Override the port the API runs on
# API_PORT=8001

# Force CPU usage even if CUDA is available
# FORCE_CPU=True

# Set allowed hosts for CORS (important for production)
# ALLOWED_HOSTS='["http://localhost:3000", "https://myfrontend.com"]'

# Adjust frame extraction FPS (default is 10.0)
# FRAME_EXTRACTION_FPS=2.0
```

Key configuration options in `app/core/config.py` include:

*   `VIDEO_DIR`: Path to the directory containing input videos.
*   `FRAME_EXTRACTION_FPS`: Frames per second to extract.
*   `CLIP_MODEL_NAME`: Which CLIP model to use.
*   `FORCE_CPU`: Set to `True` to prevent using CUDA.
*   `API_HOST`, `API_PORT`: Host and port for the API server.
*   `ALLOWED_HOSTS`: List of origins allowed for CORS.
*   Paths for `FAISS_INDEX_DIR`, `FRAMES_DIR`, `LOGS_DIR`, etc.

## Running the Application

**1. Process Videos (Generate Embeddings):**

*   Place your video files inside the directory specified by `VIDEO_DIR` (default: `./data/videos/`). Ensure the directory exists.
*   Run the processing script from the project root directory:

    ```bash
    python -m app.process_videos
    ```

*   **Options:**
    *   `--video-dir /path/to/other/videos`: Process videos from a different directory.
    *   `--extensions .mov .mkv`: Specify different video extensions.
    *   `--use-cpu`: Force CPU usage for this run.
    *   `--verbose` or `-v`: Enable more detailed logging.

    This script will:
    *   Scan the video directory.
    *   Extract frames based on `FRAME_EXTRACTION_FPS`.
    *   Generate CLIP embeddings.
    *   Filter duplicates.
    *   Save unique frames to `FRAMES_DIR` (default: `./app/static/frames/`).
    *   Build and save the FAISS index to `FAISS_INDEX_DIR` (default: `./data/faiss_index/`).

**2. Run the API Server:**

*   Start the FastAPI server using Uvicorn:

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```
    *(Adjust `--host` and `--port` if needed, or configure via `.env`/environment variables)*

*   The API will be available at `http://<your-host>:<your-port>` (e.g., `http://127.0.0.1:8000`).
*   Interactive API documentation (Swagger UI) is available at `http://<your-host>:<your-port>/docs`.
*   Alternative API documentation (ReDoc) is available at `http://<your-host>:<your-port>/redoc`.

## API Usage

**Endpoint:** `GET /api/v1/search`

**Query Parameters:**

*   `query` (string, required): The text query to search for.
*   `top_k` (integer, optional, default: 4): The maximum number of results to return (controlled by `DEFAULT_TOP_K` and `MAX_TOP_K` in settings).

**Example using `curl`:**

```bash
# Using localhost and default port 8000
curl -X GET "http://localhost:8000/api/v1/search?query=a%20dog%20playing%20fetch&top_k=5"
```

**Example using Browser:**

Open your browser and navigate to:
`http://localhost:8000/api/v1/search?query=a dog playing fetch`

**Example Response:**

```json
{
  "results": [
    {
      "image_url": "http://localhost:8000/static/frames/dog_video.mp4_frame_00077.jpg"
    },
    {
      "image_url": "http://localhost:8000/static/frames/park_fun.mov_frame_00210.jpg"
    }
    // ... more results up to top_k
  ]
}
```

*   `image_url`: Direct URL to the static frame image file.

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           └── search.py         # Search API endpoint logic
│   │   ├── core/
│   │   │   └── config.py             # Pydantic settings and configuration
│   │   │   ├── search/
│   │   │   │   └── faiss_service.py  # FAISS index loading, saving, searching
│   │   │   └── video/
│   │   │       ├── frame_processor.py  # Processes search results (e.g., creates URLs)
│   │   │       └── processor.py        # Main video processing orchestrator
│   │   ├── utils/
│   │   │   ├── duplicate_detector.py # Duplicate frame detection logic
│   │   │   ├── error_handling.py     # Custom exceptions and handlers
│   │   │   ├── file_ops.py           # File system operations (e.g., ensure directory)
│   │   │   ├── image_ops.py          # Frame extraction, saving
│   │   │   └── model_utils.py        # CLIP model loading, embedding generation
│   │   ├── static/
│   │   │   └── frames/               # Directory where extracted frames are saved
│   │   ├── __init__.py               # Makes 'app' a package
│   │   ├── main.py                   # FastAPI app definition, lifespan, middleware
│   │   └── process_videos.py         # Standalone script to process videos
│   ├── data/
│   │   ├── videos/                   # Default directory for input video files
│   │   └── faiss_index/              # Default directory for FAISS index files
│   ├── logs/                       # Directory for log files (if configured)
│   ├── tests/                      # Placeholder for tests
│   ├── venv/                       # Python virtual environment (if created with this name)
│   ├── .gitignore                  # Specifies intentionally untracked files that Git should ignore
│   └── README.md                   # This file
└── requirements.txt            # Python dependencies
```

## Notes

*   Ensure the directories specified in the configuration (`VIDEO_DIR`, `FRAMES_DIR`, `FAISS_INDEX_DIR`, `LOGS_DIR`) exist or can be created by the application.
The `ensure_configured_dirs()` function attempts to create them on startup.
*   Error handling is implemented via FastAPI exception handlers in `app/main.py`.
*   Logging is configured basically; consider using a more robust logging configuration for production. 