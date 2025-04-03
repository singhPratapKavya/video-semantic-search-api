"""Image and video frame processing utilities."""

import cv2
import numpy as np
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union
from pathlib import Path


@contextmanager
def video_capture(video_path: Union[str, Path]):
    """
    Context manager for video capture.
    
    Args:
        video_path: Path to the video file
        
    Yields:
        OpenCV VideoCapture object
    """
    cap = cv2.VideoCapture(str(video_path)) # Use str() for OpenCV
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")
    try:
        yield cap
    finally:
        cap.release()


def extract_frames(
    video_path: Union[str, Path],
    fps: float,
    max_frames: Optional[int] = None
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Extract frames from a video sequentially at a target FPS.

    Args:
        video_path: Path to video file
        fps: Target frames per second to extract
        max_frames: Maximum number of frames to extract

    Returns:
        Tuple of (frames, timestamps)
    """
    frames = []
    timestamps = []
    frame_count = 0
    time_interval = 1.0 / fps  # Time gap between frames to capture
    next_capture_time = 0.0

    with video_capture(video_path) as cap:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            raise ValueError(f"Could not read FPS from video: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            # Get current timestamp in seconds
            current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # Capture frame if it's at or after the next capture time
            if current_timestamp >= next_capture_time:
                frames.append(frame)
                timestamps.append(current_timestamp)
                frame_count += 1
                next_capture_time = current_timestamp + time_interval

                # Stop if max_frames reached
                if max_frames is not None and frame_count >= max_frames:
                    break

    return frames, timestamps


def save_frame(frame: np.ndarray, output_path: Union[str, Path]) -> None:
    """
    Save a frame to disk.
    
    Args:
        frame: Frame to save
        output_path: Path to save frame to
    """
    # Ensure the parent directory exists (optional, but good practice)
    # path = Path(output_path)
    # path.parent.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_path), frame)