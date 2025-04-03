"""File operations utilities for the application."""

import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager


def ensure_directory(directory_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state(file_path: Union[str, Path], state: Dict[str, Any]) -> None:
    """
    Save state to a JSON file.
    
    Args:
        file_path: Path to save the state file
        state: Dictionary containing state data
    """
    path = Path(file_path)
    # Ensure parent directory exists
    ensure_directory(path.parent)
    
    with path.open('w') as f:
        json.dump(state, f)


def load_state(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Load state from a JSON file.
    
    Args:
        file_path: Path to the state file
        
    Returns:
        State dictionary or None if file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        return None
        
    with path.open('r') as f:
        return json.load(f)


@contextmanager
def temporary_file(suffix: str = '.tmp') -> Path:
    """
    Context manager for creating and cleaning up temporary files.
    
    Args:
        suffix: File suffix (extension)
        
    Yields:
        Path object for the temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    temp_path = Path(temp_file.name)
    try:
        temp_file.close()
        yield temp_path
    finally:
        # Clean up the file when context exits
        if temp_path.exists():
            temp_path.unlink() 