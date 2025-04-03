#!/usr/bin/env python3
"""
CLI script to search video frames using the search service.

Usage:
    python -m scripts.search_frames --query "person walking" --top 4
"""

import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path using pathlib
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.video.frame_processor import FrameProcessor
from app.core.config import FRAMES_DIR

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _format_results_for_json(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Helper function to format search results for JSON output."""
    json_results = []
    for result in results:
        metadata = result.get('metadata', {})
        json_result = {
            'clip_score': float(result.get('clip_score', 0.0)),
            'video_name': Path(metadata.get('video_path', '')).name,
            'timestamp': float(metadata.get('timestamp', 0.0)),
            'frame_path': str(Path(FRAMES_DIR) / metadata.get('frame_path', ''))
        }
        json_results.append(json_result)
    return json_results

def search_frames(query: str, top_k: int = 4, output_format: str = 'text') -> List[Dict[str, Any]]:
    """
    Search for frames using the search service.
    
    Args:
        query: Text query
        top_k: Number of results to return
        output_format: Output format ('text', 'json', or 'both')
        
    Returns:
        List of raw result dictionaries from the service
    """
    # Initialize the frame processor
    frame_processor = FrameProcessor()
    
    # Search for frames
    logger.info(f"Searching for: {query}")
    results = frame_processor.search_frames(query, top_k=top_k)
    
    if not results:
        print("No results found.")
        return []
        
    if output_format in ('text', 'both'):
        # Print results in text format
        print(f"\nTop {len(results)} results for query: '{query}'")
        print("-" * 60)
        
        for i, result in enumerate(results):
            metadata = result.get('metadata', {})
            score = result.get('clip_score', 0.0)
            video_path = metadata.get('video_path', 'N/A')
            timestamp = metadata.get('timestamp', 0.0)
            frame_rel_path = metadata.get('frame_path', 'N/A')
            frame_abs_path = Path(FRAMES_DIR) / frame_rel_path
            
            print(f"{i+1}. Score: {score:.4f}")
            print(f"   Video: {Path(video_path).name} | Time: {timestamp:.2f}s")
            print(f"   Frame: {frame_abs_path}") # Show absolute path for clarity in script
            print("-" * 60)
    
    if output_format in ('json', 'both'):
        json_results = _format_results_for_json(results)
        # Print JSON if format is 'json' or 'both' and no output file specified (handled in main)
        if output_format == 'json':
            print(json.dumps(json_results, indent=2))
            
    return results # Return the original results for potential file saving

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Search video frames')
    parser.add_argument('--query', type=str, required=True,
                        help='Text query to search for')
    parser.add_argument('--top', type=int, default=4,
                        help='Number of top results to return')
    parser.add_argument('--format', choices=['text', 'json', 'both'], default='text',
                        help='Output format')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path (if format includes json)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Search frames
    results = search_frames(args.query, args.top, args.format)
    
    # Optionally save to file
    if args.output and args.format in ('json', 'both'):
        if results: # Only save if there are results
            json_results = _format_results_for_json(results)
            try:
                with open(args.output, 'w') as f:
                    json.dump(json_results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
            except IOError as e:
                logger.error(f"Failed to write results to {args.output}: {e}")
        else:
            logger.warning(f"No results to save to {args.output}")


if __name__ == "__main__":
    main() 