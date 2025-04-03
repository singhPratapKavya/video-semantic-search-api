"""Error handling utilities for the application."""

import functools
import logging
from typing import Callable, TypeVar, Any, Optional, Type
import traceback
import inspect
from fastapi import HTTPException

# Define a generic type for function return value
T = TypeVar('T')

logger = logging.getLogger(__name__)

# Custom exceptions
class VideoProcessingError(Exception):
    """Custom exception for video processing errors."""
    pass

class FAISSServiceError(Exception):
    """Custom exception for FAISS service errors."""
    pass


def handle_errors(
    error_message: str = "Operation failed",
    exception_to_raise: Type[Exception] = HTTPException,
    log_traceback: bool = True,
    return_value: Optional[Any] = None,
    reraise: bool = True
) -> Callable:
    """
    Decorator to handle errors in functions with standardized logging and error reporting.
    Handles both async and sync functions.
    
    Args:
        error_message: Base error message to log and include in exception
        exception_to_raise: Type of exception to raise (e.g., HTTPException, FAISSServiceError)
        log_traceback: Whether to log the full traceback
        return_value: Value to return if reraise is False
        reraise: Whether to raise the exception (True) or return return_value (False)
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Check if the original function is async
        is_async_func = inspect.iscoroutinefunction(func)

        # Define sync wrapper
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            try:
                # Get the function's signature
                sig = inspect.signature(func)
                # Bind the arguments to the function's parameters
                bound_args = sig.bind(*args, **kwargs)
                # Apply the bound arguments
                return func(*bound_args.args, **bound_args.kwargs)
            except Exception as e:
                full_error = f"{error_message}: {str(e)}"
                if log_traceback:
                    logger.error(f"{full_error}\n{traceback.format_exc()}")
                else:
                    logger.error(full_error)
                
                if reraise:
                    if exception_to_raise is HTTPException:
                        if isinstance(e, HTTPException):
                            raise e
                        else:
                            raise HTTPException(status_code=500, detail=full_error)
                    else:
                        raise exception_to_raise(full_error) from e
                else:
                    return return_value

        # Define async wrapper
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            try:
                # Get the function's signature
                sig = inspect.signature(func)
                # Bind the arguments to the function's parameters
                bound_args = sig.bind(*args, **kwargs)
                # Apply the bound arguments
                return await func(*bound_args.args, **bound_args.kwargs)
            except Exception as e:
                full_error = f"{error_message}: {str(e)}"
                if log_traceback:
                    logger.error(f"{full_error}\n{traceback.format_exc()}")
                else:
                    logger.error(full_error)
                
                if reraise:
                    if exception_to_raise is HTTPException:
                        if isinstance(e, HTTPException):
                            raise e
                        else:
                            raise HTTPException(status_code=500, detail=full_error)
                    else:
                        raise exception_to_raise(full_error) from e
                else:
                    return return_value

        # Return the correct wrapper based on whether func is async
        return async_wrapper if is_async_func else sync_wrapper
    return decorator


def handle_video_processing_errors(error_message: str = "Video processing failed") -> Callable:
    """
    Specialized decorator for video processing functions.
    
    Args:
        error_message: Base error message
        
    Returns:
        Decorated function
    """
    return handle_errors(
        error_message=error_message,
        exception_to_raise=VideoProcessingError,
        log_traceback=True,
        reraise=True
    )


def handle_faiss_errors(error_message: str = "FAISS operation failed") -> Callable:
    """
    Specialized decorator for FAISS operations.
    
    Args:
        error_message: Base error message
        
    Returns:
        Decorated function
    """
    return handle_errors(
        error_message=error_message,
        exception_to_raise=FAISSServiceError,
        log_traceback=True,
        reraise=True
    ) 