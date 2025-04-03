from pydantic import BaseModel
from typing import List

class ApiResultItem(BaseModel):
    """Represents a single API result item."""
    image_url: str

class ApiResponse(BaseModel):
    """API response structure."""
    results: List[ApiResultItem]

