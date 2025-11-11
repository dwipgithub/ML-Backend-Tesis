from typing import Any, Optional
from pydantic import BaseModel

class ApiResponse(BaseModel):
    error: bool
    message: str
    data: Optional[Any] = None