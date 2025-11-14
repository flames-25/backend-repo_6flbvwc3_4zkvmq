"""
Database Schemas for Aura Photos

Each Pydantic model represents a MongoDB collection. The collection name is the
lowercase class name.
"""

from pydantic import BaseModel, Field
from typing import Optional, List

class AuraPhoto(BaseModel):
    """Aura analysis record for a single uploaded photo."""
    user_id: Optional[str] = Field(None, description="Anonymous or linked user id")
    image_url: str = Field(..., description="Public URL/path to the stored image")
    dominant_colors: List[str] = Field(default_factory=list, description="Top hex colors detected")
    aura_label: str = Field(..., description="Primary aura category e.g., Mystic Violet")
    aura_score: float = Field(..., ge=0, le=1, description="Confidence score 0..1")
    notes: Optional[str] = Field(None, description="Short interpretation text")
