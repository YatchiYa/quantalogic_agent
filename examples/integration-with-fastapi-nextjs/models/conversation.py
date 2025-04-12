from typing import Optional, List
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class ConversationConfig(BaseModel):
    id: UUID
    title: Optional[str] = None
    description: Optional[str] = None
    model_id: Optional[str] = None
    project: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    agent_id: Optional[str] = None
    is_public: bool = False
    is_archived: bool = False
    is_favorite: bool = False

class ConversationUpdateConfig(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    model_id: Optional[str] = None
    project: Optional[str] = None
    is_public: Optional[bool] = None
    is_archived: Optional[bool] = None
    is_favorite: Optional[bool] = None
