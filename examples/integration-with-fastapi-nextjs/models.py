"""Pydantic models for the QuantaLogic API."""

from typing import Any, Dict, Optional, List

from pydantic import BaseModel

from quantalogic.agent_config import MODEL_NAME
from datetime import datetime

# Constants
SHUTDOWN_TIMEOUT = 10.0  # seconds
VALIDATION_TIMEOUT = 30.0  # seconds
UPLOAD_DIR = "/tmp/data"  # Directory for file uploads

class HtmlContent(BaseModel):
    content: str

class ToolParameters(BaseModel):
    """Parameters for tool configurations.
    
    This class defines all possible parameters that can be passed to different tools.
    Each tool will only use the parameters it needs.
    """
    # LLM and Model related parameters
    model_name: Optional[str] = None
    vision_model_name: Optional[str] = None
    provider: Optional[str] = None
    additional_info: Optional[str] = None

    # Database related parameters
    connection_string: Optional[str] = None

    # Git related parameters
    access_token: Optional[str] = None  # For Bitbucket
    auth_token: Optional[str] = None    # For GitHub and other git operations

    class Config:
        """Pydantic config for ToolParameters."""
        extra = "allow"  # Allow extra fields for future extensibility

 
class FileUploadResponse(BaseModel):
    status: str
    filename: str
    path: str
    project_path: str
    size: str
    content_type: str

class TutorialRequest(BaseModel):
    """Request model for tutorial generation."""
    markdown_content: str
    model: str = "gemini/gemini-2.0-flash"
    num_chapters: int = 5
    words_per_chapter: int = 2000
    copy_to_clipboard: bool = True
    skip_refinement: bool = True

class CourseRequest(BaseModel):
    subject: str
    number_of_chapters: int
    level: str
    words_by_chapter: int
    target_directory: Optional[str] = None
    pdf_generation: Optional[bool] = True
    docx_generation: Optional[bool] = True
    epub_generation: Optional[bool] = False
    model_name: Optional[str] = "gemini/gemini-2.0-flash"
    model: Optional[str] = "gemini/gemini-2.0-flash"


class JourneyRequest(BaseModel):
    """Request model for journey generation."""
    destination: str
    start_date: str
    end_date: str
    budget: float
    model: str = "gemini/gemini-2.0-flash"

class QuizRequest(BaseModel):
    """Request model for quiz generation.""" 
    file_path: Optional[str] = None
    model: str = "gemini/gemini-2.0-flash"
    num_questions: int = 10
    token_limit: int = 2000
    save: bool = True
    words_per_question: int = 2000

class AnalyzePaperRequest(BaseModel):
    """Request model for paper analysis."""
    file_path: str
    text_extraction_model: Optional[str] = "gemini/gemini-2.0-flash"
    cleaning_model: Optional[str] = "gemini/gemini-2.0-flash"
    writing_model: Optional[str] = "gemini/gemini-2.0-flash"
    output_dir: Optional[str] = None
    copy_to_clipboard_flag: Optional[bool] = True
    max_character_count: Optional[int] = 3000

class LinkedInIntroduceContentRequest(BaseModel):
    """Request model for LinkedIn introduce content generation."""
    file_path: str
    analysis_model: Optional[str] = "gemini/gemini-2.0-flash"
    writing_model: Optional[str] = "gemini/gemini-2.0-flash"
    cleaning_model: Optional[str] = "gemini/gemini-2.0-flash"
    formatting_model: Optional[str] = "gemini/gemini-2.0-flash"
    copy_to_clipboard_flag: Optional[bool] = True
    intent: Optional[str] = None
    mock_analysis: Optional[bool] = False

class ImageAnalysisRequest(BaseModel):
    image_url: str
    image_context: Optional[str] = None
    analysis_context: Optional[str] = None
    vision_model: Optional[str] = "openai/gpt-4o-mini"
    analysis_model: Optional[str] = "gemini/gemini-2.0-flash"

class BookNovelRequest(BaseModel):
    content: str
    model: Optional[str] = "gemini/gemini-2.0-flash"
    title: Optional[str] = None
    author: Optional[str] = None
    output_path: Optional[str] = None
    num_chapters: Optional[int] = None
    words_per_chapter: Optional[int] = None
    narration_style: Optional[dict] = None
    literary_style: Optional[dict] = None
    target_audience: Optional[str] = None

class ImageGenerationRequest(BaseModel):
    prompt: str
    model_type: Optional[str] = "stable_diffusion"
    style: Optional[str] = None
    size: Optional[str] = None
    analysis_model: Optional[str] = "gemini/gemini-2.0-flash"
    enhancement_model: Optional[str] = "gemini/gemini-2.0-flash"

class ConvertRequest(BaseModel):
    input_pdf: str
    output_md: Optional[str] = None
    model: str = "gemini/gemini-2.0-flash"
    system_prompt: Optional[str] = None

class EventMessage(BaseModel):
    """Event message model for SSE."""

    id: str
    event: str
    task_id: Optional[str] = None
    data: Dict[str, Any]
    timestamp: str

    model_config = {"extra": "forbid"}


class UserValidationRequest(BaseModel):
    """Request model for user validation."""

    validation_id: str
    tool_name: str
    arguments: Dict[str, Any]

    model_config = {"extra": "forbid"}


class UserValidationResponse(BaseModel):
    """Response model for user validation."""

    approved: bool

    model_config = {"extra": "forbid"}


class ToolConfig(BaseModel):
    """Configuration for a single tool."""
    type: str
    parameters: ToolParameters


class AgentConfig(BaseModel):
    """Configuration for creating a new agent."""
    id: str
    name: str
    description: str
    expertise: str
    mode: str = "custom"
    model_name: str
    agent_mode: str
    tools: List[ToolConfig]
    tags: Optional[List[str]] = None
    project: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None


class AgentUpdateConfig(BaseModel):
    """Configuration for updating an existing agent."""
    name: Optional[str] = None
    description: Optional[str] = None
    expertise: Optional[str] = None
    mode: Optional[str] = None
    model_name: Optional[str] = None
    agent_mode: Optional[str] = None
    tools: Optional[List[ToolConfig]] = None
    tags: Optional[List[str]] = None
    project: Optional[str] = None
    user_id: Optional[str] = None
    organization_id: Optional[str] = None


class TaskSubmission(BaseModel):
    """Request model for task submission."""

    task: str
    agent_id: Optional[str] = None
    model_name: Optional[str] = MODEL_NAME
    max_iterations: Optional[int] = 30
    mode: Optional[str] = "minimal"
    expertise: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None 
    conversation_id: Optional[str] = None  # Added for memory management
    model_config = {"extra": "forbid"}


class TaskStatus(BaseModel):
    """Task status response model."""

    task_id: str
    status: str  # "pending", "running", "completed", "failed"
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    total_tokens: Optional[int] = None
    model_name: Optional[str] = None




## conversations 

class ConversationConfig(BaseModel):
    id: str
    title: Optional[str] = None
    description: Optional[str] = None
    model_id: Optional[str] = None
    agent_id: Optional[str] = None
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
