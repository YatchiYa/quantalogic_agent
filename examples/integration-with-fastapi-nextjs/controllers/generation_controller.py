import asyncio
from fastapi import APIRouter, HTTPException
from typing import Dict
from loguru import logger
from ..models import (
    TutorialRequest,
    CourseRequest,
    QuizRequest,
    JourneyRequest,
    AnalyzePaperRequest,
    LinkedInIntroduceContentRequest,
    ConvertRequest,
    ImageGenerationRequest,
    BookNovelRequest,
    ImageAnalysisRequest,
    TaskSubmission
)
from ..agent_server import agent_state, server_state
from ..models import FileUploadResponse

router = APIRouter(prefix="/api/agent", tags=["generation"])

@router.post("/generate-tutorial")
async def generate_tutorial(request: TutorialRequest) -> Dict[str, str]:
    """Generate a tutorial from markdown content."""
    try:
        task_submission = TaskSubmission(task="generate_tutorial")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Tutorial generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_tutorial(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Tutorial generation started"
        }
    except Exception as e:
        logger.error(f"Error starting tutorial generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start tutorial generation: {str(e)}"
        )

@router.post("/generate-coursera")
async def generate_course(request: CourseRequest) -> Dict[str, str]:
    """Generate a course from markdown content."""
    try:
        task_submission = TaskSubmission(task="generate_course")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Course generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_course(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Course generation started"
        }
    except Exception as e:
        logger.error(f"Error starting course generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start course generation: {str(e)}"
        )

@router.post("/generate-quizz")
async def generate_quizz(request: QuizRequest) -> Dict[str, str]:
    """Generate a quiz from markdown content."""
    try:
        task_submission = TaskSubmission(task="generate_quizz")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Quiz generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_quizz(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Quiz generation started"
        }
    except Exception as e:
        logger.error(f"Error starting quiz generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start quiz generation: {str(e)}"
        )

@router.post("/generate-journey")
async def generate_journey(request: JourneyRequest) -> Dict[str, str]:
    """Generate a journey from markdown content."""
    try:
        task_submission = TaskSubmission(task="generate_journey")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Journey generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_journey(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Journey generation started"
        }
    except Exception as e:
        logger.error(f"Error starting journey generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start journey generation: {str(e)}"
        )

@router.post("/generate-analyze-paper")
async def generate_analyze_paper(request: AnalyzePaperRequest) -> Dict[str, str]:
    """Generate a paper analysis from markdown content."""
    try:
        task_submission = TaskSubmission(task="generate_analyze_paper")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Paper analysis task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_analyze_paper(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Paper analysis started"
        }
    except Exception as e:
        logger.error(f"Error starting paper analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start paper analysis: {str(e)}"
        )

@router.post("/generate-linkedin-introduce-content")
async def generate_linkedin_introduce_content(request: LinkedInIntroduceContentRequest) -> Dict[str, str]:
    """Generate LinkedIn introduction content."""
    try:
        task_submission = TaskSubmission(task="generate_linkedin_introduce_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"LinkedIn introduce content task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_linkedin_introduce_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "LinkedIn introduce content started"
        }
    except Exception as e:
        logger.error(f"Error starting LinkedIn introduce content: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start LinkedIn introduce content: {str(e)}"
        )

@router.post("/generate-convert")
async def generate_convert(request: ConvertRequest) -> Dict[str, str]:
    """Generate a PDF to Markdown conversion task."""
    try:
        task_submission = TaskSubmission(task="generate_convert")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"PDF to Markdown conversion task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_convert(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "PDF to Markdown conversion started"
        }
    except Exception as e:
        logger.error(f"Error starting PDF to Markdown conversion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start PDF to Markdown conversion: {str(e)}"
        )

@router.post("/generate-image")
async def generate_image(request: ImageGenerationRequest) -> Dict[str, str]:
    """Generate an image from a prompt."""
    try:
        task_submission = TaskSubmission(task="generate_image")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Image generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_image_generation(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Image generation started"
        }
    except Exception as e:
        logger.error(f"Error starting image generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start image generation: {str(e)}"
        )

@router.post("/generate-book-novel")
async def generate_book_novel(request: BookNovelRequest) -> Dict[str, str]:
    """Generate a book novel."""
    try:
        task_submission = TaskSubmission(task="generate_book_novel")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Book novel generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_book_creation_novel_only(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Book novel generation started"
        }
    except Exception as e:
        logger.error(f"Error starting book novel generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start book novel generation: {str(e)}"
        )

@router.post("/generate-image-analysis")
async def generate_image_analysis(request: ImageAnalysisRequest) -> Dict[str, str]:
    """Generate an image analysis."""
    try:
        task_submission = TaskSubmission(task="generate_image_analysis")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Image analysis task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_image_analysis(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Image analysis started"
        }
    except Exception as e:
        logger.error(f"Error starting image analysis: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start image analysis: {str(e)}"
        )
