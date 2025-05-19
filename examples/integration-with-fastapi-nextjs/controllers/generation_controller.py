import asyncio
from fastapi import APIRouter, HTTPException
from typing import Dict
from loguru import logger
from ..models import (
    CompanyAnalyzeRequest,
    CompetitiveContentRequest,
    DocumentContext,
    FacebookContentRequest,
    GitAnalyzeRequest,
    InstagramContentRequest,
    LinkedInContentRequest,
    MediumContentRequest,
    TalkWithDocumentRequest,
    TopicAnalyzeRequest,
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
    TaskSubmission,
    TwitterContentRequest,
    EuromillionsPredictionRequest,
    StarterPackRequest,
    CSVExplorerRequest
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


@router.post("/generate-facebook-content")
async def generate_facebook_content(request: FacebookContentRequest) -> Dict[str, str]:
    """Generate Facebook content."""
    try:
        task_submission = TaskSubmission(task="generate_facebook_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Facebook content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_facebook_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Facebook content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting Facebook content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Facebook content generation: {str(e)}"
        )


@router.post("/generate-instagram-content")
async def generate_instagram_content(request: InstagramContentRequest) -> Dict[str, str]:
    """Generate Instagram content."""
    try:
        task_submission = TaskSubmission(task="generate_instagram_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Instagram content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_instagram_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Instagram content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting Instagram content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Instagram content generation: {str(e)}"
        )


@router.post("/generate-linkedin-content")
async def generate_linkedin_content(request: LinkedInContentRequest) -> Dict[str, str]:
    """Generate LinkedIn content."""
    try:
        task_submission = TaskSubmission(task="generate_linkedin_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"LinkedIn content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_linkedin_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "LinkedIn content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting LinkedIn content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start LinkedIn content generation: {str(e)}"
        )

@router.post("/generate-medium-content")
async def generate_medium_content(request: MediumContentRequest) -> Dict[str, str]:
    """Generate Medium content."""
    try:
        task_submission = TaskSubmission(task="generate_medium_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Medium content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_medium_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Medium content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting Medium content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Medium content generation: {str(e)}"
        )

@router.post("/generate-twitter-content")
async def generate_twitter_content(request: TwitterContentRequest) -> Dict[str, str]:
    """Generate Twitter content."""
    try:
        task_submission = TaskSubmission(task="generate_twitter_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Twitter content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_twitter_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Twitter content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting Twitter content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Twitter content generation: {str(e)}"
        )

@router.post("/generate-euromillions-prediction")
async def generate_euromillions_prediction(request: EuromillionsPredictionRequest) -> Dict[str, str]:
    """Generate Euromillions prediction."""
    try:
        task_submission = TaskSubmission(task="generate_euromillions_prediction")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Euromillions prediction task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_euromillions_prediction(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Euromillions prediction started"
        }
    except Exception as e:
        logger.error(f"Error starting Euromillions prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start Euromillions prediction: {str(e)}"
        )

@router.post("/generate-starter-pack")
async def generate_starter_pack(request: StarterPackRequest) -> Dict[str, str]:
    """Generate a starter pack image based on a reference image."""
    try:
        task_submission = TaskSubmission(task="generate_starter_pack")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Starter pack generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_starter_pack(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Starter pack generation started"
        }
    except Exception as e:
        logger.error(f"Error starting starter pack generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start starter pack generation: {str(e)}"
        )

@router.post("/explore-csv")
async def explore_csv(request: CSVExplorerRequest) -> Dict[str, str]:
    """Explore CSV files with data visualization and LLM-powered analysis."""
    try:
        task_submission = TaskSubmission(task="explore_csv")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"CSV exploration task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_csv_explorer(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "CSV exploration started"
        }
    except Exception as e:
        logger.error(f"Error starting CSV exploration: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start CSV exploration: {str(e)}"
        )

@router.post("/generate-competitive-company")
async def generate_competitive_content(request: CompetitiveContentRequest) -> Dict[str, str]:
    """Generate competitive content."""
    try:
        task_submission = TaskSubmission(task="generate_competitive_content")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Competitive content generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_competitive_content(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Competitive content generation started"
        }
    except Exception as e:
        logger.error(f"Error starting competitive content generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start competitive content generation: {str(e)}"
        )

@router.post("/generate-talk-with-document")
async def generate_talk_with_document(request: TalkWithDocumentRequest) -> Dict[str, str]:
    """Generate talk with document content."""
    try:
        task_submission = TaskSubmission(task="generate_talk_with_document")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Talk with document generation task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_talk_with_document(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Talk with document generation started"
        }
    except Exception as e:
        logger.error(f"Error starting talk with document generation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start talk with document generation: {str(e)}"
        )

@router.post("/generate-git-report")
async def generate_git_report(request: GitAnalyzeRequest) -> Dict[str, str]:
    """Generate Git report."""
    try:
        task_submission = TaskSubmission(task="generate_git_analyze")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Git analyze task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_git_analyze(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Git analyze started"
        }
    except Exception as e:
        logger.error(f"Error starting git analyze: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start git analyze: {str(e)}"
        )

@router.post("/generate-company-report")
async def generate_company_report(request: CompanyAnalyzeRequest) -> Dict[str, str]:
    """Generate company report."""
    try:
        task_submission = TaskSubmission(task="generate_company_analyze")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Company analyze task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_company_analyzer(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Company analyze started"
        }
    except Exception as e:
        logger.error(f"Error starting company analyze: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start company analyze: {str(e)}"
        )

@router.post("/generate-topic-report")
async def generate_topic_report(request: TopicAnalyzeRequest) -> Dict[str, str]:
    """Generate topic report."""
    try:
        task_submission = TaskSubmission(task="generate_topic_analyze")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Topic analyze task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_topic_analyzer(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Topic analyze started"
        }
    except Exception as e:
        logger.error(f"Error starting topic analyze: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start topic analyze: {str(e)}"
        )

@router.post("/generate-context-analysis")
async def generate_context_analysis(request: DocumentContext) -> Dict[str, str]:
    """Generate context report."""
    try:
        task_submission = TaskSubmission(task="generate_context_analyze")
        task_id = await agent_state.submit_task(task_submission)
        logger.info(f"Context analyze task submitted with ID: {task_id}")
        
        asyncio.create_task(agent_state.execute_context_analysis(task_id, request))
        
        return {
            "status": "success",
            "task_id": task_id,
            "message": "Context analyze started"
        }
    except Exception as e:
        logger.error(f"Error starting context analyze: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start context analyze: {str(e)}"
        )
