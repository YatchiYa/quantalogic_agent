"""Controller for scheduling Chrome executions and Instagram content flow."""

import asyncio
import json
import subprocess
import sys
import os
from datetime import datetime, time
from typing import Dict, List, Optional, Union
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from loguru import logger
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..database import get_db

# Import Instagram content flow
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flows.instagram_content.instagram_content_flow import generate_instagram_content

# Router definition
router = APIRouter(prefix="/api/chrome", tags=["chrome"])

# In-memory storage for scheduled tasks
# In a production environment, you would store this in a database
chrome_schedules = {}
instagram_schedules = {}
running_tasks = {}

class ChromeScheduleTime(BaseModel):
    """Model for specific time scheduling."""
    hour: int
    minute: int
    second: int = 0
    
class InstagramScheduleResponse(BaseModel):
    """Response model for Instagram schedule operations."""
    id: str
    name: str
    status: str
    message: str

class ChromeScheduleInterval(BaseModel):
    """Model for interval-based scheduling."""
    minutes: int = 0
    hours: int = 0
    days: int = 0

class ChromeExecutionConfig(BaseModel):
    """Configuration for Chrome execution."""
    url: str
    headless: bool = False
    timeout_seconds: int = 60
    additional_args: List[str] = []

class ChromeScheduleRequest(BaseModel):
    """Request model for scheduling Chrome executions."""
    name: str
    execution_config: ChromeExecutionConfig
    schedule_type: str  # "time", "interval", or "cron"
    times: Optional[List[ChromeScheduleTime]] = None
    interval: Optional[ChromeScheduleInterval] = None
    cron_expression: Optional[str] = None
    enabled: bool = True

class ChromeScheduleResponse(BaseModel):
    """Response model for Chrome schedule operations."""
    id: str
    name: str
    status: str
    message: str

# Models for Chrome scheduler
# Models for Instagram content flow scheduling
class InstagramContentConfig(BaseModel):
    """Configuration for Instagram content generation."""
    content_context: str = Field(description="The content context for Instagram post generation")
    num_images: int = Field(default=1, description="Number of images to generate (0-3)")
    generate_images: bool = Field(default=True, description="Whether to generate images")
    analysis_model: str = Field(default="gemini/gemini-2.0-flash", description="Model for content analysis")
    content_model: str = Field(default="gemini/gemini-2.0-flash", description="Model for content generation")
    image_model: str = Field(default="gemini/gemini-2.0-flash", description="Model for image prompt generation")
    image_generator: str = Field(default="stable_diffusion", description="Image generator to use (stable_diffusion or dalle)")

class InstagramScheduleRequest(BaseModel):
    """Request model for scheduling Instagram content generation."""
    name: str = Field(description="Name of the schedule")
    instagram_config: InstagramContentConfig = Field(description="Instagram content generation configuration")
    schedule_type: str = Field(description="Type of schedule: 'time', 'interval', or 'cron'")
    times: Optional[List[ChromeScheduleTime]] = Field(default=None, description="List of specific times to run (for 'time' schedule type)")
    interval: Optional[ChromeScheduleInterval] = Field(default=None, description="Interval configuration (for 'interval' schedule type)")
    cron_expression: Optional[str] = Field(default=None, description="Cron expression (for 'cron' schedule type)")
    enabled: bool = Field(default=True, description="Whether the schedule is enabled")
    

# Helper functions
async def execute_chrome(config: ChromeExecutionConfig) -> Dict[str, str]:
    """Execute Chrome with the given configuration."""
    try:
        cmd = ["google-chrome"]
        
        if config.headless:
            cmd.extend(["--headless"])
        
        # Add URL
        cmd.append(config.url)
        
        # Add any additional arguments
        if config.additional_args:
            cmd.extend(config.additional_args)
        
        logger.info(f"Executing Chrome: {' '.join(cmd)}")
        
        # Execute Chrome process
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for process with timeout
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=config.timeout_seconds
            )
            
            if process.returncode != 0:
                logger.error(f"Chrome execution failed: {stderr.decode()}")
                return {"status": "error", "message": f"Chrome execution failed: {stderr.decode()}"}
            
            return {"status": "success", "message": "Chrome execution completed successfully"}
            
        except asyncio.TimeoutError:
            # Kill the process if it times out
            try:
                process.kill()
            except:
                pass
            return {"status": "error", "message": f"Chrome execution timed out after {config.timeout_seconds} seconds"}
            
    except Exception as e:
        logger.error(f"Error executing Chrome: {str(e)}")
        return {"status": "error", "message": f"Error executing Chrome: {str(e)}"}

async def schedule_task(schedule_id: str, schedule_type: str = "chrome", db: Session = None):
    """Background task to handle scheduled executions."""
    if schedule_type == "chrome":
        schedule = chrome_schedules.get(schedule_id)
        if not schedule or not schedule.get("enabled", False):
            return
        
        logger.info(f"Running scheduled Chrome task: {schedule['name']}")
        
        result = await execute_chrome(schedule["execution_config"])
        
        # Log execution result
        logger.info(f"Chrome execution result for {schedule['name']}: {result['status']} - {result['message']}")
        
        # Store execution history (in a real app, you would save this to a database)
        if "history" not in schedule:
            schedule["history"] = []
        
        schedule["history"].append({
            "timestamp": datetime.now().isoformat(),
            "status": result["status"],
            "message": result["message"]
        })
        
        # Limit history size
        if len(schedule["history"]) > 100:
            schedule["history"] = schedule["history"][-100:]
        
        # Update schedule in storage
        chrome_schedules[schedule_id] = schedule
    
    elif schedule_type == "instagram":
        schedule = instagram_schedules.get(schedule_id)
        if not schedule or not schedule.get("enabled", False):
            return
        
        logger.info(f"Running scheduled Instagram content flow: {schedule['name']}")
        
        try:
            # Extract Instagram content configuration
            instagram_config = schedule["instagram_config"]
            
            # Generate Instagram content
            result = await execute_instagram_content_flow(instagram_config)
            
            # Log execution result
            logger.info(f"Instagram content flow result for {schedule['name']}: {result['status']} - {result['message']}")
            
            # Store execution history
            if "history" not in schedule:
                schedule["history"] = []
            
            schedule["history"].append({
                "timestamp": datetime.now().isoformat(),
                "status": result["status"],
                "message": result["message"],
                "output_file": result.get("output_file")
            })
            
            # Limit history size
            if len(schedule["history"]) > 100:
                schedule["history"] = schedule["history"][-100:]
            
            # Update schedule in storage
            instagram_schedules[schedule_id] = schedule
            
        except Exception as e:
            logger.error(f"Error executing Instagram content flow: {str(e)}")
            
            # Record the error in history
            if "history" not in schedule:
                schedule["history"] = []
            
            schedule["history"].append({
                "timestamp": datetime.now().isoformat(),
                "status": "error",
                "message": f"Error executing Instagram content flow: {str(e)}"
            })
            
            # Update schedule in storage
            instagram_schedules[schedule_id] = schedule

# Helper function for Instagram content flow
async def execute_instagram_content_flow(config: Dict) -> Dict[str, str]:
    """Execute Instagram content flow with the given configuration."""
    try:
        # Extract parameters from config
        content_context = config.get("content_context", "")
        num_images = config.get("num_images", 1)
        generate_images = config.get("generate_images", True)
        analysis_model = config.get("analysis_model", "gemini/gemini-2.0-flash")
        content_model = config.get("content_model", "gemini/gemini-2.0-flash")
        image_model = config.get("image_model", "gemini/gemini-2.0-flash")
        image_generator = config.get("image_generator", "stable_diffusion")
        
        # Create output directory if it doesn't exist
        output_dir = Path("instagram_content_output")
        output_dir.mkdir(exist_ok=True)
        
        # Generate a unique filename for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"instagram_content_{timestamp}.md"
        
        # Generate Instagram content
        logger.info(f"Generating Instagram content for: {content_context[:50]}...")
        
        # Run the Instagram content flow
        post = await generate_instagram_content(
            content_context=content_context,
            num_images=num_images,
            generate_images=generate_images,
            analysis_model=analysis_model,
            content_model=content_model,
            image_model=image_model,
            image_generator=image_generator
        )
        
        # Save the output to a markdown file
        with open(output_file, "w") as f:
            f.write(f"# Instagram Content: {timestamp}\n\n")
            f.write(f"## Caption\n\n{post.caption}\n\n")
            
            if post.images:
                f.write(f"## Images\n\n")
                for i, image in enumerate(post.images):
                    f.write(f"### Image {i+1}\n\n")
                    f.write(f"Path: {image.image_path}\n\n")
                    
            if post.carousel_text:
                f.write(f"## Carousel Text\n\n")
                for i, text in enumerate(post.carousel_text):
                    f.write(f"### Slide {i+1}\n\n{text}\n\n")
        
        logger.info(f"Instagram content saved to: {output_file}")
        
        return {
            "status": "success",
            "message": "Instagram content generated successfully",
            "output_file": str(output_file)
        }
        
    except Exception as e:
        logger.error(f"Error generating Instagram content: {str(e)}")
        return {
            "status": "error",
            "message": f"Error generating Instagram content: {str(e)}"
        }

# Routes
@router.post("/schedule", response_model=ChromeScheduleResponse)
async def create_schedule(
    request: ChromeScheduleRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> ChromeScheduleResponse:
    """Create a new Chrome execution schedule."""
    try:
        # Validate schedule configuration
        if request.schedule_type == "time" and not request.times:
            raise HTTPException(status_code=400, detail="Times must be provided for time-based scheduling")
        elif request.schedule_type == "interval" and not request.interval:
            raise HTTPException(status_code=400, detail="Interval must be provided for interval-based scheduling")
        elif request.schedule_type == "cron" and not request.cron_expression:
            raise HTTPException(status_code=400, detail="Cron expression must be provided for cron-based scheduling")
        
        # Generate a unique ID for the schedule
        import uuid
        schedule_id = str(uuid.uuid4())
        
        # Store schedule configuration
        chrome_schedules[schedule_id] = {
            "id": schedule_id,
            "name": request.name,
            "execution_config": request.execution_config.dict(),
            "schedule_type": request.schedule_type,
            "times": [t.dict() for t in request.times] if request.times else None,
            "interval": request.interval.dict() if request.interval else None,
            "cron_expression": request.cron_expression,
            "enabled": request.enabled,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # In a production app, you would set up the actual scheduler here
        # For simplicity, we'll just return success
        
        logger.info(f"Created Chrome schedule: {request.name} (ID: {schedule_id})")
        
        return ChromeScheduleResponse(
            id=schedule_id,
            name=request.name,
            status="created",
            message="Schedule created successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating Chrome schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Chrome schedule: {str(e)}")

@router.get("/schedule/{schedule_id}", response_model=Dict)
async def get_schedule(schedule_id: str, db: Session = Depends(get_db)) -> Dict:
    """Get a Chrome execution schedule by ID."""
    if schedule_id not in chrome_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    return chrome_schedules[schedule_id]

@router.get("/schedules", response_model=List[Dict])
async def list_schedules(db: Session = Depends(get_db)) -> List[Dict]:
    """List all Chrome execution schedules."""
    return list(chrome_schedules.values())

@router.delete("/schedule/{schedule_id}", response_model=ChromeScheduleResponse)
async def delete_schedule(schedule_id: str, db: Session = Depends(get_db)) -> ChromeScheduleResponse:
    """Delete a Chrome execution schedule."""
    if schedule_id not in chrome_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = chrome_schedules.pop(schedule_id)
    
    logger.info(f"Deleted Chrome schedule: {schedule['name']} (ID: {schedule_id})")
    
    return ChromeScheduleResponse(
        id=schedule_id,
        name=schedule["name"],
        status="deleted",
        message="Schedule deleted successfully"
    )

@router.post("/schedule/{schedule_id}/execute", response_model=Dict)
async def execute_schedule_now(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict:
    """Execute a Chrome schedule immediately."""
    if schedule_id not in chrome_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = chrome_schedules[schedule_id]
    
    # Execute the Chrome task in the background
    background_tasks.add_task(schedule_task, schedule_id, db)
    
    return {
        "status": "executing",
        "message": f"Executing Chrome schedule: {schedule['name']}",
        "schedule_id": schedule_id
    }

@router.put("/schedule/{schedule_id}/toggle", response_model=ChromeScheduleResponse)
async def toggle_schedule(schedule_id: str, db: Session = Depends(get_db)) -> ChromeScheduleResponse:
    """Enable or disable a Chrome execution schedule."""
    if schedule_id not in chrome_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = chrome_schedules[schedule_id]
    schedule["enabled"] = not schedule["enabled"]
    schedule["updated_at"] = datetime.now().isoformat()
    
    status = "enabled" if schedule["enabled"] else "disabled"
    logger.info(f"{status.capitalize()} Chrome schedule: {schedule['name']} (ID: {schedule_id})")
    
    return ChromeScheduleResponse(
        id=schedule_id,
        name=schedule["name"],
        status=status,
        message=f"Schedule {status} successfully"
    )

# Instagram content flow scheduling endpoints
@router.post("/instagram/schedule", response_model=InstagramScheduleResponse)
async def create_instagram_schedule(
    request: InstagramScheduleRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> InstagramScheduleResponse:
    """Create a new Instagram content flow schedule."""
    try:
        # Validate schedule configuration
        if request.schedule_type == "time" and not request.times:
            raise HTTPException(status_code=400, detail="Times must be provided for time-based scheduling")
        elif request.schedule_type == "interval" and not request.interval:
            raise HTTPException(status_code=400, detail="Interval must be provided for interval-based scheduling")
        elif request.schedule_type == "cron" and not request.cron_expression:
            raise HTTPException(status_code=400, detail="Cron expression must be provided for cron-based scheduling")
        
        # Generate a unique ID for the schedule
        import uuid
        schedule_id = str(uuid.uuid4())
        
        # Store schedule configuration
        instagram_schedules[schedule_id] = {
            "id": schedule_id,
            "name": request.name,
            "instagram_config": request.instagram_config.dict(),
            "schedule_type": request.schedule_type,
            "times": [t.dict() for t in request.times] if request.times else None,
            "interval": request.interval.dict() if request.interval else None,
            "cron_expression": request.cron_expression,
            "enabled": request.enabled,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # In a production app, you would set up the actual scheduler here
        # For simplicity, we'll just return success
        
        logger.info(f"Created Instagram content schedule: {request.name} (ID: {schedule_id})")
        
        return InstagramScheduleResponse(
            id=schedule_id,
            name=request.name,
            status="created",
            message="Instagram content schedule created successfully"
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error creating Instagram content schedule: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating Instagram content schedule: {str(e)}")

@router.get("/instagram/schedule/{schedule_id}", response_model=Dict)
async def get_instagram_schedule(schedule_id: str, db: Session = Depends(get_db)) -> Dict:
    """Get an Instagram content flow schedule by ID."""
    if schedule_id not in instagram_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    return instagram_schedules[schedule_id]

@router.get("/instagram/schedules", response_model=List[Dict])
async def list_instagram_schedules(db: Session = Depends(get_db)) -> List[Dict]:
    """List all Instagram content flow schedules."""
    return list(instagram_schedules.values())

@router.delete("/instagram/schedule/{schedule_id}", response_model=InstagramScheduleResponse)
async def delete_instagram_schedule(schedule_id: str, db: Session = Depends(get_db)) -> InstagramScheduleResponse:
    """Delete an Instagram content flow schedule."""
    if schedule_id not in instagram_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = instagram_schedules.pop(schedule_id)
    
    logger.info(f"Deleted Instagram content schedule: {schedule['name']} (ID: {schedule_id})")
    
    return InstagramScheduleResponse(
        id=schedule_id,
        name=schedule["name"],
        status="deleted",
        message="Schedule deleted successfully"
    )

@router.post("/instagram/schedule/{schedule_id}/execute", response_model=Dict)
async def execute_instagram_schedule_now(
    schedule_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
) -> Dict:
    """Execute an Instagram content flow schedule immediately."""
    if schedule_id not in instagram_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = instagram_schedules[schedule_id]
    
    # Execute the Instagram content flow task in the background
    background_tasks.add_task(schedule_task, schedule_id, "instagram", db)
    
    return {
        "status": "executing",
        "message": f"Executing Instagram content flow schedule: {schedule['name']}",
        "schedule_id": schedule_id
    }

@router.put("/instagram/schedule/{schedule_id}/toggle", response_model=InstagramScheduleResponse)
async def toggle_instagram_schedule(schedule_id: str, db: Session = Depends(get_db)) -> InstagramScheduleResponse:
    """Enable or disable an Instagram content flow schedule."""
    if schedule_id not in instagram_schedules:
        raise HTTPException(status_code=404, detail=f"Schedule with ID {schedule_id} not found")
    
    schedule = instagram_schedules[schedule_id]
    schedule["enabled"] = not schedule["enabled"]
    schedule["updated_at"] = datetime.now().isoformat()
    
    status = "enabled" if schedule["enabled"] else "disabled"
    logger.info(f"{status.capitalize()} Instagram content schedule: {schedule['name']} (ID: {schedule_id})")
    
    return InstagramScheduleResponse(
        id=schedule_id,
        name=schedule["name"],
        status=status,
        message=f"Schedule {status} successfully"
    )
