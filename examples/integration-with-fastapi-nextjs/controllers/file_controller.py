import os
import shutil
from typing import Dict, Optional
import uuid
from datetime import datetime
from pathlib import Path
import mimetypes
import urllib.parse

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from loguru import logger
from ..agent_server import agent_state, server_state
from ..models import FileUploadResponse

router = APIRouter(prefix="/api/agent", tags=["files"])

# Constants
UPLOAD_DIR = "/tmp/data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class HtmlContent(BaseModel):
    content: str

class FileUploadResponse(BaseModel):
    status: str
    filename: str
    path: str
    project_path: str
    size: str
    content_type: str

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> Dict[str, str]:
    """Handle file uploads."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = str(uuid.uuid4())[:8]
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, new_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        return {
            "status": "success",
            "filename": new_filename,
            "path": file_path
        }
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fileupload", response_model=FileUploadResponse)
async def file_upload(
    file: UploadFile = File(...),
    project_path: str = Form(...)
) -> FileUploadResponse:
    """Handle file uploads with custom project paths."""
    logger.info(f"Received file upload request - Filename: {file.filename}, Content-Type: {file.content_type}, Project Path: {project_path}")
    try:
        if not file.filename:
            logger.error("No filename provided in upload")
            raise HTTPException(status_code=400, detail="No filename provided")
            
        file_content = await file.read()
        if not file_content:
            logger.error("Empty file content")
            raise HTTPException(status_code=400, detail="Empty file content")
            
        project_path = os.path.normpath(project_path)
        if project_path.startswith("/") or ".." in project_path:
            logger.error(f"Invalid project path detected: {project_path}")
            raise HTTPException(status_code=400, detail="Invalid project path")
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(project_path)
        full_dir = os.path.join(UPLOAD_DIR, os.path.dirname(project_path))
        new_filename = f"{filename}"
        file_path = os.path.join(full_dir, new_filename)
        
        logger.info(f"Creating directory: {full_dir}")
        os.makedirs(full_dir, exist_ok=True)
        
        logger.info(f"Saving file to: {file_path}")
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)
            
        response = FileUploadResponse(
            status="success",
            filename=new_filename,
            path=file_path,
            project_path=project_path,
            size=str(len(file_content)),
            content_type=file.content_type
        )
        logger.info(f"File upload successful: {response.dict()}")
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error uploading file to project: {str(e)}")
        logger.exception(e)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.post("/upload-html")
async def upload_html_content(payload: HtmlContent) -> Dict[str, str]:
    """Handle HTML content upload and save to file."""
    try:
        html_dir = "/tmp/html_templates"
        os.makedirs(html_dir, exist_ok=True)
        
        file_id = str(uuid.uuid4())[:8]
        filename = f"{file_id}.html"
        file_path = os.path.join(html_dir, filename)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload.content)
            
        return {
            "status": "success",
            "id": file_id,
            "filename": filename,
            "path": file_path
        }
    except Exception as e:
        logger.error(f"Error saving HTML content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/html/{file_id}")
async def get_html_content(file_id: str) -> Dict[str, str]:
    """Retrieve HTML content by file ID."""
    try:
        html_dir = "/tmp/html_templates"
        file_path = os.path.join(html_dir, f"{file_id}.html")
        
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail=f"HTML file with id {file_id} not found"
            )
            
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        return {
            "status": "success",
            "id": file_id,
            "content": content
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading HTML content: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download a file from the upload directory."""
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@router.get("/files/content")
async def get_file_content(file_path: str, raw: Optional[bool] = False) -> Response:
    """Retrieve file content by path with support for various file types."""
    try:
        decoded_path = urllib.parse.unquote(file_path)
        path = Path(decoded_path)
        
        if not path.is_absolute() or not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"File not found: {decoded_path}"
            )
        
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = 'application/octet-stream'
        
        if raw:
            return FileResponse(
                path=str(path),
                media_type=mime_type,
                filename=path.name
            )
        
        if mime_type.startswith(('text/', 'application/json', 'application/xml', 'application/javascript')):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return JSONResponse({
                    "status": "success",
                    "path": str(path),
                    "mime_type": mime_type,
                    "content": content,
                    "size": path.stat().st_size,
                    "filename": path.name
                })
            except UnicodeDecodeError:
                return FileResponse(
                    path=str(path),
                    media_type=mime_type,
                    filename=path.name
                )
        
        return FileResponse(
            path=str(path),
            media_type=mime_type,
            filename=path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))
