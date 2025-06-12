import os
import logging
from pathlib import Path
from typing import Dict
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)

class FileSaveRequest(BaseModel):
    file_path: str
    content: str

@router.post("/files/save")
async def save_file_content(payload: FileSaveRequest) -> Dict:
    """Save content to a file at the specified path."""
    try:
        path = Path(payload.file_path)
        
        if not path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file path: {payload.file_path}. Path must be absolute."
            )
        
        # Create parent directories if they don't exist
        os.makedirs(path.parent, exist_ok=True)
        
        # Write content to file
        with open(path, 'w', encoding='utf-8') as f:
            f.write(payload.content)
        
        return {
            "status": "success",
            "path": str(path),
            "message": "File saved successfully",
            "size": path.stat().st_size,
            "filename": path.name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving file content: {e}")
        raise HTTPException(status_code=500, detail=str(e))
