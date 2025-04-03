from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path
from sqlalchemy.orm import Session
from loguru import logger
from uuid import uuid4

from ..models import MessageConfig, MessageUpdateConfig
from ..middlewares.authenticate import require_auth
from ..database import get_db, QMessage, MessageRole

router = APIRouter(prefix="/api/agent", tags=["messages"])

@router.post("/messages")
async def create_message(
    config: MessageConfig,
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, bool]:
    """Create a new message and save to database."""
    logger.info("Creating new message",
        user_email=user.get('email'),
        user_id=user.get('id'),
        config=config
    )

    try:
        # Create database record
        db_message = QMessage(
            pid=config.id,
            content=config.content,
            role=MessageRole(config.role),
            conversation_id=config.conversation_id,
            prompt_id=config.prompt_id,
            events=config.events,
            message_metadata=config.message_metadata,
            tokens=config.tokens,
            loading_state=config.loading_state,
            feedback=config.feedback,
            user_id=user.get('id')  # Use authenticated user's ID
        )
        db.add(db_message)
        db.commit()
        db.refresh(db_message)
        
        logger.info(f"Message saved to database with ID: {db_message.pid}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to save message to database: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create message")

@router.get("/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: str = Path(..., title="The ID of the conversation to get messages for"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """List all messages for a conversation."""
    try:
        messages = db.query(QMessage).filter(
            QMessage.conversation_id == conversation_id,
            QMessage.user_id == user.get('id')
        ).order_by(QMessage.created_at.asc()).all()
        
        return [
            {
                "id": str(msg.pid),
                "content": msg.content,
                "role": msg.role.value,
                "conversation_id": str(msg.conversation_id),
                "prompt_id": str(msg.prompt_id) if msg.prompt_id else None,
                "events": msg.events,
                "message_metadata": msg.message_metadata,
                "tokens": msg.tokens,
                "loading_state": msg.loading_state,
                "feedback": msg.feedback,
                "created_at": msg.created_at,
                "updated_at": msg.updated_at
            }
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Failed to list messages: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list messages")

@router.get("/messages/{message_id}")
async def get_message(
    message_id: str = Path(..., title="The ID of the message to get"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get message by ID."""
    try:
        message = db.query(QMessage).filter(
            (QMessage.pid == message_id) | (QMessage.id == message_id),
            QMessage.user_id == user.get('id')
        ).first()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
            
        return {
            "id": str(message.pid),
            "content": message.content,
            "role": message.role.value,
            "conversation_id": str(message.conversation_id),
            "prompt_id": str(message.prompt_id) if message.prompt_id else None,
            "events": message.events,
            "message_metadata": message.message_metadata,
            "tokens": message.tokens,
            "loading_state": message.loading_state,
            "feedback": message.feedback,
            "created_at": message.created_at,
            "updated_at": message.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get message")

@router.patch("/messages/{message_id}")
async def update_message(
    update_data: MessageUpdateConfig,
    message_id: str = Path(..., title="The ID of the message to update"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update message partially."""
    try:
        message = db.query(QMessage).filter(
            (QMessage.pid == message_id) | (QMessage.id == message_id),
            QMessage.user_id == user.get('id')
        ).first()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        
        # Handle role conversion if present
        if "role" in update_dict:
            update_dict["role"] = MessageRole(update_dict["role"])
        
        for key, value in update_dict.items():
            setattr(message, key, value)
        
        db.commit()
        db.refresh(message)
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update message")

@router.delete("/messages/{message_id}")
async def delete_message(
    message_id: str = Path(..., title="The ID of the message to delete"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """Delete a message."""
    try:
        message = db.query(QMessage).filter(
            (QMessage.pid == message_id) | (QMessage.id == message_id),
            QMessage.user_id == user.get('id')
        ).first()
        
        if not message:
            raise HTTPException(status_code=404, detail=f"Message {message_id} not found")
        
        # Delete from database
        db.delete(message)
        db.commit()
        
        return {"message": f"Message {message_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete message: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete message")
