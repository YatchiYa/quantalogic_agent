from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path
from sqlalchemy.orm import Session
from loguru import logger
from uuid import uuid4

from ..models import ConversationConfig, ConversationUpdateConfig
from ..middlewares.authenticate import require_auth
from ..database import get_db, QConversation

router = APIRouter(prefix="/api/agent", tags=["conversations"])

@router.post("/conversations")
async def create_conversation(
    config: ConversationConfig,
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, bool]:
    """Create a new conversation and save to database."""
    logger.info("Creating new conversation",
        user_email=user.get('email'),
        user_id=user.get('id'),
        config=config
    )

    try:
        # Create database record
        db_conversation = QConversation(
            pid=config.id,
            title=config.title,
            description=config.description,
            model_id=config.model_id,
            project=config.project,
            user_id=user.get('id'),  # Use authenticated user's ID
            organization_id=config.organization_id,
            agent_id=config.agent_id,
            is_public=config.is_public,
            is_archived=config.is_archived,
            is_favorite=config.is_favorite
        )
        db.add(db_conversation)
        db.commit()
        db.refresh(db_conversation)
        
        logger.info(f"Conversation saved to database with ID: {db_conversation.pid}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Failed to save conversation to database: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create conversation")

@router.get("/conversations")
async def list_conversations(
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """List all available conversations for the user."""
    try:
        conversations = db.query(QConversation).filter(
            QConversation.user_id == user.get('id')
        ).order_by(QConversation.created_at.desc()).limit(10).all()
        return [
            {
                "id": str(conv.pid),
                "title": conv.title,
                "description": conv.description,
                "model_id": conv.model_id,
                "project": conv.project,
                "agent_id": str(conv.agent_id) if conv.agent_id else None,
                "is_public": conv.is_public,
                "is_archived": conv.is_archived,
                "is_favorite": conv.is_favorite,
                "last_message_at": conv.last_message_at,
                "created_at": conv.created_at,
                "updated_at": conv.updated_at
            }
            for conv in conversations
        ]
    except Exception as e:
        logger.error(f"Failed to list conversations: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list conversations")

@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str = Path(..., title="The ID of the conversation to get"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get conversation by ID."""
    try:
        conversation = db.query(QConversation).filter(
            (QConversation.pid == conversation_id) | (QConversation.id == conversation_id),
            QConversation.user_id == user.get('id')
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
            
        return {
            "id": str(conversation.pid),
            "title": conversation.title,
            "description": conversation.description,
            "model_id": conversation.model_id,
            "project": conversation.project,
            "agent_id": str(conversation.agent_id) if conversation.agent_id else None,
            "is_public": conversation.is_public,
            "is_archived": conversation.is_archived,
            "is_favorite": conversation.is_favorite,
            "last_message_at": conversation.last_message_at,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get conversation")

@router.patch("/conversations/{conversation_id}")
async def update_conversation(
    update_data: ConversationUpdateConfig,
    conversation_id: str = Path(..., title="The ID of the conversation to update"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update conversation partially."""
    try:
        conversation = db.query(QConversation).filter(
            (QConversation.pid == conversation_id) | (QConversation.id == conversation_id),
            QConversation.user_id == user.get('id')
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        
        for key, value in update_dict.items():
            setattr(conversation, key, value)
        
        db.commit()
        db.refresh(conversation)
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update conversation")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str = Path(..., title="The ID of the conversation to delete"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """Delete a conversation."""
    try:
        conversation = db.query(QConversation).filter(
            (QConversation.pid == conversation_id) | (QConversation.id == conversation_id),
            QConversation.user_id == user.get('id')
        ).first()
        
        if not conversation:
            raise HTTPException(status_code=404, detail=f"Conversation {conversation_id} not found")
        
        # Delete from database
        db.delete(conversation)
        db.commit()
        
        return {"message": f"Conversation {conversation_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete conversation: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete conversation")
