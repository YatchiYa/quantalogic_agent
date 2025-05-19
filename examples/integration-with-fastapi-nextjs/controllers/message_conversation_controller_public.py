from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path
from sqlalchemy.orm import Session
from loguru import logger
from uuid import uuid4

from ..database import QConversation, get_db, QMessage

router = APIRouter(prefix="/api/agent/public", tags=["messages"])
 

@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str = Path(..., title="The ID of the conversation to get"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get conversation by ID."""
    try:
        conversation = db.query(QConversation).filter(
            (QConversation.pid == conversation_id) | (QConversation.id == conversation_id),
            QConversation.is_public == True
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


@router.get("/conversations/{conversation_id}/messages")
async def list_messages(
    conversation_id: str = Path(..., title="The ID of the conversation to get messages for"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """List all messages for a conversation."""
    try:
        messages = db.query(QMessage).filter(
            QMessage.conversation_id == conversation_id
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
