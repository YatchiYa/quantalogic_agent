from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Path
from sqlalchemy.orm import Session
from loguru import logger
from pydantic import BaseModel

from ..models import AgentConfig, ToolConfig, AgentUpdateConfig
from ..app_state import agent_state, server_state
from ..middlewares.authenticate import require_auth
from ..database import get_db, Agent

router = APIRouter(prefix="/api/agent", tags=["agents"])


def convert_tools_to_json(tools: List[Any]) -> List[Dict[str, Any]]:
    """Convert list of ToolConfig objects or dicts to JSON-serializable format."""
    if not tools:
        return None
    
    result = []
    for tool in tools:
        # If tool is already a dict, use it directly
        if isinstance(tool, dict):
            tool_dict = {
                "type": tool["type"],
                "parameters": tool["parameters"] if "parameters" in tool else None
            }
        # If tool is a ToolConfig object, convert it
        else:
            tool_dict = {
                "type": tool.type,
                "parameters": {
                    "model_name": tool.parameters.model_name,
                    "vision_model_name": tool.parameters.vision_model_name,
                    "provider": tool.parameters.provider,
                    "additional_info": tool.parameters.additional_info,
                    "connection_string": tool.parameters.connection_string,
                    "access_token": tool.parameters.access_token,
                    "auth_token": tool.parameters.auth_token
                } if tool.parameters else None
            }
        result.append(tool_dict)
    return result

@router.post("/agents")
async def create_agent(
    config: AgentConfig, 
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, bool]:
    """Create a new agent with the given configuration and save to database."""
    logger.info("Processing query request",
        user_email=user.get('email'),
        config=config
    )

    # Create agent in memory
    success = await agent_state.create_agent(config)

    print(config)
    
    if success:
        try:
            # Convert tools to JSON-serializable format
            tools_json = convert_tools_to_json(config.tools)
            
            # Create database record
            db_agent = Agent(
                name=config.name,
                description=config.description,
                model_name=config.model_name,
                expertise=config.expertise,
                pid=config.id,
                agent_mode=config.agent_mode,
                project=config.project,
                tags=config.tags,
                tools=tools_json,
                user_id=config.user_id,
                organization_id=config.organization_id
            )
            db.add(db_agent)
            db.commit()
            db.refresh(db_agent)
            
            logger.info(f"Agent {config.name} saved to database with ID: {db_agent.id}")
            return {"success": True}
        except Exception as e:
            logger.error(f"Failed to save agent to database: {str(e)}")
            return {"success": False}

    return {"success": success}

# @router.get("/agents/init_wide")
async def init_wide(
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """List all available agents."""
    try:
        agents = db.query(Agent).all()
        return [
            {
                "id": str(agent.pid),
                "name": agent.name,
                "description": agent.description,
                "model_name": agent.model_name,
                "expertise": agent.expertise,
                "project": agent.project,
                "agent_mode": agent.agent_mode,
                "tags": agent.tags,
                "tools": agent.tools,
                "created_at": agent.created_at,
                "updated_at": agent.updated_at
            }
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@router.get("/agents")
async def list_agents(
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """List all available agents for the user."""
    try:
        agents = db.query(Agent).filter(
            Agent.user_id == user.get('id')
        ).all()
        return [
            {
                "id": str(agent.pid),
                "name": agent.name,
                "description": agent.description,
                "model_name": agent.model_name,
                "expertise": agent.expertise,
                "project": agent.project,
                "agent_mode": agent.agent_mode,
                "tags": agent.tags,
                "tools": agent.tools,
                "created_at": agent.created_at,
                "updated_at": agent.updated_at
            }
            for agent in agents
        ]
    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

@router.get("/agents/{agent_id}")
async def get_agent(
    agent_id: str = Path(..., title="The ID of the agent to get"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get agent configuration by ID."""
    try:
        agent = db.query(Agent).filter(
            Agent.pid == agent_id,
            Agent.user_id == user.get('id')
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            
        return {
            "id": str(agent.pid),
            "name": agent.name,
            "description": agent.description,
            "model_name": agent.model_name,
            "expertise": agent.expertise,
            "project": agent.project,
            "agent_mode": agent.agent_mode,
            "tags": agent.tags,
            "tools": agent.tools,
            "created_at": agent.created_at,
            "updated_at": agent.updated_at
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get agent")

@router.patch("/agents/{agent_id}")
async def update_agent(
    update_data: AgentUpdateConfig,
    agent_id: str = Path(..., title="The ID of the agent to update"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update agent configuration partially."""
    try:
        agent = db.query(Agent).filter(
            Agent.pid == agent_id,
            Agent.user_id == user.get('id')
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Update only provided fields
        update_dict = update_data.dict(exclude_unset=True)
        
        # Handle tools conversion if present
        if "tools" in update_dict:
            update_dict["tools"] = convert_tools_to_json(update_dict["tools"])
        
        for key, value in update_dict.items():
            setattr(agent, key, value)
        
        db.commit()
        db.refresh(agent)
        
        # Update in-memory agent if it exists
        if agent_state.get_agent_config(agent_id):
            await agent_state.update_agent_config(agent_id, update_dict)
        
        return {"success": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to update agent")

@router.delete("/agents/{agent_id}")
async def delete_agent(
    agent_id: str = Path(..., title="The ID of the agent to delete"),
    user: Dict[str, Any] = require_auth,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """Delete an agent."""
    try:
        agent = db.query(Agent).filter(
            Agent.pid == agent_id,
            Agent.user_id == user.get('id')
        ).first()
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Delete from database
        db.delete(agent)
        db.commit()
        
        # Delete from in-memory state if it exists
        if agent_state.get_agent_config(agent_id):
            await agent_state.delete_agent(agent_id)
        
        return {"message": f"Agent {agent_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete agent: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to delete agent")
