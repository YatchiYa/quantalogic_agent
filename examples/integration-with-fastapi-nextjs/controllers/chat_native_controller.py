"""Chat controller for the QuantaLogic API."""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import litellm

from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.memory import AgentMemory
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.event_emitter import EventEmitter
from ..prompts.legal_system_prompt import LEGAL_SYSTEM_PROMPT

router = APIRouter(prefix="/api/agent/chat_test", tags=["chat"])

# Store chat sessions and their memories
chat_sessions: Dict[str, AgentMemory] = {}
# Store system prompts for each session
system_prompts: Dict[str, str] = {}
# Store chat models
chat_models: Dict[str, GenerativeModel] = {}
# Track active streaming sessions
active_streams: Dict[str, bool] = {}

# Define LinkupTool as a function for LiteLLM
def perform_web_search(query: str, depth: str = "standard", output_type: str = "sourcedAnswer") -> str:
    """Perform a web search using the Linkup API."""
    try:
        tool = LinkupTool()
        result = tool.execute(query=query, depth=depth, output_type=output_type)
        logger.info(f"Web search result: {result}")
        return result
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        return f"Error performing web search: {str(e)}"

# Define tools for LiteLLM
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "perform_web_search",
            "description": "Search the web for current information using Linkup API",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to perform",
                    },
                    "depth": {
                        "type": "string",
                        "enum": ["standard", "deep"],
                        "description": "Search depth (standard or deep)",
                    },
                    "output_type": {
                        "type": "string",
                        "enum": ["searchResults", "sourcedAnswer"],
                        "description": "Type of output (searchResults or sourcedAnswer)",
                    },
                },
                "required": ["query"],
            },
        },
    }
]

class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="The message to send")
    session_id: str = Field(..., description="Unique session identifier")
    model: str = Field(default="gpt-3.5-turbo-1106", description="Model to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    web_search: bool = Field(default=True, description="Whether to perform web search")
    search_depth: str = Field(default="standard", description="Search depth (standard or deep)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    system_prompt: Optional[str] = Field(default=None, description="System prompt to set the assistant's behavior")

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="The model's response")
    session_id: str = Field(..., description="Session identifier")
    sources: Optional[List[str]] = Field(default=None, description="Web sources if web search was used")

def get_or_create_memory(session_id: str) -> AgentMemory:
    """Get or create memory for a session."""
    if session_id not in chat_sessions:
        chat_sessions[session_id] = AgentMemory()
    return chat_sessions[session_id]

def update_system_prompt(session_id: str, new_prompt: Optional[str]) -> None:
    """Update system prompt for a session if it has changed."""
    current_prompt = system_prompts.get(session_id)
    if new_prompt != current_prompt:
        system_prompts[session_id] = new_prompt
        # Clear memory if system prompt changes
        if session_id in chat_sessions:
            chat_sessions[session_id] = AgentMemory()

def track_cost_callback(kwargs, completion_response, start_time, end_time):
    """Track cost of LiteLLM API calls."""
    try:
        response_cost = kwargs.get("response_cost", 0)
        logger.info(f"API call cost: {response_cost}")
    except Exception as e:
        logger.error(f"Error tracking cost: {str(e)}")

# Set LiteLLM callback
litellm.success_callback = [track_cost_callback]

async def stream_response(response_iter, session_id=None, user_message=None):
    """Stream response chunks and update memory when done."""
    logger.info(f"Streaming response for session {session_id}")
    logger.info(f"User message: {user_message}")
    full_response = ""
    try:
        # Register this stream as active if we have a session_id
        if session_id:
            active_streams[session_id] = True
            logger.info(f"Started streaming for session {session_id}")
            
        if hasattr(response_iter, '__aiter__'):  # Check if it's an async iterator
            async for chunk in response_iter:
                # Check if streaming should be stopped
                if session_id and not active_streams.get(session_id, True):
                    logger.info(f"Streaming cancelled for session {session_id}")
                    break
                    
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
        else:  # Handle sync iterator
            for chunk in response_iter:
                # Check if streaming should be stopped
                if session_id and not active_streams.get(session_id, True):
                    logger.info(f"Streaming cancelled for session {session_id}")
                    break
                    
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Clean up active stream tracking
        if session_id and session_id in active_streams:
            del active_streams[session_id]
            logger.info(f"Finished streaming for session {session_id}")
            
        # Update memory if session_id and user_message are provided
        if session_id and user_message and full_response:
            try:
                memory = chat_sessions.get(session_id)
                if memory:
                    memory.add(Message(role="user", content=user_message))
                    memory.add(Message(role="assistant", content=full_response))
                    
                    # Compact memory if needed
                    if len(memory.memory) > 10:
                        memory.compact(n=2)
                    logger.info(f"Updated memory for streaming session {session_id}")
            except Exception as e:
                logger.error(f"Error updating memory after streaming: {str(e)}")
        
        # Send a cancelled message if streaming was stopped
        if session_id and not active_streams.get(session_id, True):
            yield f"data: {json.dumps({'cancelled': True})}\n\n"
            
        yield "data: [DONE]\n\n"

@router.post("/send")
async def send_message(request: ChatRequest):
    """Send a message to the chat model.""" 
    logger.info(f"Sending message: {request.message}")
    try:
        # If no system prompt is provided, use the default legal system prompt
        if request.system_prompt is None:
            logger.info("No system prompt provided, using default legal system prompt")
            request.system_prompt = LEGAL_SYSTEM_PROMPT
            
        # Update system prompt and get memory
        update_system_prompt(request.session_id, request.system_prompt)
        memory = get_or_create_memory(request.session_id)
        
        # Prepare messages for LiteLLM
        messages = []
        
        # Add system prompt if it exists for this session
        if system_prompt := system_prompts.get(request.session_id):
            messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history
        messages.extend([{"role": msg.role, "content": msg.content} for msg in memory.memory])
        messages.append({"role": "user", "content": request.message})
        
        sources = []
        # Call LiteLLM with function calling
        if request.web_search:
            try:
                # First call to check if we need to search
                response = await litellm.acompletion(
                    model=request.model,
                    messages=messages,
                    temperature=request.temperature,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream_options={"include_usage": True}
                )
                
                # Process response
                response_message = response.choices[0].message
                
                # Handle tool calls if present
                if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                    for tool_call in response_message.tool_calls:
                        if tool_call.function.name == "perform_web_search": 
                            args = json.loads(tool_call.function.arguments)
                            args['depth'] = request.search_depth
                            search_result = perform_web_search(**args)
                            sources.append(search_result)
                            # Add search results to messages for follow-up
                            messages.append({
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [tool_call]
                            })
                            
                            # Convert search_result to string if it's not already
                            if not isinstance(search_result, str):
                                search_result = json.dumps(search_result)
                                
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": search_result
                            })
            except Exception as e:
                logger.error(f"Error in web search: {str(e)}")
                # Continue without web search if it fails
        
        # Final response (streaming or not)
        if request.stream:
            # Stream the response
            response_iter = await litellm.acompletion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            return StreamingResponse(
                stream_response(response_iter, session_id=request.session_id, user_message=request.message),
                media_type="text/event-stream"
            )
        else:
            # Get final response
            final_response = await litellm.acompletion(
                model=request.model,
                messages=messages,
                temperature=request.temperature,
                stream_options={"include_usage": True}
            )
            response_content = final_response.choices[0].message.content
            
            # Add messages to memory
            memory.add(Message(role="user", content=request.message))
            memory.add(Message(role="assistant", content=response_content))
            
            # Compact memory if needed
            if len(memory.memory) > 10:
                memory.compact(n=2)
            
            return ChatResponse(
                response=response_content,
                session_id=request.session_id,
                sources=sources if sources else None
            )
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End a chat session and clean up resources."""
    try:
        if session_id in chat_sessions:
            del chat_sessions[session_id]
        if session_id in system_prompts:
            del system_prompts[session_id]
        if session_id in chat_models:
            del chat_models[session_id]
        return {"status": "success", "message": "Session ended"}
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Get the chat history for a session."""
    try:
        if session_id not in chat_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        memory = chat_sessions[session_id]
        return [{"role": msg.role, "content": msg.content} for msg in memory.memory]
    except Exception as e:
        logger.error(f"Error getting session history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop/{session_id}")
async def stop_streaming(session_id: str):
    """Stop an ongoing streaming response for a session."""
    try:
        if session_id in active_streams:
            active_streams[session_id] = False
            logger.info(f"Requested to stop streaming for session {session_id}")
            return {"status": "success", "message": f"Streaming for session {session_id} will be stopped"}
        else:
            logger.info(f"No active streaming found for session {session_id}")
            return {"status": "not_found", "message": f"No active streaming found for session {session_id}"}
    except Exception as e:
        logger.error(f"Error stopping streaming: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
