"""Chat controller for the QuantaLogic API."""

import asyncio
import json
from typing import Dict, List, Optional, AsyncGenerator, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger
import litellm
from litellm import completion_cost, token_counter

from quantalogic.generative_model import GenerativeModel, Message
from quantalogic.memory import AgentMemory
from quantalogic.tools.linkup_tool import LinkupTool
from quantalogic.event_emitter import EventEmitter
from ..prompts.legal_system_prompt import LEGAL_SYSTEM_PROMPT
from ..prompts.agent_system_prompt import AGENT_SYSTEM_PROMPT
from ..prompts.document_system_prompt import DOCUMENT_SYSTEM_PROMPT

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
    provider: Optional[str] = Field(default="openai", description="Provider to use")
    temperature: float = Field(default=0.7, description="Temperature for generation")
    web_search: bool = Field(default=True, description="Whether to perform web search")
    search_depth: str = Field(default="standard", description="Search depth (standard or deep)")
    stream: bool = Field(default=False, description="Whether to stream the response")
    system_prompt: Optional[str] = Field(default=None, description="System prompt to set the assistant's behavior")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Optional history of messages to initialize the memory with, format: [{'role': 'user'|'assistant', 'content': 'message'}]")
    persona_mode: Optional[str] = Field(default=None, description="Persona mode to set the assistant's behavior")

class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="The model's response")
    session_id: str = Field(..., description="Session identifier")
    sources: Optional[List[str]] = Field(default=None, description="Web sources if web search was used")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="Token usage and cost information")

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
    """Track cost of LiteLLM API calls using the built-in LiteLLM functions."""
    try:
        # Get the model name from kwargs if available
        model_name = kwargs.get("model", "default")
        
        # Extract messages from kwargs for token counting
        messages = kwargs.get("messages", [])
        
        # Calculate prompt tokens using LiteLLM's token_counter
        prompt_tokens = token_counter(model=model_name, messages=messages)
        
        # Get assistant response content for token counting
        assistant_content = ""
        if hasattr(completion_response, 'choices') and len(completion_response.choices) > 0:
            if hasattr(completion_response.choices[0], 'message'):
                assistant_content = completion_response.choices[0].message.content
        
        # Calculate completion tokens using LiteLLM's token_counter
        completion_tokens = 0
        if assistant_content:
            completion_tokens = token_counter(model=model_name, messages=[{"role": "assistant", "content": assistant_content}])
        
        # Calculate total tokens
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost using LiteLLM's completion_cost
        response_cost = completion_cost(completion_response=completion_response)
        
        logger.info(f"API call cost: ${response_cost:.10f}, Tokens: {prompt_tokens}/{completion_tokens}/{total_tokens} (prompt/completion/total)")
        
        # Store usage data in the kwargs so it can be accessed by the calling function
        usage_data = {
            "cost": response_cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "timestamp": end_time.isoformat() if end_time else None
        }
        kwargs["usage_data"] = usage_data
        
        # Store the last usage data for access by other functions
        litellm._last_usage_data = usage_data
    except Exception as e:
        logger.error(f"Error tracking cost: {str(e)}")

# Set LiteLLM callback
litellm.success_callback = [track_cost_callback]

# Add a place to store the last usage data
litellm._last_usage_data = {}

def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Estimate the number of tokens in a text string using LiteLLM's token_counter.
    
    Args:
        text: The text to estimate tokens for
        model: The model to use for token counting
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
        
    # Use litellm's token_counter for accurate token counting
    return token_counter(model=model, messages=[{"role": "user", "content": text}])

def log_token_usage(session_id: str, model: str, usage_data: Dict[str, Any]):
    """Log detailed token usage and cost information."""
    prompt_tokens = usage_data.get("prompt_tokens", 0)
    completion_tokens = usage_data.get("completion_tokens", 0)
    total_tokens = usage_data.get("total_tokens", 0)
    cost = usage_data.get("cost", 0.0)
    
    logger.info(f"===== TOKEN USAGE REPORT =====")
    logger.info(f"Session: {session_id}")
    logger.info(f"Model: {model}")
    logger.info(f"Prompt tokens: {prompt_tokens}")
    logger.info(f"Completion tokens: {completion_tokens}")
    logger.info(f"Total tokens: {total_tokens}")
    logger.info(f"Cost: ${cost:.6f}")
    logger.info(f"==============================")

async def stream_response(response_iter, session_id=None, user_message=None, model=None, messages=None):
    """Stream response chunks and update memory when done."""
    logger.info(f"Streaming response for session {session_id}")
    logger.info(f"User message: {user_message}")
    full_response = ""
    # Track tokens for streaming responses
    completion_tokens = 0
    # Store the messages for token counting
    original_messages = messages or []
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
                        # No need to increment token count here as we'll estimate at the end
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
                        # No need to increment token count here as we'll estimate at the end
                        yield f"data: {json.dumps({'content': content})}\n\n"
    except Exception as e:
        logger.error(f"Error in stream_response: {str(e)}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Clean up active stream tracking
        if session_id and session_id in active_streams:
            del active_streams[session_id]
            # We'll log the token count after we've calculated it properly
            
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
            
        # Use LiteLLM's token_counter for the full response
        if full_response:
            # Get model name from session
            model_name = model
            
            # Calculate completion tokens using LiteLLM's token_counter
            completion_tokens = token_counter(model=model_name, messages=[{"role": "assistant", "content": full_response}])
            
            # Calculate prompt tokens directly from the original messages
            prompt_tokens = 0
            cost = 0.0
            try:
                # First try to get prompt tokens from the original messages
                if original_messages:
                    prompt_tokens = token_counter(model=model_name, messages=original_messages)
                    logger.info(f"Calculated prompt tokens directly: {prompt_tokens}")
                
                # If we couldn't get prompt tokens from original messages, try the callback data
                if prompt_tokens == 0:
                    callback_data = getattr(litellm, "_last_usage_data", {})
                    if callback_data and "prompt_tokens" in callback_data:
                        prompt_tokens = callback_data["prompt_tokens"]
                        logger.info(f"Got prompt tokens from callback: {prompt_tokens}")
                
                # If we still don't have prompt tokens and we have a user message, estimate from that
                if prompt_tokens == 0 and user_message:
                    # Get memory for this session if available
                    memory = chat_sessions.get(session_id)
                    if memory:
                        # Create messages from memory
                        memory_messages = [{"role": msg.role, "content": msg.content} for msg in memory.memory]
                        # Add the current user message
                        memory_messages.append({"role": "user", "content": user_message})
                        # Calculate tokens
                        prompt_tokens = token_counter(model=model_name, messages=memory_messages)
                        logger.info(f"Calculated prompt tokens from memory: {prompt_tokens}")
                
                # Create a mock completion response to calculate cost
                mock_response = {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    },
                    "model": model_name
                }
                
                # Calculate cost using LiteLLM's completion_cost
                cost = completion_cost(completion_response=mock_response)
                logger.info(f"Calculated streaming cost using LiteLLM: ${cost:.10f}")
            except Exception as e:
                logger.error(f"Error calculating cost with LiteLLM for streaming: {str(e)}")
            
            # Send final usage data before completing
            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "cost": cost
            }
            
            logger.info(f"Streaming response completed for session {session_id}")
            logger.info(f"Token usage: {prompt_tokens} prompt, {completion_tokens} completion, {prompt_tokens + completion_tokens} total, Cost: ${cost:.10f}")
            
            # Make sure the usage data is sent as a separate event
            yield f"data: {json.dumps({'usage': usage_data, 'model': model_name})}\n\n"
        else:
            # No response generated
            yield f"data: {json.dumps({'usage': {'completion_tokens': 0, 'total_tokens': 0, 'cost': 0.0}})}\n\n"
        yield "data: [DONE]\n\n"

def get_prompt_by_persona_mode(persona_mode: str):
    if persona_mode == "legal":
        return LEGAL_SYSTEM_PROMPT
    elif persona_mode == "agent":
        return AGENT_SYSTEM_PROMPT
    elif persona_mode == "document":
        return DOCUMENT_SYSTEM_PROMPT
    else:
        return AGENT_SYSTEM_PROMPT

@router.post("/send")
async def send_message(request: ChatRequest):
    """Send a message to the chat model.""" 
    logger.info(f"Sending message: {request.message}")
    try:
        if request.persona_mode != "custom":
            request.system_prompt = get_prompt_by_persona_mode(request.persona_mode)
            
        # Update system prompt and get memory
        update_system_prompt(request.session_id, request.system_prompt)
        memory = get_or_create_memory(request.session_id)
        
        # Initialize memory with history if provided, but only if it doesn't exist yet
        if request.history and request.session_id not in chat_sessions:
            logger.info(f"Initializing memory with provided history for session {request.session_id}")
            # Create new memory instance
            chat_sessions[request.session_id] = AgentMemory()
            memory = chat_sessions[request.session_id]
            # Add history messages to memory
            for msg in request.history:
                if 'role' in msg and 'content' in msg:
                    memory.add(Message(role=msg['role'], content=msg['content']))
        
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
                    # model_provider=request.provider,
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
                # model_provider=request.provider,
                messages=messages,
                temperature=request.temperature,
                stream=True,
                stream_options={"include_usage": True}
            )
            
            # Create streaming response
            return StreamingResponse(
                stream_response(
                    response_iter, 
                    session_id=request.session_id, 
                    user_message=request.message, 
                    model=request.model,
                    messages=messages  # Pass the messages for token counting
                ),
                media_type="text/event-stream"
            )
        else:
            # Get final response
            final_response = await litellm.acompletion(
                model=request.model,
                # model_provider=request.provider,
                messages=messages,
                temperature=request.temperature,
                stream_options={"include_usage": True}
            )
            response_content = final_response.choices[0].message.content
            
            # Extract usage information
            usage_data = {}
            if hasattr(final_response, 'usage'):
                usage_data = {
                    "prompt_tokens": final_response.usage.prompt_tokens,
                    "completion_tokens": final_response.usage.completion_tokens,
                    "total_tokens": final_response.usage.total_tokens,
                    "cost": 0.0  # Will be updated by callback if available
                }
                logger.debug(f"================================  Final response usage: {usage_data}")
                
            # Get cost from the LiteLLM callback
            try:
                # Get the most recent callback data for this model
                callback_data = getattr(litellm, "_last_usage_data", {})
                logger.debug(f"LiteLLM callback data: {callback_data}")
                
                if callback_data and "cost" in callback_data:
                    usage_data["cost"] = callback_data["cost"]
                    logger.info(f"Cost from LiteLLM callback: ${callback_data['cost']:.6f}")
                else:
                    # If no cost data available from callback, calculate it using our pricing module
                    if "prompt_tokens" in usage_data and "completion_tokens" in usage_data:
                        prompt_tokens = usage_data["prompt_tokens"]
                        completion_tokens = usage_data["completion_tokens"]
                        usage_data["cost"] = calculate_cost(request.model, prompt_tokens, completion_tokens)
                        logger.info(f"Calculated cost using pricing module: ${usage_data['cost']:.6f}")
                    else:
                        usage_data["cost"] = 0.0
                        logger.warning("No token data available to calculate cost")
            except Exception as e:
                logger.error(f"Error getting cost data from LiteLLM: {str(e)}")
                usage_data["cost"] = 0.0
                
            # Log detailed usage information
            log_token_usage(request.session_id, request.model, usage_data)
            
            # Add messages to memory
            memory.add(Message(role="user", content=request.message))
            memory.add(Message(role="assistant", content=response_content))
            
            # Compact memory if needed
            if len(memory.memory) > 10:
                memory.compact(n=2)
            
            return ChatResponse(
                response=response_content,
                session_id=request.session_id,
                sources=sources if sources else None,
                usage=usage_data
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
