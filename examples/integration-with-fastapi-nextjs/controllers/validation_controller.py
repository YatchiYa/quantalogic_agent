import asyncio
import time
from fastapi import APIRouter, HTTPException
from loguru import logger
from ..models import UserValidationResponse
from ..agent_server import agent_state, server_state

router = APIRouter(prefix="/api/agent", tags=["validation"])

@router.post("/validation/{validation_id}")
async def submit_validation_response(validation_id: str, response: UserValidationResponse):
    """Submit a validation response."""
    start_time = time.time()
    logger.info(f"[{validation_id}] Processing validation response")
    
    with agent_state._validation_lock:
        response_queue = agent_state._validation_responses.get(validation_id)
        if not response_queue:
            logger.warning(f"[{validation_id}] No validation request found")
            raise HTTPException(status_code=404, detail="Validation request not found")
        
        if validation_id in agent_state._validation_timeouts:
            timeout_task = agent_state._validation_timeouts[validation_id]
            timeout_task.cancel()
            del agent_state._validation_timeouts[validation_id]
            logger.debug(f"[{validation_id}] Cancelled timeout task")
        
        if validation_id in agent_state._validation_requests:
            agent_state._validation_requests[validation_id]["status"] = "responded"
            try:
                response_queue.put_nowait(response.approved)
                elapsed = time.time() - start_time
                logger.info(f"[{validation_id}] Processed validation response in {elapsed:.2f} seconds")
                return {"status": "success"}
            except asyncio.QueueFull:
                elapsed = time.time() - start_time
                logger.warning(f"[{validation_id}] Response already processed after {elapsed:.2f} seconds")
                return {"status": "already_processed"}

    elapsed = time.time() - start_time
    logger.error(f"[{validation_id}] Failed to process validation after {elapsed:.2f} seconds")
    raise HTTPException(status_code=500, detail="Failed to process validation response")
