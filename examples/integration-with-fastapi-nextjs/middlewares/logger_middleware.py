from functools import wraps
import time
import uuid
from typing import Callable

from fastapi import Request, Response, HTTPException
from .logger import app_logger as logger

async def log_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log API access with detailed information
    """
    # Generate unique request ID
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Add request ID to headers
    request.headers.__dict__["_list"].append(
        (b"x-request-id", request_id.encode())
    )

    # Log initial access
    logger.bind(
        request_id=request_id,
        method=request.method,
        original_url=str(request.url),
        client_ip=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("user-agent", "empty")
    )#.debug("ACCESS")

    try:
        response = await call_next(request)
        
                
        process_time = (time.time() - start_time) * 1000
        # Prepare log data
        log_data = {
            "request_id": request_id,
            "user_id": getattr(request.state, "user", {}).get("id", "unknown"),
            "method": request.method,
            "original_url": str(request.url),
            "status_code": response.status_code,
            "duration": round(process_time, 2),
            "status_msg": str(response.status_code),
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "empty")
        }



        # Create bound logger with the data
        bound_logger = logger.bind(**log_data)
        bound_logger.info("FINISH")
        
        # # Log based on response status
        # if response.status_code >= 500:
        #     bound_logger.error("FINISH !")
        # elif response.status_code >= 400:
        #    # For 4xx errors, attempt to log request body ONLY for specific content types
        #     if request.headers.get("content-type") and not request.headers.get("content-type").startswith("multipart/form-data"):
        #         try:
        #             body = await request.body()
        #             if body:
        #                 bound_logger = bound_logger.bind(request_body=body.decode())
        #         except Exception as e:
        #             logger.error(f"Failed to log request body: {str(e)}")
        #             raise
                
        #     bound_logger.warning("FINISH")
        # else:
        #     bound_logger.info("FINISH")

        return response
    except HTTPException as http_exc:
        # Handle HTTPException specifically
        logger.bind(
            request_id=request_id,
            method=request.method,
            original_url=str(request.url),
            status_code=http_exc.status_code,
            detail=http_exc.detail,
        ).warning("[logger] HTTP_EXCEPTION")
        
        raise http_exc
    except Exception as e:
        # Log unhandled exceptions
        logger.bind(
            request_id=request_id,
            method=request.method,
            original_url=str(request.url),
            error=str(e)
        ).error("[logger] ERROR")
        raise
