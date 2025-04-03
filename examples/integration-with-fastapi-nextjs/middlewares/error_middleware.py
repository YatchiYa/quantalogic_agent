from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing import Callable, Dict, Any, Optional, Union, Type
from pydantic import BaseModel
import traceback
import json
import time
from datetime import datetime
from .logger import app_logger as logger


# class ErrorDetail(BaseModel):
#     """Schema for detailed error information"""
#     status_msg: str
#     status_code: int
#     original_url: str
#     method: str
#     exception_name: str
#     timestamp: str
#     detailed_message: Optional[str] = None
#     request_body: Optional[Dict[str, Any]] = None
#     request_query: Optional[Dict[str, Any]] = None
#     request_params: Optional[Dict[str, Any]] = None
#     stack: Optional[str] = None
#     payload: Optional[Dict[str, Any]] = None


# async def error_middleware(request: Request, call_next: Callable) -> Response:
#     """
#     Middleware to handle exceptions and log them in a consistent format
#     """
#     request_id = request.headers.get("x-request-id", "unknown")
#     user_id = getattr(request.state, "user", {}).get("id", "unknown")
#     start_time = time.time()
    
#     try:
#         response = await call_next(request)
#         return response
#     except Exception as exc:
#         # Capture exception details
#         status_code = 500
#         detail = str(exc)
#         detailed_message = str(exc)
        
#         # Determine status code for different exception types
#         if isinstance(exc, HTTPException) or isinstance(exc, StarletteHTTPException):
#             status_code = exc.status_code
#             detail = getattr(exc, "detail", str(exc))
#             detailed_message = detail  # For HTTPExceptions, use detail as the detailed message
#         elif isinstance(exc, RequestValidationError):
#             status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
#             detail = str(exc)
#             detailed_message = detail
#         else:
#             detailed_message = str(exc)
        
#         # Attempt to get request body (will only work for certain content types)
#         request_body = None
#         request_query = None
#         request_params = None
        
#         try:
#             if request.headers.get("content-type") and not request.headers.get("content-type").startswith("multipart/form-data"):
#                 request_body = await request.json()
#         except Exception:
#             # Could not parse JSON body, not critical
#             pass
        
#         # Get query parameters
#         try:
#             request_query = dict(request.query_params)
#         except Exception:
#             pass
        
#         # Get path parameters
#         try:
#             request_params = dict(request.path_params)
#         except Exception:
#             pass
        
#         # Extract any additional payload from the exception
#         payload = None
#         if hasattr(exc, 'payload'):
#             payload = exc.payload
        
#         # Create detailed error
#         detailed_error = {
#             "status_msg": detail,
#             "status_code": status_code,
#             "original_url": str(request.url),
#             "method": request.method,
#             "exception_name": type(exc).__name__,
#             "timestamp": datetime.now().isoformat(),
#             "detailed_message": detailed_message,  
#             "request_body": request_body,
#             "request_query": request_query,
#             "request_params": request_params,
#             "stack": traceback.format_exc(),
#             "payload": payload,
#         }
        
#         # Log based on status code
#         process_time = (time.time() - start_time) * 1000
#         log_data = {
#             "request_id": request_id,
#             "user_id": user_id,
#             "duration": round(process_time, 2),
#             **detailed_error
#         }
        
#         if status_code >= 500:
#             logger.error("REQUEST ERROR", **log_data)
#         else:
#             logger.warning("REQUEST ERROR", **log_data)
        
#         # Return JSON response with error
#         return JSONResponse(
#             status_code=status_code,
#             # content={"detail": detail, "error_details": detailed_error}
#         )


def register_exception_handlers(app):
    """
    Register custom exception handlers for FastAPI app
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # Log the HTTP exception
        request_id = request.headers.get("x-request-id", "unknown")
        user_id = getattr(request.state, "user", {}).get("id", "unknown")
        
        # For HTTPException, the detail is the detailed message
        detailed_message = exc.detail
                
        # Get any additional payload
        payload = None
        if hasattr(exc, 'payload'):
            payload = exc.payload
        
        detailed_error = {
            "status_msg": exc.detail,  # Ensure we use exc.detail directly
            "status_code": exc.status_code,
            "original_url": str(request.url),
            "method": request.method,
            "exception_name": "HTTPException",
            "timestamp": datetime.now().isoformat(),
            "detailed_message": exc.detail,  # This should contain exc.detail
            "stack": traceback.format_exc(),
            "payload": payload,
        }
        
        if exc.status_code >= 500:
            logger.error("HTTP EXCEPTION", request_id=request_id, user_id=user_id, **detailed_error)
        else:
            logger.warning("HTTP EXCEPTION", request_id=request_id, user_id=user_id, **detailed_error)
        
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "error_details": detailed_error}
        )
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # Log the validation exception
        request_id = request.headers.get("x-request-id", "unknown")
        user_id = getattr(request.state, "user", {}).get("id", "unknown")
        
        # For validation errors, the string representation is the detailed message
        detailed_message = str(exc)
        
        detailed_error = {
            "status_msg": str(exc),
            "status_code": status.HTTP_422_UNPROCESSABLE_ENTITY,
            "original_url": str(request.url),
            "method": request.method,
            "exception_name": "RequestValidationError",
            "timestamp": datetime.now().isoformat(),
            "detailed_message": detailed_message,
            "stack": traceback.format_exc(),
            "validation_errors": exc.errors()
        }
        
        logger.warning("VALIDATION ERROR", request_id=request_id, user_id=user_id, **detailed_error)
        
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": exc.errors(), "error_details": detailed_error}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        # Log the general exception
        request_id = request.headers.get("x-request-id", "unknown")
        user_id = getattr(request.state, "user", {}).get("id", "unknown")
        
        # Get detailed message
        detailed_message = str(exc)
        
        # Get any additional payload
        payload = None
        if hasattr(exc, 'payload'):
            payload = exc.payload
        
        detailed_error = {
            "status_msg": str(exc),
            "status_code": 500,
            "original_url": str(request.url),
            "method": request.method,
            "exception_name": type(exc).__name__,
            "timestamp": datetime.now().isoformat(),
            "detailed_message": detailed_message,
            "stack": traceback.format_exc(),
            "payload": payload
        }
        
        logger.error("UNCAUGHT EXCEPTION", request_id=request_id, user_id=user_id, **detailed_error)
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error", "error_details": detailed_error}
        )
