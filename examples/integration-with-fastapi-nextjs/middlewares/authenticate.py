from typing import Any, Dict, Optional
import json
from functools import wraps

from fastapi import HTTPException, Security, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from hkdf import Hkdf
from jose.jwe import decrypt
from .logger import app_logger as logger
import os

security = HTTPBearer()

def __encryption_key(secret: str) -> bytes:
    """Generate the encryption key using HKDF."""
    return Hkdf("", bytes(secret, "utf-8")).expand(b"NextAuth.js Generated Encryption Key", 32)

def decode_jwe(token: str, secret: str) -> Optional[Dict[str, Any]]:
    """Decrypt and decode a JWE token."""
    try:
        decrypted = decrypt(token, __encryption_key(secret))
        if decrypted:
            return json.loads(bytes.decode(decrypted, "utf-8"))
        return None
    except Exception as e:
        logger.error(f"Error decoding JWE token: {str(e)}")
        return None

async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """Get the current authenticated user from the token and set it in request.state."""
    request_id = request.headers.get("x-request-id", "unknown")
    
    try:
        jwt_secret = os.getenv("JWT_SECRET")
        
        if not jwt_secret:
            logger.error("JWT_SECRET not configured", request_id=request_id)
            raise HTTPException(
                status_code=500,
                detail="Server configuration error"
            )
        
        token = credentials.credentials
        
        payload = decode_jwe(token, jwt_secret)
        
        if not payload:
            logger.warning("Invalid or expired token", request_id=request_id)
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Set user in request.state for the logger middleware
        request.state.user = payload
        logger.debug("Authentication successful", 
                     request_id=request_id, 
                     user_id=payload.get('id', 'unknown'),
                     user_email=payload.get('email', 'unknown'))
        
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication error: {str(e)}")

# Use this as a dependency in FastAPI routes
require_auth = Depends(get_current_user)