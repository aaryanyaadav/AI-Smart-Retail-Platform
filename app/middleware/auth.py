from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.utils.config import settings

api_key_header = APIKeyHeader(name=settings.API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Middleware dependency to verify API Key authorization header.
    Expects 'X-API-Key' header matching configured settings.API_KEY.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing mandatory API Key header ('X-API-Key')."
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or unauthorized API Key."
        )
    return api_key
