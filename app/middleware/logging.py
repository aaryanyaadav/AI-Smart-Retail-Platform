import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("smart_retail.audit_logger")

class AuditLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware that records HTTP audit logs for all incoming API requests and outgoing responses.
    """
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        client_ip = request.client.host if request.client else "127.0.0.1"
        method = request.method
        url_path = request.url.path

        logger.info(f"Incoming Request: {method} {url_path} from IP {client_ip}")

        try:
            response = await call_next(request)
            process_time_ms = (time.time() - start_time) * 1000
            status_code = response.status_code

            logger.info(
                f"Completed Request: {method} {url_path} | Status: {status_code} | Duration: {process_time_ms:.2f}ms"
            )
            return response
        except Exception as exc:
            process_time_ms = (time.time() - start_time) * 1000
            logger.error(
                f"Failed Request: {method} {url_path} | Error: {str(exc)} | Duration: {process_time_ms:.2f}ms"
            )
            raise exc
