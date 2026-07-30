import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.utils.config import settings
from app.services.pipeline import model_registry
from app.schemas.response import HealthResponse
from app.routers import vision, sentiment, chatbot, dashboard

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("smart_retail.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI startup & shutdown events.
    Loads all AI models into memory once at startup.
    """
    logger.info("==================================================")
    logger.info("  STARTING SMART RETAIL AI PLATFORM BACKEND       ")
    logger.info("==================================================")
    
    # Initialize ModelRegistry singleton (Loads all 7 model artifacts into memory)
    model_registry.initialize()
    
    yield # Server runs here
    
    logger.info("==================================================")
    logger.info("  SHUTTING DOWN SMART RETAIL AI PLATFORM BACKEND  ")
    logger.info("==================================================")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Production-Grade Smart Retail & Customer Intelligence AI API Platform",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable Cross-Origin Resource Sharing (CORS) for frontend UI integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Timing & Performance Middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time-Sec"] = f"{process_time:.4f}"
    return response

# Resolve static directory path dynamically
STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

# Root redirect to Web Dashboard UI
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/")

# Register Feature APIRouters under /api/v1
app.include_router(vision.router, prefix=settings.API_V1_STR)
app.include_router(sentiment.router, prefix=settings.API_V1_STR)
app.include_router(chatbot.router, prefix=settings.API_V1_STR)
app.include_router(dashboard.router, prefix=settings.API_V1_STR)

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System Health"],
    summary="Health check endpoint",
    description="Returns backend system health status and in-memory AI model load states."
)
async def health_check():
    models_status = {
        "product_classifier": model_registry.get_product_model() is not None,
        "sentiment_model": model_registry.get_sentiment_model() is not None,
        "sentiment_vectorizer": model_registry.get_sentiment_vectorizer() is not None,
        "chatbot_model": model_registry.get_chatbot_model() is not None,
        "chatbot_vectorizer": model_registry.get_chatbot_vectorizer() is not None,
        "face_database": len(model_registry.get_face_database()) > 0,
        "intents_dataset": len(model_registry.get_intents().get("intents", [])) > 0
    }
    return HealthResponse(
        message="Smart Retail AI Platform is operational.",
        version=settings.VERSION,
        models_loaded=models_status
    )
