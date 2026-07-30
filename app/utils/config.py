import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Root directory of the repository (parent of app/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "Smart Retail AI Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Security & Authentication
    API_KEY_NAME: str = "X-API-Key"
    API_KEY: str = os.getenv("SMART_RETAIL_API_KEY", "smart-retail-secret-api-key-2026")
    
    # Model Artifact Paths
    MODELS_DIR: Path = BASE_DIR / "models"
    PRODUCT_MODEL_PATH: Path = MODELS_DIR / "product_classifier.h5"
    SENTIMENT_MODEL_PATH: Path = MODELS_DIR / "sentiment_model.pkl"
    SENTIMENT_VECTORIZER_PATH: Path = MODELS_DIR / "sentiment_vectorizer.pkl"
    CHATBOT_MODEL_PATH: Path = MODELS_DIR / "chatbot_model.pkl"
    CHATBOT_VECTORIZER_PATH: Path = MODELS_DIR / "chatbot_vectorizer.pkl"
    FACE_DB_PATH: Path = MODELS_DIR / "face_db.pkl"
    INTENTS_JSON_PATH: Path = MODELS_DIR / "intents.json"
    
    # Log File Paths
    LOGS_DIR: Path = BASE_DIR / "app" / "logs"
    VISITS_LOG_PATH: Path = LOGS_DIR / "visits.csv"
    SENTIMENT_LOG_PATH: Path = LOGS_DIR / "sentiment_logs.csv"
    CHATBOT_LOG_PATH: Path = LOGS_DIR / "chatbot_logs.csv"
    PRODUCT_LOG_PATH: Path = LOGS_DIR / "product_logs.csv"
    
    # Vision & Face Parameters
    FACE_SIMILARITY_THRESHOLD: float = 0.60
    PRODUCT_IMAGE_SIZE: tuple = (96, 96)
    
    class Config:
        case_sensitive = True

settings = Settings()

# Ensure required logs directory exists
os.makedirs(settings.LOGS_DIR, exist_ok=True)
