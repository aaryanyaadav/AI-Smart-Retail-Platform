import json
import pickle
import logging
from typing import Dict, Any, Optional
import joblib

from app.utils.config import settings

logger = logging.getLogger("smart_retail.pipeline")

class ModelRegistry:
    """
    Singleton Model Registry responsible for loading and holding all AI models,
    vectorizers, face database encodings, and intent datasets in memory.
    
    Adheres to SOLID principles: Single Responsibility for Model Lifecycle Management.
    """
    _instance: Optional['ModelRegistry'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._instance._is_initialized = False
        return cls._instance

    def initialize(self):
        if self._is_initialized:
            logger.info("ModelRegistry already initialized.")
            return

        logger.info("Initializing ModelRegistry - Loading all AI models into memory...")

        # 1. Load Product Classifier Model (TensorFlow / Keras with dummy fallback if TF unavailable on Python 3.14)
        try:
            import tensorflow as tf
            self._product_model = tf.keras.models.load_model(str(settings.PRODUCT_MODEL_PATH))
            logger.info(f"Loaded Product Classifier Model from {settings.PRODUCT_MODEL_PATH}")
        except Exception as e:
            logger.warning(f"TensorFlow not loaded ({e}). Using mock/fallback for Product Classifier.")
            self._product_model = "MOCK_PRODUCT_MODEL"

        # 2. Load Sentiment Analysis Model & Vectorizer
        try:
            self._sentiment_model = joblib.load(str(settings.SENTIMENT_MODEL_PATH))
            self._sentiment_vectorizer = joblib.load(str(settings.SENTIMENT_VECTORIZER_PATH))
            logger.info(f"Loaded Sentiment Model & Vectorizer from {settings.SENTIMENT_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Sentiment artifacts: {e}")
            self._sentiment_model = None
            self._sentiment_vectorizer = None

        # 3. Load Chatbot Model & Vectorizer
        try:
            self._chatbot_model = joblib.load(str(settings.CHATBOT_MODEL_PATH))
            self._chatbot_vectorizer = joblib.load(str(settings.CHATBOT_VECTORIZER_PATH))
            logger.info(f"Loaded Chatbot Model & Vectorizer from {settings.CHATBOT_MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Chatbot artifacts: {e}")
            self._chatbot_model = None
            self._chatbot_vectorizer = None

        # 4. Load Face Database Encodings
        try:
            with open(settings.FACE_DB_PATH, 'rb') as f:
                self._face_database = pickle.load(f)
            logger.info(f"Loaded Face Database containing {len(self._face_database)} customers from {settings.FACE_DB_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Face Database: {e}")
            self._face_database = {}

        # 5. Load Intents Dataset
        try:
            with open(settings.INTENTS_JSON_PATH, 'r', encoding='utf-8') as f:
                self._intents = json.load(f)
            logger.info(f"Loaded Intents Dataset containing {len(self._intents.get('intents', []))} categories from {settings.INTENTS_JSON_PATH}")
        except Exception as e:
            logger.error(f"Failed to load Intents JSON: {e}")
            self._intents = {"intents": []}

        self._is_initialized = True
        logger.info("ModelRegistry successfully initialized all models!")

    # Getter methods (No preprocessing or inference logic allowed in ModelRegistry)
    def get_product_model(self) -> Any:
        return self._product_model

    def get_sentiment_model(self) -> Any:
        return self._sentiment_model

    def get_sentiment_vectorizer(self) -> Any:
        return self._sentiment_vectorizer

    def get_chatbot_model(self) -> Any:
        return self._chatbot_model

    def get_chatbot_vectorizer(self) -> Any:
        return self._chatbot_vectorizer

    def get_face_database(self) -> Dict[str, Any]:
        return self._face_database

    def get_intents(self) -> Dict[str, Any]:
        return self._intents

# Global accessor instance
model_registry = ModelRegistry()
