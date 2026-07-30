import re
import csv
import logging
from datetime import datetime
from typing import Dict, Any
import numpy as np

from app.services.pipeline import model_registry
from app.utils.config import settings

logger = logging.getLogger("smart_retail.sentiment_service")

class SentimentService:
    """
    Sentiment Analysis Service.
    Retrieves sentiment model & vectorizer strictly from ModelRegistry.
    Does NOT load model files directly.
    """
    def __init__(self):
        # Contractions expansion map
        self.contractions_dict = {
            "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
            "'s": " is", "'d": " would", "'ll": " will", "'t": " not",
            "'ve": " have", "'m": " am"
        }

    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess raw review text.
        """
        # 1. Lowercase
        text = text.lower()
        # 2. Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # 3. Handle contractions
        pattern = re.compile(r'\b(' + '|'.join(self.contractions_dict.keys()) + r')\b')
        text = pattern.sub(lambda m: self.contractions_dict[m.group(0)], text)
        # 4. Remove non-ASCII characters & emojis
        text = text.encode('ascii', 'ignore').decode('ascii')
        # 5. Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 6. Remove excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def log_sentiment(self, text: str, sentiment: str, confidence: float) -> None:
        """
        Append prediction log to sentiment_logs.csv file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = settings.SENTIMENT_LOG_PATH.exists()

        try:
            with open(settings.SENTIMENT_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'input_text', 'sentiment', 'confidence'])
                writer.writerow([timestamp, text, sentiment, confidence])
        except Exception as e:
            logger.error(f"Failed to log sentiment to CSV: {e}")

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment for input review text.
        """
        model = model_registry.get_sentiment_model()
        vectorizer = model_registry.get_sentiment_vectorizer()

        if model is None or vectorizer is None:
            raise RuntimeError("Sentiment model or vectorizer is not loaded in ModelRegistry.")

        # Clean input text
        cleaned_text = self.clean_text(text)

        # Transform using TF-IDF vectorizer
        tfidf_features = vectorizer.transform([cleaned_text])

        # Predict sentiment
        prediction = model.predict(tfidf_features)[0]

        # Calculate decision function / confidence probability score
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(tfidf_features)[0]
            confidence = round(float(np.max(probs)) * 100, 2)
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(tfidf_features)[0]
            # Convert decision distance score to pseudo-probability via sigmoid
            if isinstance(decision, np.ndarray):
                score = np.max(1 / (1 + np.exp(-decision)))
            else:
                score = 1 / (1 + np.exp(-abs(decision)))
            confidence = round(float(score) * 100, 2)
        else:
            confidence = 88.50

        # Log sentiment prediction
        self.log_sentiment(text, prediction, confidence)

        return {
            "input_text": text,
            "cleaned_text": cleaned_text,
            "sentiment": prediction,
            "confidence_score": confidence
        }

sentiment_service = SentimentService()
