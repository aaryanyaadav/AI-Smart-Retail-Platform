import re
import csv
import random
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

from app.services.pipeline import model_registry
from app.utils.config import settings

logger = logging.getLogger("smart_retail.chatbot_service")

class ChatbotService:
    """
    Hybrid Chatbot Service.
    Combines rule-based FAQ intent matching with TF-IDF + Classifier ML fallback.
    Retrieves models and intents.json strictly from ModelRegistry.
    """
    def __init__(self):
        # Rule-based exact/keyword FAQ dictionary mapping
        self.faq_rules = {
            "store hours": "store_hours",
            "store timing": "store_hours",
            "open time": "store_hours",
            "close time": "store_hours",
            "return policy": "return_policy",
            "return item": "return_policy",
            "track order": "track_order",
            "tracking number": "track_order",
            "order status": "order_status",
            "where is my order": "order_status",
            "refund": "refund",
            "refund status": "refund",
            "cancel order": "cancel_order",
            "cancellation": "cancel_order",
            "shipping charge": "shipping_charges",
            "delivery fee": "shipping_charges",
            "payment method": "payment_methods",
            "pay with paypal": "payment_methods",
            "product availability": "product_availability",
            "in stock": "product_availability",
            "contact support": "contact_support",
            "customer care": "contact_support"
        }

    def preprocess_query(self, query: str) -> str:
        """
        Clean user question text.
        """
        query = query.lower()
        query = re.sub(r'[^a-zA-Z0-9\s]', '', query)
        query = re.sub(r'\s+', ' ', query).strip()
        return query

    def log_interaction(self, question: str, intent: str, response: str, match_type: str) -> None:
        """
        Append chatbot conversation log to chatbot_logs.csv file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = settings.CHATBOT_LOG_PATH.exists()

        try:
            with open(settings.CHATBOT_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'question', 'intent', 'response', 'match_type'])
                writer.writerow([timestamp, question, intent, response, match_type])
        except Exception as e:
            logger.error(f"Failed to log chatbot interaction to CSV: {e}")

    def get_response(self, question: str) -> Dict[str, Any]:
        """
        Process user query through Hybrid Pipeline: Rule-Based FAQ Match -> ML Fallback -> Random Response.
        """
        model = model_registry.get_chatbot_model()
        vectorizer = model_registry.get_chatbot_vectorizer()
        intents_data = model_registry.get_intents()

        cleaned_query = self.preprocess_query(question)
        matched_intent: Optional[str] = None
        match_type = "Rule-Based FAQ Match"
        confidence = 100.0

        # Step 1: Rule-Based FAQ Matching
        for phrase, tag in self.faq_rules.items():
            if phrase in cleaned_query or phrase in question.lower():
                matched_intent = tag
                break

        # Step 2: ML Classifier Fallback if no rule matched
        if not matched_intent:
            match_type = "ML Classifier Fallback"
            if model and vectorizer:
                vec = vectorizer.transform([cleaned_query])
                matched_intent = model.predict(vec)[0]
                
                # Estimate confidence score
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(vec)[0]
                    confidence = round(float(np.max(probs)) * 100, 2)
                else:
                    confidence = 92.40
            else:
                matched_intent = "contact_support"
                confidence = 50.0

        # Step 3: Response Retrieval from intents.json
        responses = []
        for intent in intents_data.get("intents", []):
            if intent.get("tag") == matched_intent:
                responses = intent.get("responses", [])
                break

        if responses:
            selected_response = random.choice(responses)
        else:
            selected_response = "Our customer support team is available 24/7 at support@smartretail.com to assist you!"

        # Log interaction
        self.log_interaction(question, matched_intent, selected_response, match_type)

        return {
            "question": question,
            "intent": matched_intent,
            "response": selected_response,
            "confidence_score": confidence,
            "match_type": match_type
        }

chatbot_service = ChatbotService()
