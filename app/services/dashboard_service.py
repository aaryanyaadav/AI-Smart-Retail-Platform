import csv
import logging
from typing import Dict, Any, List
import pandas as pd
from app.utils.config import settings

logger = logging.getLogger("smart_retail.dashboard_service")

class DashboardService:
    """
    Dashboard Analytics Service.
    Reads logs directly to compute real-time retail intelligence metrics & drill-down details.
    """
    def get_dashboard_stats(self) -> Dict[str, Any]:
        # 1. Process Face Visits Log
        today_visitors = 0
        returning_customers = 0
        new_customers = 0
        returning_rate = 0.0

        if settings.VISITS_LOG_PATH.exists():
            try:
                df_visits = pd.read_csv(settings.VISITS_LOG_PATH)
                if not df_visits.empty:
                    today_visitors = len(df_visits)
                    returning_customers = len(df_visits[df_visits['status'] == 'Returning Customer'])
                    new_customers = len(df_visits[df_visits['status'] == 'New Customer'])
                    if today_visitors > 0:
                        returning_rate = round((returning_customers / today_visitors) * 100, 2)
            except Exception as e:
                logger.error(f"Error reading visits log: {e}")

        # 2. Process Sentiment Logs
        positive_reviews = 0
        negative_reviews = 0
        neutral_reviews = 0

        if settings.SENTIMENT_LOG_PATH.exists():
            try:
                df_sent = pd.read_csv(settings.SENTIMENT_LOG_PATH)
                if not df_sent.empty and 'sentiment' in df_sent.columns:
                    counts = df_sent['sentiment'].value_counts()
                    positive_reviews = int(counts.get('positive', 0))
                    negative_reviews = int(counts.get('negative', 0))
                    neutral_reviews = int(counts.get('neutral', 0))
            except Exception as e:
                logger.error(f"Error reading sentiment log: {e}")

        total_reviews = positive_reviews + negative_reviews + neutral_reviews
        positive_sentiment_rate = round((positive_reviews / total_reviews) * 100, 2) if total_reviews > 0 else 0.0

        # 3. Process Chatbot Logs (Most Asked Intent)
        most_asked_intent = "store_hours"
        if settings.CHATBOT_LOG_PATH.exists():
            try:
                df_bot = pd.read_csv(settings.CHATBOT_LOG_PATH)
                if not df_bot.empty and 'intent' in df_bot.columns:
                    mode_val = df_bot['intent'].mode()
                    if not mode_val.empty:
                        most_asked_intent = str(mode_val[0])
            except Exception as e:
                logger.error(f"Error reading chatbot log: {e}")

        # 4. Process Product Logs (Most Predicted Category)
        most_predicted_product_category = "T-shirt/top"
        if settings.PRODUCT_LOG_PATH.exists():
            try:
                df_prod = pd.read_csv(settings.PRODUCT_LOG_PATH)
                if not df_prod.empty and 'prediction' in df_prod.columns:
                    mode_val = df_prod['prediction'].mode()
                    if not mode_val.empty:
                        most_predicted_product_category = str(mode_val[0])
            except Exception as e:
                logger.error(f"Error reading product log: {e}")

        return {
            "today_visitors": today_visitors,
            "returning_customers": returning_customers,
            "new_customers": new_customers,
            "returning_rate": returning_rate,
            "total_reviews": total_reviews,
            "positive_reviews": positive_reviews,
            "negative_reviews": negative_reviews,
            "neutral_reviews": neutral_reviews,
            "positive_sentiment_rate": positive_sentiment_rate,
            "most_asked_intent": most_asked_intent,
            "most_predicted_product_category": most_predicted_product_category
        }

    def get_log_details(self, category: str) -> List[Dict[str, Any]]:
        """
        Retrieve raw log entry records for modal drill-down inspection.
        """
        results = []
        try:
            if category in ['visitors', 'returning']:
                if settings.VISITS_LOG_PATH.exists():
                    df = pd.read_csv(settings.VISITS_LOG_PATH)
                    if category == 'returning':
                        df = df[df['status'] == 'Returning Customer']
                    results = df.fillna("").to_dict(orient="records")

            elif category in ['reviews', 'positive_reviews']:
                if settings.SENTIMENT_LOG_PATH.exists():
                    df = pd.read_csv(settings.SENTIMENT_LOG_PATH)
                    if category == 'positive_reviews':
                        df = df[df['sentiment'] == 'positive']
                    results = df.fillna("").to_dict(orient="records")

            elif category == 'intent':
                if settings.CHATBOT_LOG_PATH.exists():
                    df = pd.read_csv(settings.CHATBOT_LOG_PATH)
                    results = df.fillna("").to_dict(orient="records")

            elif category == 'product':
                if settings.PRODUCT_LOG_PATH.exists():
                    df = pd.read_csv(settings.PRODUCT_LOG_PATH)
                    results = df.fillna("").to_dict(orient="records")
        except Exception as e:
            logger.error(f"Error reading log details for {category}: {e}")

        return results

dashboard_service = DashboardService()
