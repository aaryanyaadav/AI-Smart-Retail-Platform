import csv
import logging
from datetime import datetime
from app.utils.config import settings

logger = logging.getLogger("smart_retail.csv_logger")

def log_product_prediction(filename: str, prediction: str, confidence: float) -> None:
    """
    Log product classification result to product_logs.csv.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_exists = settings.PRODUCT_LOG_PATH.exists()

    try:
        with open(settings.PRODUCT_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'filename', 'prediction', 'confidence'])
            writer.writerow([timestamp, filename, prediction, confidence])
    except Exception as e:
        logger.error(f"Failed to log product prediction to CSV: {e}")
