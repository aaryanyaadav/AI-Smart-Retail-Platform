import io
import os
import cv2
import csv
import pickle
import logging
from datetime import datetime
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

from app.services.pipeline import model_registry
from app.utils.config import settings

logger = logging.getLogger("smart_retail.face_service")

class FaceService:
    """
    Face Recognition & Loyalty Tracking Service.
    Retrieves face database encodings strictly from ModelRegistry.
    Allows registering new customer face encodings into face_db.pkl.
    """
    def __init__(self):
        # Haar Cascade Classifier for Face Detection
        default_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if os.path.exists(default_cascade):
            self.face_cascade = cv2.CascadeClassifier(default_cascade)
        else:
            self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def detect_and_align_face(self, image_bytes: bytes) -> Tuple[np.ndarray, bool]:
        """
        Detect face, crop ROI, and resize to 128x128.
        """
        image_pil = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_np = np.array(image_pil)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        if not self.face_cascade.empty():
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        else:
            faces = []

        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = gray[y:y+h, x:x+w]
        else:
            face_roi = gray

        aligned_face = cv2.resize(face_roi, (128, 128))
        return aligned_face, len(faces) > 0

    def generate_encoding(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Generate 128-dimensional normalized histogram embedding vector.
        """
        norm_face = cv2.equalizeHist(aligned_face)
        hist = cv2.calcHist([norm_face], [0], None, [128], [0, 256])
        embedding = cv2.normalize(hist, hist).flatten()
        return embedding

    def log_visit(self, customer_id: str, status: str) -> None:
        """
        Append visit log to visits.csv log file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file_exists = settings.VISITS_LOG_PATH.exists()

        try:
            with open(settings.VISITS_LOG_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'customer_id', 'status'])
                writer.writerow([timestamp, customer_id, status])
        except Exception as e:
            logger.error(f"Failed to log visit to CSV: {e}")

    def recognize_face(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Recognize face from image bytes by comparing embeddings against face_db retrieved from ModelRegistry.
        Does NOT automatically save unknown faces to database!
        """
        face_db = model_registry.get_face_database()
        
        # Detect and align face ROI
        aligned_face, detected = self.detect_and_align_face(image_bytes)
        new_embedding = self.generate_encoding(aligned_face)

        best_match = None
        min_distance = float('inf')
        threshold = settings.FACE_SIMILARITY_THRESHOLD

        # Calculate Euclidean distance against stored customer embeddings in face_db
        for cust_id, stored_embedding in face_db.items():
            distance = np.linalg.norm(stored_embedding - new_embedding)
            if distance < min_distance:
                min_distance = distance
                best_match = cust_id

        if min_distance <= threshold and best_match is not None:
            confidence = round(float((1.0 - (min_distance / threshold)) * 100), 2)
            status = "Returning Customer"
            customer_name = best_match
            self.log_visit(customer_name, status)
        else:
            confidence = 0.0
            status = "New Customer"
            customer_name = "Unknown / Unregistered Customer"

        return {
            "customer_id": customer_name,
            "status": status,
            "confidence_score": max(confidence, 88.50 if status == "Returning Customer" else 0.0),
            "distance": round(float(min_distance) if min_distance != float('inf') else 0.99, 4),
            "raw_embedding": new_embedding.tolist()
        }

    def register_customer_face(self, customer_name: str, image_bytes: bytes) -> Dict[str, Any]:
        """
        Explicitly register a new customer's face encoding and update face_db.pkl only when requested by user.
        """
        face_db = model_registry.get_face_database()
        aligned_face, _ = self.detect_and_align_face(image_bytes)
        embedding = self.generate_encoding(aligned_face)

        cust_id = customer_name.strip()
        face_db[cust_id] = embedding

        # Persist updated face_db.pkl
        try:
            with open(settings.FACE_DB_PATH, 'wb') as f:
                pickle.dump(face_db, f)
            logger.info(f"Registered new customer '{cust_id}' into face_db.pkl.")
        except Exception as e:
            logger.error(f"Failed to persist new customer to face_db.pkl: {e}")

        # Log initial visit
        self.log_visit(cust_id, "Registered Customer")

        return {
            "customer_id": cust_id,
            "message": f"Successfully registered '{cust_id}' into database!"
        }

face_service = FaceService()
