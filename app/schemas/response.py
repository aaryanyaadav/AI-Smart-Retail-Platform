from typing import Dict, Any
from pydantic import BaseModel, Field

class BaseResponse(BaseModel):
    status: str = Field("success", description="Response status message.")

class ProductPrediction(BaseModel):
    category: str = Field(..., description="Fashion category name.")
    confidence: float = Field(..., description="Prediction confidence score.")

class ProductClassificationResponse(BaseResponse):
    message: str = Field(..., description="Classification result description.")
    predicted_category: str = Field(..., description="Top predicted product category.")
    confidence_score: float = Field(..., description="Prediction confidence score percentage.")
    top_3_predictions: list[ProductPrediction] = Field(..., description="Top 3 predicted categories with scores.")

# Face Recognition Response
class FaceRecognitionResponse(BaseResponse):
    message: str = Field(..., description="Face recognition result description.")
    customer_id: str = Field(..., description="Customer ID or full name.")
    customer_status: str = Field(..., description="Status: 'Returning Customer' or 'New Customer'.")
    confidence_score: float = Field(..., description="Face recognition confidence score.")
    distance: float = Field(..., description="Euclidean distance to closest face encoding.")

# Sentiment Analysis Response (Supported with alias alias/backward compatibility)
class SentimentAnalysisResponse(BaseResponse):
    message: str = Field("Sentiment analyzed successfully.", description="Status message.")
    input_text: str = Field("", description="Original input review text.")
    cleaned_text: str = Field(..., description="Preprocessed review text.")
    sentiment: str = Field(..., description="Predicted sentiment class: 'positive', 'negative', or 'neutral'.")
    confidence_score: float = Field(..., description="Prediction confidence score percentage.")

SentimentResponse = SentimentAnalysisResponse

# Chatbot Response
class ChatbotResponse(BaseResponse):
    question: str = Field(..., description="Original customer query.")
    intent: str = Field(..., description="Identified query intent tag.")
    response: str = Field(..., description="Generated chatbot response.")
    confidence_score: float = Field(..., description="Intent classification confidence percentage.")
    match_type: str = Field(..., description="Strategy used: 'Rule-Based FAQ Match' or 'ML Classifier Fallback'.")

# Dashboard Analytics Response
class DashboardStatsResponse(BaseResponse):
    today_visitors: int = Field(..., description="Total visitors logged today.")
    returning_customers: int = Field(..., description="Total returning customers logged today.")
    new_customers: int = Field(..., description="Total new customers logged today.")
    returning_rate: float = Field(..., description="Percentage of returning customer visits.")
    total_reviews: int = Field(..., description="Total review feedback entries logged.")
    positive_reviews: int = Field(..., description="Total positive sentiment reviews logged.")
    negative_reviews: int = Field(..., description="Total negative sentiment reviews logged.")
    neutral_reviews: int = Field(..., description="Total neutral sentiment reviews logged.")
    positive_sentiment_rate: float = Field(..., description="Percentage of positive sentiment reviews.")
    most_asked_intent: str = Field(..., description="Top intent category queried by customers.")
    most_predicted_product_category: str = Field(..., description="Top product category classified.")

# Health Endpoint Response
class HealthResponse(BaseResponse):
    version: str = Field("1.0.0", description="Application version.")
    models_loaded: Dict[str, bool] = Field(..., description="In-memory model load status.")
