from fastapi import APIRouter
from app.services.sentiment_service import sentiment_service
from app.schemas.request import SentimentRequest
from app.schemas.response import SentimentAnalysisResponse

router = APIRouter(tags=["Sentiment Analysis API"])

@router.post(
    "/analyze-sentiment",
    response_model=SentimentAnalysisResponse,
    summary="Analyze customer review sentiment",
    description="Clean review text, extract TF-IDF features, and classify sentiment as positive, negative, or neutral."
)
async def analyze_sentiment(payload: SentimentRequest):
    result = sentiment_service.analyze_sentiment(payload.text)
    return SentimentAnalysisResponse(
        status="success",
        message="Sentiment analyzed successfully.",
        input_text=result["input_text"],
        cleaned_text=result["cleaned_text"],
        sentiment=result["sentiment"],
        confidence_score=result["confidence_score"]
    )
