from fastapi import APIRouter
from app.services.chatbot_service import chatbot_service
from app.schemas.request import ChatbotRequest
from app.schemas.response import ChatbotResponse

router = APIRouter(tags=["Retail Chatbot API"])

@router.post(
    "/chatbot",
    response_model=ChatbotResponse,
    summary="Process customer chatbot query",
    description="Processes user query via hybrid pipeline: rule-based FAQ match -> ML intent classification fallback -> response generation."
)
async def chatbot(payload: ChatbotRequest):
    result = chatbot_service.get_response(payload.question)
    return ChatbotResponse(
        status="success",
        message="Chatbot response generated successfully.",
        question=result["question"],
        intent=result["intent"],
        response=result["response"],
        confidence_score=result["confidence_score"],
        match_type=result["match_type"]
    )
