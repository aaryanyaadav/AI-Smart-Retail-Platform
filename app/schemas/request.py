from pydantic import BaseModel, Field

class SentimentRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=2,
        max_length=2000,
        description="Review or feedback text to analyze for sentiment.",
        example="This jacket fits great and has excellent material quality!"
    )

class ChatbotRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="Customer question or message sent to the retail chatbot.",
        example="What is your return policy for worn clothes?"
    )

class RegisterFaceRequest(BaseModel):
    customer_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Customer name or ID to register.",
        example="Sarah Jenkins"
    )
