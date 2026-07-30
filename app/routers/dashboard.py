from typing import List, Dict, Any
from fastapi import APIRouter, Query
from app.services.dashboard_service import dashboard_service
from app.schemas.response import DashboardStatsResponse

router = APIRouter(prefix="/dashboard", tags=["Retail Analytics Dashboard API"])

@router.get(
    "/stats",
    response_model=DashboardStatsResponse,
    summary="Get real-time retail intelligence dashboard metrics",
    description="Calculates real-time metrics including visitor counts, loyalty return rates, review sentiment breakdowns, and top queries."
)
async def get_dashboard_stats():
    result = dashboard_service.get_dashboard_stats()
    return DashboardStatsResponse(
        status="success",
        message="Dashboard stats calculated successfully.",
        today_visitors=result["today_visitors"],
        returning_customers=result["returning_customers"],
        new_customers=result["new_customers"],
        returning_rate=result["returning_rate"],
        total_reviews=result["total_reviews"],
        positive_reviews=result["positive_reviews"],
        negative_reviews=result["negative_reviews"],
        neutral_reviews=result["neutral_reviews"],
        positive_sentiment_rate=result["positive_sentiment_rate"],
        most_asked_intent=result["most_asked_intent"],
        most_predicted_product_category=result["most_predicted_product_category"]
    )

@router.get(
    "/details",
    summary="Get detailed log records for drill-down inspection",
    description="Returns raw CSV log records for visitors, reviews, chatbot queries, and product classifications."
)
async def get_dashboard_details(category: str = Query(..., description="Category: visitors, returning, reviews, positive_reviews, intent, product")):
    records = dashboard_service.get_log_details(category)
    return {
        "status": "success",
        "category": category,
        "total_records": len(records),
        "data": records
    }
