from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.services.product_service import product_service
from app.services.face_service import face_service
from app.schemas.response import ProductClassificationResponse, FaceRecognitionResponse, BaseResponse

router = APIRouter(prefix="/vision", tags=["Computer Vision APIs"])

@router.post(
    "/classify-product",
    response_model=ProductClassificationResponse,
    summary="Classify retail product category from image",
    description="Upload or camera-capture a product image to classify it into retail categories using MobileNetV2."
)
async def classify_product(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
        
    filename = file.filename or "product_image.jpg"
    result = product_service.classify_product(contents, filename=filename)
    return ProductClassificationResponse(
        message="Product classified successfully.",
        predicted_category=result["predicted_category"],
        confidence_score=result["confidence_score"],
        top_3_predictions=result["top_3_predictions"]
    )

@router.post(
    "/recognize-face",
    response_model=FaceRecognitionResponse,
    summary="Recognize customer face for loyalty tracking",
    description="Detect face in camera frame, compare against registered customer database, and log visit."
)
async def recognize_face(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
        
    result = face_service.recognize_face(contents)
    return FaceRecognitionResponse(
        message="Face processed successfully.",
        customer_id=result["customer_id"],
        customer_status=result["status"],
        confidence_score=result["confidence_score"],
        distance=result["distance"]
    )

@router.post(
    "/register-face",
    response_model=BaseResponse,
    summary="Register a new customer face",
    description="Captures image and registers a new customer name into face_db.pkl."
)
async def register_face(customer_name: str = Form(...), file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )
        
    result = face_service.register_customer_face(customer_name, contents)
    return BaseResponse(
        message=result["message"]
    )
