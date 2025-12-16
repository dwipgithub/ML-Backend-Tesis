from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.controllers import prediksi_logreg_controller
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/prediksi-logreg")
def create_prediksi_knn(input_data: dict):
    try:
        result = prediksi_logreg_controller.create_predict_knn_controller(input_data)

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message="Prediksi berhasil dilakukan.",
                data=result["data"]
            ).model_dump()
        )
    except Exception as e:
        return e