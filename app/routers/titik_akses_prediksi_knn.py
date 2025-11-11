from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.controllers import prediksi_knn_controller
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/prediksi-knn")
def create_prediksi_knn(input_data: dict):
    try:
        result = prediksi_knn_controller.create_predict_knn_controller(input_data)

        # if not result["status"]:
        #     return JSONResponse(
        #         status_code=status.HTTP_400_BAD_REQUEST,
        #         content=ApiResponse(
        #             error=True,
        #             message=result["message"],
        #             data=None
        #         ).model_dump()
        #     )

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