from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.controllers import evaluasi_model_controller
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/evaluasi-model")
def create_evaluasi_model_router():
    try:
        result = evaluasi_model_controller.create_evaluasi_model_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Evaluate Model Created",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True,
                message= f"{e}",
                data=None
            ).model_dump()
        )

@router.get("/evaluasi-model")
def read_evaluasi_model_router():
    try:
        result = evaluasi_model_controller.read_evaluasi_model_controller()

        if result is None:
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content=ApiResponse(
                    error=True,
                    message="Data Not Found",
                    data=result
                ).model_dump()
            )
            

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message="Data Found",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True,
                message= f"{e}",
                data=None
            ).model_dump()
        )