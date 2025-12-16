from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.controllers import peringkat_fitur_controller
from app.models.response import ApiResponse

router = APIRouter()

@router.get("/peringkat-fitur")
def create_peringkat_fitur():
    try:
        result =  peringkat_fitur_controller.create_peringkat_fitur_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Peringkat Fitur Created",
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