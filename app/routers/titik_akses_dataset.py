from fastapi import APIRouter, status
from app.controllers import dataset_controller
from fastapi.responses import JSONResponse
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/dataset", status_code=status.HTTP_201_CREATED)
def create_dataset():
    matrics = dataset_controller.create()
    return {
        "status": True,
        "message": "Dataset Created Successfully",
    }

@router.get("/dataset")
def read_dataset():
    try:
        result = dataset_controller.read()
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