from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.controllers import logreg_controller
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/pemodelan-logreg")
def create_pemodelan_router():
    try:
        result = logreg_controller.create_pemodelan_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Modeling Created",
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

@router.get("/pemodelan-logreg")
def read_pemodelan_logreg_router():
    try:
        result = logreg_controller.read_pemodelan_controller()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message="Model Logreg berhasil dimuat.",
                data={
                    "features": result["features"],
                    "mi_scores_norm": result["mi_scores_norm"].tolist(),
                    "model_type": type(result["model"]).__name__
                }
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