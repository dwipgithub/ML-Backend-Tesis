from fastapi import APIRouter, status
from app.controllers import dataset_controller
from fastapi.responses import JSONResponse
from app.models.response import ApiResponse

router = APIRouter()

@router.post("/dataset/penyakit-jantung", status_code=status.HTTP_201_CREATED)
def create_dataset():
    matrics = dataset_controller.create()
    return {
        "status": True,
        "message": "Dataset Created Successfully",
    }

@router.get("/dataset/penyakit-jantung")
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

@router.get("/dataset/penyakit-jantung/distribusi-kelas")
def read_dataset_class_distribution():
    try:
        distribution = dataset_controller.read_class_distribution()
        total = distribution.sum()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message="Data Found",
                data=[
                    {
                        "label": "Tidak Penyakit Jantung",
                        "nilai": int(distribution.get(0, 0)),
                        "persentase": round((distribution.get(0, 0) / total) * 100, 2)
                    },
                    {
                        "label": "Penyakit Jantung",
                        "nilai": int(distribution.get(1, 0)),
                        "persentase": round((distribution.get(1, 0) / total) * 100, 2)
                    }
                ]
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

@router.get("/dataset/penyakit-jantung/statistik")
def read_dataset_statistic():
    try:
        result = dataset_controller.statistic()
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