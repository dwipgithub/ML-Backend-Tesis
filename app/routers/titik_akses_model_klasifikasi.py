from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from app.models.response import ApiResponse

from app.controllers.model_klasifikasi_controller import (
    create_pelatihan_knn_controller,
    create_pelatihan_nb_controller,
    create_pelatihan_lr_controller,
    create_prediksi_knn_controller,
    create_prediksi_lr_controller,
    create_prediksi_nb_controller,
    read_pelatihan_controller,
    read_evaluasi_model_controller,
    read_peringkat_fitur_controller
    
)

router = APIRouter()

@router.post("/model-klasifikasi/penyakit-jantung/knn/pelatihan")
def create_pelatihan_knn_router():
    try:
        result = create_pelatihan_knn_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Model KNN berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.post("/model-klasifikasi/penyakit-jantung/nb/pelatihan")
def create_pelatihan_nb_router():
    try:
        result = create_pelatihan_nb_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Model NB berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.post("/model-klasifikasi/penyakit-jantung/lr/pelatihan")
def create_pelatihan_nb_router():
    try:
        result = create_pelatihan_lr_controller()
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Model LR berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.post("/model-klasifikasi/penyakit-jantung/knn/prediksi")
def create_prediksi_knn_router(input_data: dict):
    try:
        result = create_prediksi_knn_controller(input_data)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Prediksi KNN berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.post("/model-klasifikasi/penyakit-jantung/lr/prediksi")
def create_prediksi_lr_router(input_data: dict):
    try:
        result = create_prediksi_lr_controller(input_data)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Prediksi LR berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.post("/model-klasifikasi/penyakit-jantung/nb/prediksi")
def create_prediksi_nb_router(input_data: dict):
    try:
        result = create_prediksi_nb_controller(input_data)
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content=ApiResponse(
                error=False,
                message="Prediksi NB berhasil dibuat.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True, 
                message=str(e), 
                data=None
            ).model_dump()
        )

@router.get("/model-klasifikasi/penyakit-jantung/evaluasi")
def evaluasi_model_router():
    try:
        result = read_evaluasi_model_controller()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message=f".",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True,
                message=str(e),
                data=None
            ).model_dump()
        )

@router.get("/model-klasifikasi/penyakit-jantung/peringkat-fitur")
def peringkat_fitur_router():
    try:
        result = read_peringkat_fitur_controller()
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message=f".",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True,
                message=str(e),
                data=None
            ).model_dump()
        )

@router.get("/model-klasifikasi/penyakit-jantung/{algoritma}/pelatihan")
def read_pelatihan_router(algoritma: str):
    try:
        result = read_pelatihan_controller(algoritma)
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=ApiResponse(
                error=False,
                message=f"Daftar model {algoritma}.",
                data=result
            ).model_dump()
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ApiResponse(
                error=True,
                message=str(e),
                data=None
            ).model_dump()
        )
