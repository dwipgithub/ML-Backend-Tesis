from fastapi import APIRouter
from . import titik_akses_dataset
from . import titik_akses_evaluasi_model
from . import titik_akses_knn
from . import titik_akses_prediksi_knn

router = APIRouter(
    prefix="/penyakit-jantung",
    tags=["sakit jantung"]
)

router.include_router(titik_akses_dataset.router)
router.include_router(titik_akses_evaluasi_model.router)
router.include_router(titik_akses_knn.router)
router.include_router(titik_akses_prediksi_knn.router)
