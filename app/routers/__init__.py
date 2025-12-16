from fastapi import APIRouter
from . import titik_akses_dataset
from . import titik_akses_knn
from . import titik_akses_logreg
from . import titik_akses_prediksi_knn
from . import titik_akses_prediksi_logreg
from . import titik_akses_model_klasifikasi

router = APIRouter(
    prefix="/api/v1",
    tags=["sakit jantung"]
)

router.include_router(titik_akses_dataset.router)
router.include_router(titik_akses_knn.router)
router.include_router(titik_akses_prediksi_knn.router)
router.include_router(titik_akses_prediksi_logreg.router)
router.include_router(titik_akses_logreg.router)
router.include_router(titik_akses_model_klasifikasi.router)