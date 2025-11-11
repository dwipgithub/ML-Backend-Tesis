from fastapi import APIRouter, status
from app.controllers import dataset_controller

router = APIRouter()

@router.post("/dataset", status_code=status.HTTP_201_CREATED)
def create_dataset():
    matrics = dataset_controller.create()
    return {
        "status": True,
        "message": "Dataset Created Successfully",
    }