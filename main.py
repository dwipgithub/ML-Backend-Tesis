from fastapi import FastAPI
from app.routers import router

app = FastAPI()

# for r in [
#     evaluasi_model_router.router, 
#     dataset_router.router, 
#     knn_router.router]:
#     app.include_router(r)

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "API is running"
    }