from app.services.evaluasi_model_service import read_evaluasi_model_service, create_evaluasi_model_service

def create_evaluasi_model_controller():
    try:
        result = create_evaluasi_model_service()
        return result
    except Exception as e:
        return e

def read_evaluasi_model_controller():
    try:
        result = read_evaluasi_model_service()
        return result
    except Exception as e:
        return e
    