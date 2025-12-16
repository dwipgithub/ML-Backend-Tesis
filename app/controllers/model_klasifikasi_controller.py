from app.services.model_klasifikasi_service import (
    create_pelatihan_knn_service,
    create_pelatihan_nb_service,
    create_pelatihan_lr_service,
    create_prediksi_knn_service,
    create_prediksi_lr_service,
    create_prediksi_nb_service,
    read_pelatihan_service,
    read_evaluasi_model_service,
    read_peringkat_fitur_service
)

def create_pelatihan_knn_controller():
    try:
        return create_pelatihan_knn_service()
    except Exception as e:
        return {"error": str(e)}

def create_pelatihan_nb_controller():
    try:
        return create_pelatihan_nb_service()
    except Exception as e:
        return {"error": str(e)}

def create_pelatihan_lr_controller():
    try:
        return create_pelatihan_lr_service()
    except Exception as e:
        return {"error": str(e)}

def create_prediksi_knn_controller(input_data: dict):
    try:
        return create_prediksi_knn_service(input_data)
    except Exception as e:
        return {"error": str(e)}
    
def create_prediksi_lr_controller(input_data: dict):
    try:
        return create_prediksi_lr_service(input_data)
    except Exception as e:
        return {"error": str(e)}

def create_prediksi_nb_controller(input_data: dict):
    try:
        return create_prediksi_nb_service(input_data)
    except Exception as e:
        return {"error": str(e)}

def read_pelatihan_controller(algoritma: str):
    try:
        return read_pelatihan_service(algoritma)
    except Exception as e:
        return {"error": str(e)}

def read_evaluasi_model_controller():
    try:
        return read_evaluasi_model_service()
    except Exception as e:
        return {"error": str(e)}

def read_peringkat_fitur_controller():
    try:
        return read_peringkat_fitur_service()
    except Exception as e:
        return {"error": str(e)}


    try:
        return create_prediksi_knn_service()
    except Exception as e:
        return {"error": str(e)}