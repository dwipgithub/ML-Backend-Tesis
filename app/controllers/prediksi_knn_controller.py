from app.services.prediksi_knn_service import create_predict_knn_service

def create_predict_knn_controller(input_data: dict):
    try:
        result = create_predict_knn_service(input_data)
        return result
    except Exception as e:
        return e
