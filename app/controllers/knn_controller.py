from app.services.knn_service import create_knn_service, read_knn_service

def create_pemodelan_controller():
    try:
        result = create_knn_service()
        return result
    except Exception as e:
        return e

def read_pemodelan_controller():
    try:
        result = read_knn_service()
        return result
    except Exception as e:
        return e