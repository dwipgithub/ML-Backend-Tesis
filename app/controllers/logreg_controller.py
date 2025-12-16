from app.services.logreg_service import create_logreg_service, read_logreg_service 

def create_pemodelan_controller():
    try:
        result = create_logreg_service()
        return result
    except Exception as e:
        return e

def read_pemodelan_controller():
    try:
        result = read_logreg_service()
        return result
    except Exception as e:
        return e