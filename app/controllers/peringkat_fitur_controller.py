from app.services.peringkat_fitur import create_peringkat_fitur_service

def create_peringkat_fitur_controller():
    try:
        result = create_peringkat_fitur_service()
        return result
    except Exception as e:
        return e