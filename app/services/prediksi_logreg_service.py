import os
import pickle
import pandas as pd

def create_predict_logreg_service(input_data: dict):
    try:
        # Tentukan path model Logistic Regression
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "logreg.pkl")

        if not os.path.exists(model_path):
            return {
                "status": False,
                "message": "Model Logistic Regression tidak ditemukan. Harap jalankan training terlebih dahulu.",
                "data": None
            }

        # 📦 Load model logistic regression
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        mi_scores_norm = model_data["mi_scores_norm"]
        features = model_data["features"]

        # 🧾 Cek kelengkapan fitur input
        missing_features = [f for f in features if f not in input_data]
        if missing_features:
            return {
                "status": False,
                "message": f"Input tidak lengkap. Fitur yang hilang: {missing_features}",
                "data": None
            }

        # 📊 Buat dataframe sesuai urutan fitur model
        input_df = pd.DataFrame([input_data])[features]

        # 🔢 Scaling
        scaled_input = scaler.transform(input_df)

        # ⚖️ MI weighting (sama seperti saat training)
        weighted_input = scaled_input * mi_scores_norm

        # 🧠 Prediksi
        prediction = int(model.predict(weighted_input)[0])

        # Probabilitas kelas 1 (berisiko)
        probability = float(model.predict_proba(weighted_input)[0][1])

        return {
            "status": True,
            "message": "Prediksi berhasil dilakukan.",
            "data": {
                "prediction": prediction,
                "probability": probability,
                "label": "Berisiko Penyakit Jantung" if prediction == 1 else "Tidak Berisiko"
            }
        }

    except Exception as e:
        return {
            "status": False,
            "message": f"Terjadi kesalahan saat melakukan prediksi: {e}",
            "data": None
        }
