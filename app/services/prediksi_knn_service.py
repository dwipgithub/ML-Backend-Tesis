import os
import pickle
import pandas as pd

def create_predict_knn_service(input_data: dict):
    try:
        # Tentukan path model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "knn.pkl")

        if not os.path.exists(model_path):
            return {
                "status": False,
                "message": "Model KNN tidak ditemukan. Harap jalankan training terlebih dahulu.",
                "data": None
            }

        # 📦 Muat model, scaler, dan informasi lainnya
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        mi_scores_norm = model_data["mi_scores_norm"]
        features = model_data["features"]

        # 🧾 Pastikan input memiliki semua fitur yang diperlukan
        missing_features = [f for f in features if f not in input_data]
        if missing_features:
            return {
                "status": False,
                "message": f"Input tidak lengkap. Fitur yang hilang: {missing_features}",
                "data": None
            }

        # 📊 Buat dataframe dari input_data
        input_df = pd.DataFrame([input_data])[features]

        # 🔢 Scaling
        scaled_input = scaler.transform(input_df)

        # ⚖️ MI weighting
        weighted_input = scaled_input * mi_scores_norm

        # 🧠 Prediksi
        prediction = model.predict(weighted_input)[0]

        probability = (
            model.predict_proba(weighted_input)[0][1]
            if hasattr(model, "predict_proba")
            else None
        )

        # 🎯 Hitung manual probabilitas dengan melihat tetangga
        distances, indices = model.kneighbors(weighted_input)

        # Ambil kelas masing-masing tetangga
        neighbor_labels = model._y[indices][0]

        # Hitung jumlah kelas
        count_class_1 = int(sum(neighbor_labels))
        count_class_0 = int(len(neighbor_labels) - count_class_1)

        # Probabilitas manual
        manual_probability = count_class_1 / len(neighbor_labels)

        return {
            "status": True,
            "message": "Prediksi berhasil dilakukan.",
            "data": {
                "prediction": int(prediction),
                "probability": float(probability) if probability is not None else None,
                "probabilityCalculation": {
                    "k": len(neighbor_labels),
                    "neighborLabels": neighbor_labels.tolist(),
                    "countClass0": count_class_0,
                    "countClass1": count_class_1,
                    "manualProbability": manual_probability
                },
                "label": "Berisiko Penyakit Jantung" if prediction == 1 else "Tidak Berisiko"
            }
        }

    except Exception as e:
        return {
            "status": False,
            "message": f"Terjadi kesalahan saat melakukan prediksi: {e}",
            "data": None
        }
