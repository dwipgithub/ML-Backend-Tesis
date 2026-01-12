import os
import time
import pickle
import pandas as pd
from skrebate import ReliefF
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def load_and_preprocess():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(base_dir, "data", "dataset-1-cleaned.csv")
    df = pd.read_csv(dataset_path, sep=';')

    X = df.drop("Status_Penyakit_Jantung", axis=1)
    y = df["Status_Penyakit_Jantung"]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    return X, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def weighting_features(X, X_train_scaled, X_test_scaled, y_train):
    features = X.columns.tolist()

    # === Mutual Information ===
    mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())

    X_train_mi = X_train_scaled * mi_scores_norm
    X_test_mi = X_test_scaled * mi_scores_norm

    # === ReliefF ===
    relief = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1])
    relief.fit(X_train_scaled, y_train.reset_index(drop=True))
    relief_scores = relief.feature_importances_

    X_train_relief = X_train_scaled * relief_scores
    X_test_relief = X_test_scaled * relief_scores

    # Output dengan bobot dan fitur
    return (
        {
            "dasar": (X_train_scaled, X_test_scaled),
            "mi": (X_train_mi, X_test_mi),
            "relieff": (X_train_relief, X_test_relief)
        },
        mi_scores_norm,
        features
    )

def train_and_save(model_name, model, scaler, mi_scores_norm, features,
                        X_train, y_train, variant_name, X_test, y_test):

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    training_time = round(end - start, 4)

    # Save directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_dir = os.path.join(base_dir, "pkl", "models", model_name)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.join(save_dir, f"{variant_name}.pkl")

    # Performance test
    pred = model.predict(X_test)
    performance = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "f1_score": round(f1_score(y_test, pred), 4)
    }

    model_id = generate_model_id(model_name, variant_name)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = model.get_params()

    # Build package (lengkap untuk prediksi)
    model_package = {
        "model": model,
        "scaler": scaler,
        "mi_scores_norm": mi_scores_norm,
        "features": features,
        "metadata": {
            "model": model_name,
            "varian": variant_name,
            "model_id": model_id,
            "timestamp_training": timestamp,
            "training_time_seconds": training_time,
            "performance": performance
        }
    }

    # Save to file
    with open(filename, "wb") as f:
        pickle.dump(model_package, f)

    size_kb = round(os.path.getsize(filename) / 1024, 2)
    model_package["metadata"]["file_size_kb"] = size_kb

    return model_package["metadata"]

def generate_model_id(model_name: str, variant: str):
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"{model_name}-{variant}-{ts}"

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

def create_pelatihan_knn_service():
    try:
        # Pastikan load_and_preprocess mengembalikan scaler
        X, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess()

        # weighting_features harus mengembalikan (variants_dict, mi_scores_norm, features)
        feature_variants, mi_scores_norm, features = weighting_features(
            X, X_train_scaled, X_test_scaled, y_train
        )

        results = []

        for varian, (X_train_variant, X_test_variant) in feature_variants.items():
            # Gunakan penulisan parameter yang jelas
            model = KNeighborsClassifier(n_neighbors=5)

            # Panggil train_and_save dengan semua argumen yang dibutuhkan
            res = train_and_save(
                model_name="knn",
                model=model,
                scaler=scaler,
                mi_scores_norm=mi_scores_norm,
                features=features,
                X_train=X_train_variant,
                y_train=y_train,
                variant_name=varian,
                X_test=X_test_variant,
                y_test=y_test
            )

            results.append(res)

        return {
            "error": False,
            "message": "Model KNN berhasil dibuat.",
            "data": results
        }

    except Exception as e:
        return {"error": True, "message": str(e)}

def create_pelatihan_nb_service():
    try:
        # Pastikan load_and_preprocess mengembalikan scaler
        X, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess()

        # weighting_features mengembalikan (variants, mi_scores_norm, features)
        feature_variants, mi_scores_norm, features = weighting_features(
            X, X_train_scaled, X_test_scaled, y_train
        )

        results = []

        for varian, (X_train_variant, X_test_variant) in feature_variants.items():

            # Gunakan Gaussian Naive Bayes
            model = GaussianNB()

            # Panggil train_and_save dengan parameter lengkap
            res = train_and_save(
                model_name="nb",
                model=model,
                scaler=scaler,
                mi_scores_norm=mi_scores_norm,
                features=features,
                X_train=X_train_variant,
                y_train=y_train,
                variant_name=varian,
                X_test=X_test_variant,
                y_test=y_test
            )

            results.append(res)

        return {
            "error": False,
            "message": "Model Naive Bayes berhasil dibuat.",
            "data": results
        }

    except Exception as e:
        return {"error": True, "message": str(e)}

def create_pelatihan_lr_service():
    try:
        # Pastikan load_and_preprocess mengembalikan scaler
        X, X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess()

        # weighting_features mengembalikan (variants, mi_scores_norm, features)
        feature_variants, mi_scores_norm, features = weighting_features(
            X, X_train_scaled, X_test_scaled, y_train
        )

        results = []

        for varian, (X_train_variant, X_test_variant) in feature_variants.items():

            # Model Logistic Regression
            model = LogisticRegression(
                max_iter=500,
                solver="lbfgs"
            )

            # Panggil train_and_save
            res = train_and_save(
                model_name="lr",
                model=model,
                scaler=scaler,
                mi_scores_norm=mi_scores_norm,
                features=features,
                X_train=X_train_variant,
                y_train=y_train,
                variant_name=varian,
                X_test=X_test_variant,
                y_test=y_test
            )

            results.append(res)

        return {
            "error": False,
            "message": "Model Logistic Regression berhasil dibuat.",
            "data": results
        }

    except Exception as e:
        return {"error": True, "message": str(e)}

def create_prediksi_knn_service(input_data: dict):
    try:
        # Tentukan path model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "knn", "mi.pkl")

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

def create_prediksi_lr_service(input_data: dict):
    try:
        import os, pickle, numpy as np, pandas as pd
        from math import exp

        # ======================================================
        # Load model
        # ======================================================
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "lr", "relieff.pkl")

        if not os.path.exists(model_path):
            return {
                "status": False,
                "message": "Model Logistic Regression tidak ditemukan.",
                "data": None
            }

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        mi_scores_norm = model_data["mi_scores_norm"]
        features = model_data["features"]

        # ======================================================
        # Validasi input
        # ======================================================
        missing_features = [f for f in features if f not in input_data]
        if missing_features:
            return {
                "status": False,
                "message": f"Input tidak lengkap. Fitur yang hilang: {missing_features}",
                "data": None
            }

        # ======================================================
        # Persiapan data
        # ======================================================
        input_df = pd.DataFrame([input_data])[features]
        scaled_input = scaler.transform(input_df)
        weighted_input = scaled_input * mi_scores_norm

        # ======================================================
        # Prediksi
        # ======================================================
        prediction = int(model.predict(weighted_input)[0])
        probability = float(model.predict_proba(weighted_input)[0][1])

        # ======================================================
        # INTERPRETASI TEKNIS (log-odds & odds ratio)
        # ======================================================
        coef = model.coef_[0]
        logit_score = float(np.dot(weighted_input[0], coef) + model.intercept_[0])

        kontribusi_fitur = []
        for i, fitur in enumerate(features):
            kontribusi = weighted_input[0][i] * coef[i]
            kontribusi_fitur.append({
                "fitur": fitur,
                "nilai_input": float(input_df.iloc[0, i]),
                "koefisien": float(coef[i]),
                "kontribusi_log_odds": float(kontribusi),
                "odds_ratio": float(exp(coef[i]))
            })

        # Urutkan kontribusi
        kontribusi_sorted = sorted(
            kontribusi_fitur,
            key=lambda x: abs(x["kontribusi_log_odds"]),
            reverse=True
        )

        faktor_risiko = [f for f in kontribusi_sorted if f["kontribusi_log_odds"] > 0][:5]
        faktor_protektif = [f for f in kontribusi_sorted if f["kontribusi_log_odds"] < 0][:5]

        # ======================================================
        # 🩺 INTERPRETASI KLINIS (Bahasa Dokter)
        # ======================================================
        faktor_risiko_nama = [f["fitur"].replace("_", " ") for f in faktor_risiko]
        faktor_protektif_nama = [f["fitur"].replace("_", " ") for f in faktor_protektif]

        if probability >= 0.7:
            tingkat_risiko = "tinggi"
        elif probability >= 0.5:
            tingkat_risiko = "sedang"
        else:
            tingkat_risiko = "rendah"

        interpretasi_klinis = (
            f"Pasien diprediksi memiliki risiko penyakit jantung dengan probabilitas "
            f"sekitar {round(probability * 100, 1)}%, yang termasuk dalam kategori "
            f"risiko {tingkat_risiko}. "
        )

        if faktor_risiko_nama:
            interpretasi_klinis += (
                "Faktor utama yang berkontribusi terhadap peningkatan risiko meliputi "
                + ", ".join(faktor_risiko_nama[:3]) + ". "
            )

        if faktor_protektif_nama:
            interpretasi_klinis += (
                "Di sisi lain, terdapat faktor yang bersifat protektif, yaitu "
                + ", ".join(faktor_protektif_nama[:2]) + ", yang membantu menurunkan risiko. "
            )

        interpretasi_klinis += (
            "Hasil ini dapat digunakan sebagai alat bantu pendukung keputusan klinis "
            "dan tidak menggantikan diagnosis dokter."
        )

        # ======================================================
        # RESPONSE
        # ======================================================
        return {
            "prediction": prediction,
            "label": "Berisiko Penyakit Jantung" if prediction == 1 else "Tidak Berisiko",
            "probability": f"{round(probability * 100, 1)}%",

            "interpretasi_akademis": {
                "logit_score": logit_score,
                "top_faktor_peningkat_risiko": faktor_risiko,
                "top_faktor_penurun_risiko": faktor_protektif,
                "seluruh_kontribusi_fitur": kontribusi_sorted
            },

            "interpretasi_klinis": interpretasi_klinis
        }

    except Exception as e:
        return {
            "status": False,
            "message": f"Terjadi kesalahan saat prediksi: {e}",
            "data": None
        }

def create_prediksi_nb_service(input_data: dict):
    try:
        # Tentukan path model variant MI
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "nb", "mi.pkl")

        if not os.path.exists(model_path):
            return {
                "status": False,
                "message": "Model Naive Bayes tidak ditemukan. Harap jalankan training terlebih dahulu.",
                "data": None
            }

        # 📦 Muat model, scaler, MI score, dan fitur
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        scaler = model_data["scaler"]
        mi_scores_norm = model_data["mi_scores_norm"]
        features = model_data["features"]

        # 🧾 Validasi kelengkapan input
        missing_features = [f for f in features if f not in input_data]
        if missing_features:
            return {
                "status": False,
                "message": f"Input tidak lengkap. Fitur yang hilang: {missing_features}",
                "data": None
            }

        # 📊 DataFrame dari input
        input_df = pd.DataFrame([input_data])[features]

        # 🔢 Scaling
        scaled_input = scaler.transform(input_df)

        # ⚖️ MI weighting
        weighted_input = scaled_input * mi_scores_norm

        # 🧠 Prediksi Naive Bayes
        prediction = model.predict(weighted_input)[0]

        # Naive Bayes SELALU punya predict_proba
        probability = float(model.predict_proba(weighted_input)[0][1])

        return {
            "status": True,
            "message": "Prediksi Naive Bayes berhasil dilakukan.",
            "data": {
                "prediction": int(prediction),
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

def read_pelatihan_service(algoritma):
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_dir = os.path.join(base_dir, "pkl", "models", algoritma)

        if not os.path.exists(model_dir):
            return {"error": f"Model {algoritma} tidak ditemukan."}

        order = {"dasar": 1, "mi": 2, "relieff": 3}

        result = []

        for file in os.listdir(model_dir):
            if not file.endswith(".pkl"):
                continue

            full_path = os.path.join(model_dir, file)

            with open(full_path, "rb") as f:
                pkg = pickle.load(f)

            metadata = pkg.get("metadata", {})

            size_kb = round(os.path.getsize(full_path) / 1024, 2)

            metadata["file_size_kb"] = size_kb
            metadata["lokasi"] = full_path

            # KEY FIXED: varian (bukan variant)
            variant = metadata.get("varian", "").lower()

            metadata["sort_index"] = order.get(variant, 999)

            result.append(metadata)

        # Sorting benar sekarang
        result = sorted(result, key=lambda x: x["sort_index"])

        return {
            "algoritma": algoritma,
            "models": result
        }

    except Exception as e:
        return {"error": str(e)}

def read_evaluasi_model_service():
    try:
        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset-1-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        # Pisahkan fitur dan target
        X = df.drop("Status_Penyakit_Jantung", axis=1)
        y = df["Status_Penyakit_Jantung"]

        # Split sebelum scaling
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # 1. Mutual Information
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
        mi_df = pd.DataFrame({'Fitur': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Normalisasi skor MI ke [0,1]
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())

        X_train_mi = X_train_scaled * mi_scores_norm
        X_test_mi = X_test_scaled * mi_scores_norm

        # 2. ReliefF
        relief = ReliefF(n_neighbors=10, n_features_to_select=X.shape[1])
        relief.fit(X_train_scaled, y_train.reset_index(drop=True))  # Hindari KeyError
        relief_scores = relief.feature_importances_

        # Buat dataframe seperti MI
        relief_df = pd.DataFrame({
            'Fitur': X.columns,
            'ReliefF Score': relief_scores
        }).sort_values(by='ReliefF Score', ascending=False)


        X_train_relief = X_train_scaled * relief_scores
        X_test_relief = X_test_scaled * relief_scores

        X_train_relief = X_train_scaled * relief_scores
        X_test_relief = X_test_scaled * relief_scores

        # --- Evaluasi semua kombinasi model ---
        results = {}

        # KNN
        results["KNN Tanpa Pembobotan"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["KNN MI Pembobotan"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_mi, y_train).predict(X_test_mi))
        results["KNN ReliefF Pembobotan"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_relief, y_train).predict(X_test_relief))

        # Naive Bayes
        results["NB Tanpa Pembobotan"] = get_metrics(y_test, GaussianNB().fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["NB MI Pembobotan"] = get_metrics(y_test, GaussianNB().fit(X_train_mi, y_train).predict(X_test_mi))
        results["NB ReliefF Pembobotan"] = get_metrics(y_test, GaussianNB().fit(X_train_relief, y_train).predict(X_test_relief))

        # Logistic Regression
        results["LR Tanpa Pembobotan"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["LR MI Pembobotan"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_mi, y_train).predict(X_test_mi))
        results["LR ReliefF Pembobotan"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_relief, y_train).predict(X_test_relief))

        # Buat DataFrame hasil
        df_results = pd.DataFrame(results).T.round(4)
        df_results.insert(0, "Model", df_results.index)
        df_results.reset_index(drop=True, inplace=True)

        # Tentukan folder tujuan relatif terhadap lokasi file service ini
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(base_dir, "pkl", "evaluations")

        # Pastikan folder 'data' ada
        os.makedirs(save_dir, exist_ok=True)

        return df_results.to_dict(orient="records")
    except Exception as e:
        return e

def read_peringkat_fitur_service():
    try:
        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset-1-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        # Pisahkan fitur dan target
        X = df.drop("Status_Penyakit_Jantung", axis=1)
        y = df["Status_Penyakit_Jantung"]

        # Split sebelum scaling
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        

        # -----------------------------
        # 1. Hitung Mutual Information
        # -----------------------------
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)

        mi_df = (
            pd.DataFrame({
                "fitur": X.columns,
                "nilai_mi": mi_scores
            })
            .sort_values(by="nilai_mi", ascending=False)
            .reset_index(drop=True)
        )

        # -----------------------------
        # 2. Hitung ReliefF
        # -----------------------------
        relief = ReliefF(
            n_neighbors=10,
            n_features_to_select=X.shape[1]
        )
        relief.fit(X_train_scaled, y_train.reset_index(drop=True)) 

        relief_scores = relief.feature_importances_

        relief_df = (
            pd.DataFrame({
                "fitur": X.columns,
                "nilai_relieff": relief_scores
            })
            .sort_values(by="nilai_relieff", ascending=False)
            .reset_index(drop=True)
        )

        # -----------------------------
        # Return hasil saja (MI & ReliefF)
        # -----------------------------
        return {
            "peringkat_mutual_information": mi_df.to_dict(orient="records"),
            "peringkat_relieff": relief_df.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}