import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

def create_logreg_service():
    try:
        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset.csv")
        df = pd.read_csv(dataset_path)

        # Pisahkan fitur dan target
        X = df.drop("Status_Penyakit_Jantung", axis=1)
        y = df["Status_Penyakit_Jantung"]

        # Split sebelum scaling
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Mutual Information
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
        mi_df = pd.DataFrame({'Fitur': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        # Normalisasi MI
        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())

        # Terapkan bobot MI ke fitur
        X_train_mi = X_train_scaled * mi_scores_norm

        # ============================
        # GANTIAN: MODEL LOGISTIC REGRESSION
        # ============================
        model = LogisticRegression(
            max_iter=500,      # agar konvergen stabil
            solver='lbfgs',
            class_weight='balanced',   # lebih bagus untuk dataset tidak seimbang
        )

        model.fit(X_train_mi, y_train)

        # Buat folder tujuan project_root/pkl/models/
        save_dir = os.path.join(base_dir, "pkl", "models")
        os.makedirs(save_dir, exist_ok=True)

        # Path file model
        model_path = os.path.join(save_dir, "logreg.pkl")

        # Simpan data model
        model_data = {
            "model": model,
            "scaler": scaler,
            "mi_scores_norm": mi_scores_norm,
            "features": list(X.columns)
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        return list(X.columns)

    except Exception as e:
        return e


def read_logreg_service():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "logreg.pkl")

        if not os.path.exists(model_path):
            return None

        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        return model_data

    except Exception as e:
        return {
            "status": False,
            "message": f"Gagal membaca model: {e}"
        }
