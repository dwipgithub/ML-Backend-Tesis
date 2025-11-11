import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

def create_knn_service():
    try:
        # Load dataset
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset.csv")
        df = pd.read_csv(dataset_path)

        # Pisahkan fitur dan target
        X = df.drop("Status_Penyakit_Jantung", axis=1)
        y = df["Status_Penyakit_Jantung"]

        # Split sebelum scaling
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_test_scaled = scaler.transform(X_test_raw)

        # Mutual Information
        mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
        mi_df = pd.DataFrame({'Fitur': X.columns, 'MI Score': mi_scores})
        mi_df = mi_df.sort_values(by='MI Score', ascending=False)

        mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())
        X_train_mi = X_train_scaled * mi_scores_norm
        
        model = KNeighborsClassifier(n_neighbors=5)
        model.fit(X_train_mi, y_train)

        # Buat folder tujuan jika belum ada → project_root/models/
        save_dir = os.path.join(base_dir, "pkl", "models")
        os.makedirs(save_dir, exist_ok=True)

        # Path lengkap file model
        model_path = os.path.join(save_dir, "knn.pkl")

        model_data = { 
            "model": model, 
            "scaler": scaler, 
            "mi_scores_norm":mi_scores_norm, 
            "features": list(X.columns) 
        }

        with open( model_path, "wb") as f: 
            pickle.dump(model_data, f) 
        
        return list(X.columns)
    except Exception as e:
        return e
    
def read_knn_service():
    try:
        # Cari path model KNN
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, "pkl", "models", "knn.pkl")

        # Cek apakah file model ada
        if not os.path.exists(model_path):
            return None

        # Load isi model dari file pickle
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        # Kembalikan hasil pembacaan model
        # return {
        #     "status": True,
        #     "message": "Model KNN berhasil dimuat.",
        #     "features": model_data["features"],
        #     "mi_scores_norm": model_data["mi_scores_norm"].tolist(),  # ubah numpy array ke list agar bisa di-JSON-kan
        #     "model_type": type(model_data["model"]).__name__
        # }

        return model_data

    except Exception as e:
        return {
            "status": False,
            "message": f"Gagal membaca model: {e}"
        }