import os
import pandas as pd
from skrebate import ReliefF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

def create_peringkat_fitur_service():
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
