import os
import pickle
import pandas as pd
from skrebate import ReliefF
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred)
    }

def create_evaluasi_model_service():
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
        results["KNN Tanpa Weighting"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["KNN MI Weighting"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_mi, y_train).predict(X_test_mi))
        results["KNN ReliefF Weighting"] = get_metrics(y_test, KNeighborsClassifier(5).fit(X_train_relief, y_train).predict(X_test_relief))

        # Naive Bayes
        results["NB Tanpa Weighting"] = get_metrics(y_test, GaussianNB().fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["NB MI Weighting"] = get_metrics(y_test, GaussianNB().fit(X_train_mi, y_train).predict(X_test_mi))
        results["NB ReliefF Weighting"] = get_metrics(y_test, GaussianNB().fit(X_train_relief, y_train).predict(X_test_relief))

        # Logistic Regression
        results["LR Tanpa Weighting"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train).predict(X_test_scaled))
        results["LR MI Weighting"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_mi, y_train).predict(X_test_mi))
        results["LR ReliefF Weighting"] = get_metrics(y_test, LogisticRegression(max_iter=1000, random_state=42).fit(X_train_relief, y_train).predict(X_test_relief))

        # Buat DataFrame hasil
        df_results = pd.DataFrame(results).T.round(4)
        df_results.insert(0, "Model", df_results.index)
        df_results.reset_index(drop=True, inplace=True)

        # Tentukan folder tujuan relatif terhadap lokasi file service ini
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        save_dir = os.path.join(base_dir, "pkl", "evaluations")

        # Pastikan folder 'data' ada
        os.makedirs(save_dir, exist_ok=True)

        # # Buat nama file
        # filename = os.path.join(save_dir, "hasil_evaluasi_model.pkl")

        # # Simpan DataFrame hasil evaluasi ke file pickle
        # with open(filename, "wb") as f:
        #     pickle.dump(df_results, f)

        return df_results.to_dict(orient="records")
    except Exception as e:
        return e

def read_evaluasi_model_service():
    try:
        # Dapatkan path absolut dari root project
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        file_path = os.path.join(base_dir, "pkl", "evaluations", "hasil_evaluasi_model.pkl")

        if not os.path.exists(file_path):
            return None
        
        with open(file_path, "rb") as f:
            hasil = pickle.load(f)

        return hasil.to_dict(orient="records")
    except Exception as e:
        return e