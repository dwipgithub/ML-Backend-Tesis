import pandas as pd
import numpy as np
import os

def create():
    os.makedirs("data", exist_ok=True)

    np.random.seed(42)
    n = 850  # jumlah data

    # 🔹 Distribusi target (realistic)
    status = np.random.choice([0, 1], size=n, p=[0.60, 0.40])  # 60% sehat, 40% jantung

    # --- Data dasar ---
    data = {
        "Usia": np.random.normal(46 + 8 * status, 11, n).astype(int),
        "Jenis_Kelamin": np.random.choice([0, 1], n, p=[0.45, 0.55]),  # 0=female, 1=male
        "Status_Penyakit_Jantung": status
    }

    # --- Faktor risiko ---
    data["Riwayat_Hipertensi"] = np.where(
        status == 1,
        np.random.choice([0, 1], n, p=[0.35, 0.65]),
        np.random.choice([0, 1], n, p=[0.69, 0.31])
    )

    data["Riwayat_Diabetes"] = np.where(
        status == 1,
        np.random.choice([0, 1], n, p=[0.45, 0.55]),
        np.random.choice([0, 1], n, p=[0.63, 0.37])
    )

    data["Riwayat_Merokok"] = np.where(
        status == 1,
        np.random.choice([0, 1], n, p=[0.35, 0.65]),
        np.random.choice([0, 1], n, p=[0.6, 0.4])
    )

    data["Riwayat_Jantung_Keluarga"] = np.where(
        status == 1,
        np.random.choice([0, 1], n, p=[0.3, 0.7]),
        np.random.choice([0, 1], n, p=[0.75, 0.25])
    )

    # ⚖️ BMI
    data["BMI"] = np.round(
        np.random.normal(30 + 1.5 * status + 0.8 * data["Riwayat_Diabetes"], 3.5, n), 1
    )
    data["BMI"] = np.clip(data["BMI"], 17, 40)

    # 💓 Tekanan darah
    tds = (
        120
        + 0.5 * np.maximum(0, data["Usia"] - 40)
        + 9 * data["Riwayat_Hipertensi"]
        + 4 * data["Riwayat_Merokok"]
        + 6 * status
        + np.random.normal(0, 8, n)
    )
    tdd = (
        80
        + 0.3 * np.maximum(0, data["Usia"] - 40)
        + 6 * data["Riwayat_Hipertensi"]
        + 3 * data["Riwayat_Merokok"]
        + 5 * status
        + np.random.normal(0, 7, n)
    )
    data["Tekanan_Darah_Sistolik"] = np.clip(tds, 90, 200).astype(int)
    data["Tekanan_Darah_Diastolik"] = np.clip(tdd, 55, 130).astype(int)

    # 🧬 Kadar lemak dan gula darah
    data["Kadar_LDL"] = np.clip(
        np.random.normal(110 + 18 * status + 7 * data["Riwayat_Merokok"], 18, n),
        60, 250
    ).astype(int)

    data["Kadar_HDL"] = np.clip(
        np.where(
            data["Jenis_Kelamin"] == 1,
            np.random.normal(55 - 6 * status - 2 * data["Riwayat_Merokok"], 8, n),
            np.random.normal(60 - 6 * status - 2 * data["Riwayat_Merokok"], 8, n)
        ),
        25, 85
    ).astype(int)

    data["Kolesterol_Total"] = np.clip(
        np.random.normal(180 + 22 * status + 9 * data["Riwayat_Merokok"], 24, n),
        120, 330
    ).astype(int)

    data["Gula_Darah_Puasa"] = np.clip(
        np.random.normal(
            90 + 10 * data["Riwayat_Diabetes"] + 11 * status, 22, n
        ),
        60, 250
    ).astype(int)

    # 💓 Denyut nadi
    data["Denyut_Nadi"] = np.clip(
        np.random.normal(75 + 2 * status + 1.3 * data["Riwayat_Merokok"], 7, n),
        55, 120
    ).astype(int)

    # 🧪 Tambahkan noise alami moderat
    for col in [
        "Tekanan_Darah_Sistolik", "Tekanan_Darah_Diastolik",
        "Kadar_LDL", "Kolesterol_Total", "Gula_Darah_Puasa", "BMI"
    ]:
        noise = np.random.normal(0, 9, n)
        data[col] = np.clip(np.array(data[col]) + noise, 0, None)

    # --- Konversi ke DataFrame ---
    df = pd.DataFrame(data)
    df["Usia"] = df["Usia"].clip(25, 85)

    # 🔹 Pindahkan kolom target ke posisi terakhir
    target_col = "Status_Penyakit_Jantung"
    columns_order = [c for c in df.columns if c != target_col] + [target_col]
    df = df[columns_order]

    # Simpan hasil
    output_path = "data/dataset.csv"
    df.to_csv(output_path, index=False)

def read():
    try:
        # ======================================================
        # MEMUAT DATASET
        # ======================================================
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        dataset_path = os.path.join(base_dir, "data", "dataset-1.csv")

        df = pd.read_csv(dataset_path, sep=';')

        # ======================================================
        # KONDISI DATASET SEBELUM PEMBERSIHAN DATA
        # ======================================================
        total_data_awal = len(df)
        jumlah_duplikat_awal = int(df.duplicated().sum())
        jumlah_data_hilang_awal = df.isnull().sum().to_dict()

        # Deteksi pencilan (outlier) sebelum pembersihan data
        # menggunakan metode Interquartile Range (IQR)
        outlier_awal = {}
        kolom_numerik = df.select_dtypes(include=[np.number]).columns

        for kolom in kolom_numerik:
            Q1 = df[kolom].quantile(0.25)
            Q3 = df[kolom].quantile(0.75)
            IQR = Q3 - Q1

            batas_bawah = Q1 - 1.5 * IQR
            batas_atas = Q3 + 1.5 * IQR

            outlier_awal[kolom] = int(
                df[(df[kolom] < batas_bawah) | (df[kolom] > batas_atas)].shape[0]
            )

        # ======================================================
        # PROSES PEMBERSIHAN DATA (DATA CLEANING)
        # ======================================================
        df_clean = df.copy()

        # 1. Menghapus data duplikat (menyisakan satu rekaman)
        df_clean = df_clean.drop_duplicates()

        # 2. Menangani nilai hilang menggunakan metode modus
        for kolom in df_clean.columns:
            if df_clean[kolom].isnull().sum() > 0:
                modus = df_clean[kolom].mode()
                if not modus.empty:
                    df_clean[kolom] = df_clean[kolom].fillna(modus[0])

        # 3. Menangani pencilan (outlier) menggunakan metode IQR (capping)
        for kolom in kolom_numerik:
            Q1 = df_clean[kolom].quantile(0.25)
            Q3 = df_clean[kolom].quantile(0.75)
            IQR = Q3 - Q1

            batas_bawah = Q1 - 1.5 * IQR
            batas_atas = Q3 + 1.5 * IQR

            df_clean[kolom] = np.where(
                df_clean[kolom] < batas_bawah, batas_bawah,
                np.where(df_clean[kolom] > batas_atas, batas_atas, df_clean[kolom])
            )

        # 4. Pengkodean variabel kategorikal menjadi biner (0 dan 1)
        kolom_biner = [
            "Jenis_Kelamin",
            "Riwayat_Hipertensi",
            "Riwayat_Diabetes",
            "Riwayat_Merokok",
            "Status_Penyakit_Jantung"
        ]

        for kolom in kolom_biner:
            if kolom in df_clean.columns:
                df_clean[kolom] = df_clean[kolom].map({
                    "Ya": 1, "Tidak": 0,
                    "L": 1, "P": 0
                })

        # ======================================================
        # KONDISI DATASET SETELAH PEMBERSIHAN DATA
        # ======================================================
        total_data_akhir = len(df_clean)
        jumlah_duplikat_akhir = int(df_clean.duplicated().sum())
        jumlah_data_hilang_akhir = df_clean.isnull().sum().to_dict()

        outlier_akhir = {}
        for kolom in kolom_numerik:
            Q1 = df_clean[kolom].quantile(0.25)
            Q3 = df_clean[kolom].quantile(0.75)
            IQR = Q3 - Q1

            batas_bawah = Q1 - 1.5 * IQR
            batas_atas = Q3 + 1.5 * IQR

            outlier_akhir[kolom] = int(
                df_clean[(df_clean[kolom] < batas_bawah) | (df_clean[kolom] > batas_atas)].shape[0]
            )

        # ======================================================
        # MENYIMPAN DATASET HASIL PEMBERSIHAN KE FILE CSV
        # ======================================================
        path_dataset_bersih = os.path.join(
            base_dir, "data", "dataset-1-cleaned.csv"
        )
        df_clean.to_csv(path_dataset_bersih, sep=';', index=False)

        # ======================================================
        # PENYESUAIAN FORMAT JSON (NaN, Inf)
        # ======================================================
        df_sebelum = df.astype(object).replace(
            {np.nan: None, np.inf: None, -np.inf: None}
        )
        df_sesudah = df_clean.astype(object).replace(
            {np.nan: None, np.inf: None, -np.inf: None}
        )

        # ======================================================
        # RESPONS DATA
        # ======================================================
        return {
            "ringkasan_dataset_sebelum_pembersihan": {
                "jumlah_seluruh_data": total_data_awal,
                "jumlah_data_duplikat": jumlah_duplikat_awal,
                "jumlah_data_hilang_per_kolom": jumlah_data_hilang_awal,
                "jumlah_outlier_per_kolom": outlier_awal
            },

            "ringkasan_dataset_setelah_pembersihan": {
                "jumlah_data": total_data_akhir,
                "jumlah_data_duplikat": jumlah_duplikat_akhir,
                "jumlah_data_hilang_per_kolom": jumlah_data_hilang_akhir,
                "jumlah_outlier_per_kolom": outlier_akhir
            },

            "dataset_sebelum_pembersihan": df_sebelum.to_dict(orient="records"),
            "dataset_setelah_pembersihan": df_sesudah.to_dict(orient="records")
        }

    except Exception as e:
        return {
            "status": False,
            "error": str(e)
        }

def read_class_distribution():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset-1-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        distribution = df["Status_Penyakit_Jantung"].value_counts().sort_index()

        return distribution
    except Exception as e:
        return e

def statistic():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset-1-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        stats = df.describe().transpose()
        return stats.to_dict(orient="index")
    except Exception as e:
        return e