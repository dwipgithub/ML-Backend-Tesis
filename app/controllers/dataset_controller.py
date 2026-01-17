import pandas as pd
import numpy as np
import os

def read():
    try:
        # ======================================================
        # MEMUAT DATASET
        # ======================================================
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        dataset_path = os.path.join(base_dir, "data", "dataset.csv")

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
            base_dir, "data", "dataset-cleaned.csv"
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
        dataset_path = os.path.join(base_dir, "data", "dataset-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        distribution = df["Status_Penyakit_Jantung"].value_counts().sort_index()

        return distribution
    except Exception as e:
        return e

def statistic():
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        dataset_path = os.path.join(base_dir, "data", "dataset-cleaned.csv")
        df = pd.read_csv(dataset_path, sep=';')

        stats = df.describe().transpose()
        return stats.to_dict(orient="index")
    except Exception as e:
        return e