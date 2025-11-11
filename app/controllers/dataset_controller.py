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