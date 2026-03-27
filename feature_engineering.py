import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = Path().resolve()
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_FILE = DATASET_DIR / "cicids2017_procesado.parquet"
OUTPUT_CSV = DATASET_DIR / "cicids2017_procesado.csv"

def load_and_merge_data(dataset_dir):
    csv_files = sorted(Path(dataset_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {dataset_dir}")
    
    print(f"Encontrados {len(csv_files)} archivos CSV. Cargando...")
    dfs = []
    for file in csv_files:
        print(f" -> Cargando {file.name}")
        df = pd.read_csv(file)
        # Limpiar nombres de columnas (eliminar espacios extra)
        df.columns = [col.strip() for col in df.columns]
        dfs.append(df)
        
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n✅ Dataset consolidado con {merged_df.shape[0]:,} filas y {merged_df.shape[1]} columnas.")
    return merged_df

def clean_data(df):
    initial_rows = df.shape[0]
    
    # 2.1 Eliminar duplicados
    print("Eliminando filas duplicadas...")
    df.drop_duplicates(inplace=True)
    new_rows = df.shape[0]
    print(f" -> {initial_rows - new_rows:,} filas duplicadas eliminadas.")
    
    # 2.2 Reemplazar infinitos por NaN
    print("Manejando infinitos y valores nulos...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    
    # 2.3 Eliminar filas con NaN
    before_dropna = df.shape[0]
    df.dropna(inplace=True)
    print(f" -> {before_dropna - df.shape[0]:,} filas con NaNs o Infinitos eliminadas.")
    
    print(f"✅ Limpieza completada. Filas actuales: {df.shape[0]:,}")
    return df

def encode_labels(df):
    print("Limpiando y codificando etiquetas...")
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].str.replace(r"Web Attack.*Brute Force", "Web Attack - Brute Force", regex=True)
    df["Label"] = df["Label"].str.replace(r"Web Attack.*XSS", "Web Attack - XSS", regex=True)
    df["Label"] = df["Label"].str.replace(r"Web Attack.*Sql Injection", "Web Attack - Sql Injection", regex=True)

    # Generación Label_Binary
    df["Label_Binary"] = np.where(df["Label"] == "BENIGN", 0, 1)

    # Generación Label_Multiclass
    le = LabelEncoder()
    df["Label_Multiclass"] = le.fit_transform(df["Label"])

    class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("\nMapeo de clases a números:")
    for k, v in class_mapping.items():
        print(f" {v} -> {k}")
        
    return df, le

def remove_redundant_features(df):
    # Columnas constantes
    constant_features = [
        "Bwd PSH Flags", "Bwd URG Flags", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
        "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"
    ]

    # Columnas redundantes
    redundant_features = [
        "Fwd Header Length.1",  # Redundante con Fwd Header Length
        "Subflow Fwd Packets",  # Redundante con Total Fwd Packets
        "Subflow Bwd Packets"   # Redundante con Total Backward Packets
    ]

    cols_to_drop = [col for col in constant_features + redundant_features if col in df.columns]
    if cols_to_drop:
        print(f"Eliminando columnas constantes y redundantes: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)
        
    return df

def select_top_features(df, target_col="Label_Binary", n_features=20):
    print("\nGenerando muestra para cálculo de Mutual Information...")
    sample_df = resample(df, n_samples=100000, random_state=42, stratify=df["Label_Multiclass"])

    features = sample_df.drop(columns=["Label", "Label_Binary", "Label_Multiclass"])
    target = sample_df[target_col]

    print("Calculando Mutual Information...")
    start_mi = time.time()
    mi_scores = mutual_info_classif(features, target, random_state=42)
    print(f"MI calculado en {time.time() - start_mi:.2f} segundos")

    mi_df = pd.DataFrame({"Feature": features.columns, "Mutual_Info": mi_scores})
    mi_df = mi_df.sort_values(by="Mutual_Info", ascending=False).reset_index(drop=True)

    top_features = mi_df.head(40)["Feature"].tolist()
    print("\nTop características basadas en Mutual Information:")
    print(mi_df.head(10))

    # Filtrado por correlación
    print("\nFiltrando características altamente correlacionadas (>0.95)...")
    corr_matrix = df[top_features].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.95)]
    
    print(f"Características eliminadas por superponerse: {to_drop_corr}")

    final_features = [f for f in top_features if f not in to_drop_corr]
    final_features = final_features[:n_features]

    print(f"\n✅ Top {len(final_features)} Características Finales Seleccionadas:")
    for f in final_features:
        print(f" - {f}")
        
    return final_features

def main():
    start_time = time.time()
    
    # 1. Cargar
    df = load_and_merge_data(DATASET_DIR)
    
    # 2. Limpiar
    df = clean_data(df)
    
    # 3. Label Encoder
    df, le = encode_labels(df)
    
    # 4. Remover Redundantes
    df = remove_redundant_features(df)
    
    # 5. Seleccionar Top Features
    final_features = select_top_features(df, n_features=20)
    
    # 6. Exportar
    cols_to_keep = final_features + ["Label", "Label_Binary", "Label_Multiclass"]
    print(f"\nExtrayendo subset optimizado ({len(cols_to_keep)} columnas)...")
    df_final = df[cols_to_keep].copy()

    # Optimización de tipos de datos para Parquet / Memoria DL
    for col in df_final.columns:
        if col == "Label":
            continue
        if df_final[col].dtype == 'float64':
            df_final[col] = df_final[col].astype('float32')
        elif df_final[col].dtype == 'int64':
            if df_final[col].max() < 2147483647:
                df_final[col] = df_final[col].astype('int32')

    print("Guardando archivo en formato Parquet...")
    df_final.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Archivo guardado exitosamente: {OUTPUT_FILE}")
    
    # df_final.to_csv(OUTPUT_CSV, index=False)
    # print(f"✅ Archivo guardado exitosamente: {OUTPUT_CSV}")

    print(f"\nProceso finalizado en {time.time() - start_time:.2f} segundos.")

if __name__ == '__main__':
    main()
