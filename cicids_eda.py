from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown, display
from sklearn.feature_selection import mutual_info_classif


RANDOM_STATE = 42


FEATURE_DESCRIPTIONS = {
    "Destination Port": {
        "category": "Servicio objetivo",
        "description": "Puerto de destino observado en el flujo.",
        "security_value": "Ayuda a detectar escaneos y ataques concentrados en servicios concretos como HTTP, FTP, SSH o puertos efimeros.",
    },
    "Flow Duration": {
        "category": "Temporalidad",
        "description": "Duracion total del flujo desde el primer hasta el ultimo paquete.",
        "security_value": "Separa ataques muy breves y repetitivos, como scans o floods, de sesiones legitimas mas sostenidas.",
    },
    "Flow Bytes/s": {
        "category": "Volumen",
        "description": "Tasa de bytes por segundo del flujo.",
        "security_value": "Captura picos de trafico y ayuda a identificar flujos anormalmente intensos o inestables.",
    },
    "Flow Packets/s": {
        "category": "Volumen",
        "description": "Tasa de paquetes por segundo del flujo.",
        "security_value": "Es especialmente util para evidenciar DDoS, DoS y trafico automatizado con rafagas intensas.",
    },
    "Flow IAT Mean": {
        "category": "Temporalidad",
        "description": "Promedio del tiempo entre llegadas sucesivas de paquetes en el flujo.",
        "security_value": "Distingue patrones regulares de automatizacion frente a trafico humano mas variable.",
    },
    "Flow IAT Std": {
        "category": "Temporalidad",
        "description": "Desviacion estandar del tiempo entre llegadas dentro del flujo.",
        "security_value": "Mide la regularidad del ritmo del trafico; ataques scripted suelen ser mas uniformes.",
    },
    "Flow IAT Max": {
        "category": "Temporalidad",
        "description": "Mayor tiempo entre llegadas dentro del flujo.",
        "security_value": "Ayuda a encontrar flujos con pausas largas o comportamiento bursty caracteristico de ciertos ataques.",
    },
    "Fwd IAT Total": {
        "category": "Temporalidad",
        "description": "Tiempo total entre llegadas en direccion forward.",
        "security_value": "Resume el patron temporal del emisor y suele cambiar en ataques de fuerza bruta, scan y DoS.",
    },
    "Fwd IAT Mean": {
        "category": "Temporalidad",
        "description": "Promedio del tiempo entre llegadas en direccion forward.",
        "security_value": "Permite comparar la cadencia con la que el origen envia paquetes hacia el destino.",
    },
    "Fwd IAT Max": {
        "category": "Temporalidad",
        "description": "Maximo tiempo entre llegadas en direccion forward.",
        "security_value": "Resalta pausas largas del emisor, utiles para distinguir trafico benigno de ataques intermitentes.",
    },
    "Total Length of Fwd Packets": {
        "category": "Volumen",
        "description": "Total de bytes enviados hacia adelante durante el flujo.",
        "security_value": "Indica cuanta carga envia el origen; ayuda a diferenciar scans, bruteforce y trafico normal.",
    },
    "Total Length of Bwd Packets": {
        "category": "Volumen",
        "description": "Total de bytes enviados de regreso en el flujo.",
        "security_value": "Mide la respuesta del destino y ayuda a capturar asimetrias de trafico sospechosas.",
    },
    "Fwd Header Length": {
        "category": "Cabeceras y control",
        "description": "Longitud acumulada de cabeceras en direccion forward.",
        "security_value": "Suele correlacionar con la cantidad y tipo de paquetes de control emitidos por el origen.",
    },
    "Bwd Header Length": {
        "category": "Cabeceras y control",
        "description": "Longitud acumulada de cabeceras en direccion backward.",
        "security_value": "Ayuda a entender el peso del trafico de respuesta y la simetria del flujo.",
    },
    "Fwd Packets/s": {
        "category": "Volumen",
        "description": "Paquetes por segundo emitidos en direccion forward.",
        "security_value": "Es util para reconocer rafagas del cliente durante scans, bruteforce y varios ataques de denegacion.",
    },
    "Bwd Packets/s": {
        "category": "Volumen",
        "description": "Paquetes por segundo emitidos en direccion backward.",
        "security_value": "Describe la intensidad de respuesta del destino y complementa la lectura de simetria del flujo.",
    },
    "Packet Length Mean": {
        "category": "Tamano de paquetes",
        "description": "Tamano promedio de los paquetes observados en el flujo.",
        "security_value": "Los ataques automatizados suelen mostrar perfiles de tamanos mas rigidos que el trafico benigno.",
    },
    "Packet Length Std": {
        "category": "Tamano de paquetes",
        "description": "Desviacion estandar del tamano de paquetes del flujo.",
        "security_value": "Mide variabilidad; secuencias muy uniformes pueden delatar automatizacion o floods.",
    },
    "Packet Length Variance": {
        "category": "Tamano de paquetes",
        "description": "Varianza del tamano de paquetes.",
        "security_value": "Complementa la dispersion y refuerza la separacion entre trafico estable y trafico anomalo.",
    },
    "Average Packet Size": {
        "category": "Tamano de paquetes",
        "description": "Promedio general de bytes por paquete del flujo.",
        "security_value": "Resume el perfil de tamano de la sesion y suele ser muy discriminante frente a varias familias de ataque.",
    },
    "Max Packet Length": {
        "category": "Tamano de paquetes",
        "description": "Mayor tamano de paquete observado en el flujo.",
        "security_value": "Permite distinguir flujos con paquetes grandes ocasionales frente a trafico corto y repetitivo.",
    },
    "Min Packet Length": {
        "category": "Tamano de paquetes",
        "description": "Menor tamano de paquete observado en el flujo.",
        "security_value": "Captura si predominan paquetes muy pequenos de control, algo comun en scans y ataques automatizados.",
    },
    "Fwd Packet Length Max": {
        "category": "Tamano de paquetes",
        "description": "Mayor tamano de paquete en direccion forward.",
        "security_value": "Ayuda a separar flujos donde el origen envia carga real frente a trafico de sondeo o control.",
    },
    "Init_Win_bytes_forward": {
        "category": "TCP",
        "description": "Tamano inicial de ventana TCP en direccion forward.",
        "security_value": "Puede reflejar comportamiento del sistema o herramienta que origina el trafico, util para perfilar ataques automatizados.",
    },
    "Down/Up Ratio": {
        "category": "Simetria",
        "description": "Relacion entre trafico descendente y ascendente.",
        "security_value": "Resume la asimetria del flujo; muchos ataques producen conversaciones muy desbalanceadas.",
    },
    "PSH Flag Count": {
        "category": "Banderas TCP",
        "description": "Numero de paquetes con bandera PSH dentro del flujo.",
        "security_value": "Aporta contexto sobre el patron de entrega de datos a la aplicacion y puede diferenciar trafico normal de scripts.",
    },
    "min_seg_size_forward": {
        "category": "TCP",
        "description": "Tamano minimo del segmento forward.",
        "security_value": "Ayuda a capturar flujos donde predominan segmentos de control o patrones muy pequenos repetidos.",
    },
}


def _markdown(text: str) -> None:
    display(Markdown(text))


def _dataset_files(dataset_dir: str | Path) -> list[Path]:
    dataset_path = Path(dataset_dir)
    files = sorted(dataset_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {dataset_path.resolve()}")
    return files


def _clean_labels(series: pd.Series) -> pd.Series:
    labels = series.astype(str).str.strip()
    labels = labels.str.replace(r"Web Attack.*Brute Force", "Web Attack - Brute Force", regex=True)
    labels = labels.str.replace(r"Web Attack.*XSS", "Web Attack - XSS", regex=True)
    labels = labels.str.replace(r"Web Attack.*Sql Injection", "Web Attack - Sql Injection", regex=True)
    return labels


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    if "Label" in df.columns:
        df["Label"] = _clean_labels(df["Label"])
    return df


def _label_only(path: Path) -> pd.Series:
    df = pd.read_csv(path, usecols=lambda col: col.strip() == "Label")
    df.columns = ["Label"]
    return _clean_labels(df["Label"])


def _set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def build_overview(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = _dataset_files(dataset_dir)
    file_summary_rows: list[dict[str, object]] = []
    label_totals: dict[str, int] = {}
    file_label_rows: list[dict[str, object]] = []
    total_rows = 0
    feature_count = None

    for path in files:
        labels = _label_only(path)
        counts = labels.value_counts().sort_values(ascending=False)
        rows = int(labels.shape[0])
        benign_rows = int(counts.get("BENIGN", 0))
        attack_counts = counts.drop(labels=["BENIGN"], errors="ignore")
        file_summary_rows.append(
            {
                "file": path.name,
                "rows": rows,
                "classes": int(counts.shape[0]),
                "benign_pct": round(100 * benign_rows / rows, 2),
                "attack_pct": round(100 * (rows - benign_rows) / rows, 2),
                "dominant_attack": attack_counts.index[0] if not attack_counts.empty else "None",
                "dominant_attack_rows": int(attack_counts.iloc[0]) if not attack_counts.empty else 0,
            }
        )

        for label, count in counts.items():
            label_totals[label] = label_totals.get(label, 0) + int(count)
            file_label_rows.append({"file": path.name, "label": label, "count": int(count)})

        total_rows += rows

        if feature_count is None:
            head = pd.read_csv(path, nrows=1)
            feature_count = len([col.strip() for col in head.columns]) - 1

    label_distribution = (
        pd.Series(label_totals, name="count")
        .sort_values(ascending=False)
        .rename_axis("label")
        .reset_index()
    )
    label_distribution["pct"] = (100 * label_distribution["count"] / total_rows).round(4)

    overview = pd.DataFrame(
        {
            "metric": [
                "Archivos CSV",
                "Registros totales",
                "Caracteristicas numericas y de red",
                "Clases de trafico",
            ],
            "value": [
                len(files),
                total_rows,
                feature_count,
                int(label_distribution.shape[0]),
            ],
        }
    )

    file_summary = pd.DataFrame(file_summary_rows).sort_values("rows", ascending=False).reset_index(drop=True)
    file_label_matrix = (
        pd.DataFrame(file_label_rows)
        .pivot(index="file", columns="label", values="count")
        .fillna(0)
        .astype(int)
        .sort_index()
    )
    return overview, file_summary, label_distribution, file_label_matrix


def build_quality_report(dataset_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:
    files = _dataset_files(dataset_dir)
    missing_total = None
    inf_total = None
    mins = None
    maxs = None
    duplicate_rows = []
    candidate_pairs = [
        ("Total Fwd Packets", "Subflow Fwd Packets"),
        ("Total Backward Packets", "Subflow Bwd Packets"),
        ("Fwd Header Length", "Fwd Header Length.1"),
    ]
    pair_status = {pair: True for pair in candidate_pairs}

    for path in files:
        df = _load_csv(path)
        numeric_df = df.select_dtypes(include=[np.number])
        file_missing = df.isna().sum()
        file_inf = pd.Series(np.isinf(numeric_df.to_numpy()).sum(axis=0), index=numeric_df.columns)
        file_min = numeric_df.min(skipna=True)
        file_max = numeric_df.max(skipna=True)

        missing_total = file_missing if missing_total is None else missing_total.add(file_missing, fill_value=0)
        inf_total = file_inf if inf_total is None else inf_total.add(file_inf, fill_value=0)
        mins = file_min if mins is None else pd.concat([mins, file_min], axis=1).min(axis=1, skipna=True)
        maxs = file_max if maxs is None else pd.concat([maxs, file_max], axis=1).max(axis=1, skipna=True)

        duplicate_rows.append(
            {
                "file": path.name,
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_pct": round(100 * df.duplicated().mean(), 2),
            }
        )

        for left, right in candidate_pairs:
            if pair_status[(left, right)] and not df[left].equals(df[right]):
                pair_status[(left, right)] = False

    quality = pd.DataFrame(
        {
            "feature": missing_total.index,
            "missing_values": missing_total.astype(int).values,
            "infinite_values": inf_total.reindex(missing_total.index, fill_value=0).astype(int).values,
        }
    )
    quality = quality[(quality["missing_values"] > 0) | (quality["infinite_values"] > 0)]
    quality = quality.sort_values(["infinite_values", "missing_values"], ascending=False).reset_index(drop=True)

    constant_features = mins[mins.eq(maxs)].index.tolist()

    redundant_rows = []
    for left, right in candidate_pairs:
        if pair_status[(left, right)]:
            redundant_rows.append({"feature_a": left, "feature_b": right})

    duplicates_df = pd.DataFrame(duplicate_rows).sort_values("duplicate_rows", ascending=False).reset_index(drop=True)
    redundant_df = pd.DataFrame(redundant_rows)
    return quality, duplicates_df, constant_features, redundant_df


def build_balanced_sample(
    dataset_dir: str | Path,
    per_file_cap: int = 4000,
    per_label_cap: int = 2000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    files = _dataset_files(dataset_dir)
    samples = []

    for path in files:
        df = _load_csv(path)
        per_file_parts = []
        for _, group in df.groupby("Label"):
            per_file_parts.append(group.sample(n=min(len(group), per_file_cap), random_state=random_state))
        samples.append(pd.concat(per_file_parts, ignore_index=True))

    sample_df = pd.concat(samples, ignore_index=True)
    balanced_parts = []
    for _, group in sample_df.groupby("Label"):
        balanced_parts.append(group.sample(n=min(len(group), per_label_cap), random_state=random_state))
    sample_df = pd.concat(balanced_parts, ignore_index=True)
    return sample_df


def rank_features(sample_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    numeric_df = sample_df.drop(columns=["Label"]).select_dtypes(include=[np.number])
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.fillna(numeric_df.median())

    constant_features = numeric_df.columns[numeric_df.nunique(dropna=True) <= 1].tolist()
    numeric_df = numeric_df.drop(columns=constant_features)

    duplicate_mask = numeric_df.T.duplicated(keep="first")
    duplicate_features = numeric_df.columns[duplicate_mask].tolist()
    numeric_df = numeric_df.loc[:, ~duplicate_mask]

    mi_scores = mutual_info_classif(numeric_df, sample_df["Label"], random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": numeric_df.columns, "mutual_info": mi_scores})
    mi_df = mi_df.sort_values("mutual_info", ascending=False).reset_index(drop=True)
    return mi_df, constant_features, duplicate_features


def build_feature_dictionary(sample_df: pd.DataFrame, mi_df: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    top_features = mi_df.head(top_n).copy()
    selected = top_features["feature"].tolist()

    numeric_subset = sample_df[selected].replace([np.inf, -np.inf], np.nan)
    traffic_flag = np.where(sample_df["Label"].eq("BENIGN"), "BENIGN", "ATTACK")
    medians = numeric_subset.groupby(pd.Index(traffic_flag)).median(numeric_only=True)
    stds = numeric_subset.std(numeric_only=True).replace(0, np.nan)
    shift = ((medians.loc["ATTACK"] - medians.loc["BENIGN"]).abs() / stds).fillna(0)

    description_rows = []
    for feature in selected:
        metadata = FEATURE_DESCRIPTIONS.get(
            feature,
            {
                "category": "Otras metricas",
                "description": "Metrica tecnica derivada del flujo de red.",
                "security_value": "Su utilidad se interpreta comparando como cambia entre trafico benigno y trafico de ataque.",
            },
        )
        description_rows.append(
            {
                "feature": feature,
                "category": metadata["category"],
                "description": metadata["description"],
                "security_value": metadata["security_value"],
                "mutual_info": round(float(top_features.loc[top_features["feature"] == feature, "mutual_info"].iloc[0]), 4),
                "attack_vs_benign_shift": round(float(shift[feature]), 4),
            }
        )

    return pd.DataFrame(description_rows).sort_values("mutual_info", ascending=False).reset_index(drop=True)


def plot_label_distribution(label_distribution: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=label_distribution,
        x="count",
        y="label",
        hue="label",
        dodge=False,
        legend=False,
        palette="viridis",
    )
    ax.set_title("Distribucion de clases del dataset")
    ax.set_xlabel("Cantidad de registros (escala log)")
    ax.set_ylabel("Clase")
    ax.set_xscale("log")
    plt.tight_layout()
    plt.show()


def plot_file_label_heatmap(file_label_matrix: pd.DataFrame) -> None:
    plt.figure(figsize=(14, 8))
    ax = sns.heatmap(np.log1p(file_label_matrix), cmap="YlOrRd", linewidths=0.4)
    ax.set_title("Presencia de clases por archivo (log(1 + conteo))")
    ax.set_xlabel("Clase")
    ax.set_ylabel("Archivo")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_quality(quality_df: pd.DataFrame, duplicates_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if quality_df.empty:
        axes[0].text(0.5, 0.5, "No se encontraron NaN ni infinitos", ha="center", va="center", fontsize=12)
        axes[0].axis("off")
    else:
        melted = quality_df.melt(id_vars="feature", value_vars=["missing_values", "infinite_values"], var_name="issue", value_name="count")
        melted = melted[melted["count"] > 0]
        sns.barplot(data=melted, x="count", y="feature", hue="issue", ax=axes[0], palette="magma")
        axes[0].set_title("Problemas de calidad por variable")
        axes[0].set_xlabel("Conteo")
        axes[0].set_ylabel("Variable")

    sns.barplot(
        data=duplicates_df,
        x="duplicate_rows",
        y="file",
        hue="file",
        dodge=False,
        legend=False,
        ax=axes[1],
        palette="crest",
    )
    axes[1].set_title("Filas duplicadas por archivo")
    axes[1].set_xlabel("Cantidad de filas duplicadas")
    axes[1].set_ylabel("Archivo")

    plt.tight_layout()
    plt.show()


def plot_feature_importance(mi_df: pd.DataFrame, top_n: int = 12) -> None:
    data = mi_df.head(top_n).sort_values("mutual_info", ascending=True)
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=data,
        x="mutual_info",
        y="feature",
        hue="feature",
        dodge=False,
        legend=False,
        palette="flare",
    )
    ax.set_title(f"Top {top_n} variables mas informativas segun mutual information")
    ax.set_xlabel("Mutual information")
    ax.set_ylabel("Variable")
    plt.tight_layout()
    plt.show()


def plot_key_feature_distributions(
    sample_df: pd.DataFrame,
    label_distribution: pd.DataFrame,
    features: list[str],
) -> None:
    attack_labels = label_distribution.loc[label_distribution["label"] != "BENIGN", "label"].head(5).tolist()
    label_order = ["BENIGN"] + attack_labels
    plot_df = sample_df[sample_df["Label"].isin(label_order)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()

    for axis, feature in zip(axes, features):
        values = plot_df[feature].replace([np.inf, -np.inf], np.nan)
        transformed = np.log1p(values.clip(lower=0))
        local_df = pd.DataFrame({"Label": plot_df["Label"], feature: transformed})
        sns.boxplot(
            data=local_df,
            x="Label",
            y=feature,
            order=label_order,
            hue="Label",
            dodge=False,
            legend=False,
            showfliers=False,
            ax=axis,
            palette="Set2",
        )
        axis.set_title(f"{feature} (log1p)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=30)

    for axis in axes[len(features):]:
        axis.axis("off")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(sample_df: pd.DataFrame, features: list[str]) -> None:
    corr = sample_df[features].replace([np.inf, -np.inf], np.nan).fillna(sample_df[features].median(numeric_only=True)).corr()
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.4)
    ax.set_title("Correlacion entre variables clave")
    plt.tight_layout()
    plt.show()


def build_findings_markdown(
    overview_df: pd.DataFrame,
    file_summary_df: pd.DataFrame,
    label_distribution_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    duplicates_df: pd.DataFrame,
    constant_features: list[str],
    redundant_df: pd.DataFrame,
    key_feature_df: pd.DataFrame,
) -> str:
    total_files = int(overview_df.loc[overview_df["metric"] == "Archivos CSV", "value"].iloc[0])
    total_rows = int(overview_df.loc[overview_df["metric"] == "Registros totales", "value"].iloc[0])
    total_features = int(overview_df.loc[overview_df["metric"] == "Caracteristicas numericas y de red", "value"].iloc[0])
    total_classes = int(overview_df.loc[overview_df["metric"] == "Clases de trafico", "value"].iloc[0])

    top_labels = label_distribution_df.head(4)
    top_labels_text = ", ".join(
        f"{row.label} ({row.pct:.2f}%)" for row in top_labels.itertuples(index=False)
    )

    most_attack_heavy = file_summary_df.sort_values("attack_pct", ascending=False).iloc[0]
    most_diverse = file_summary_df.sort_values("classes", ascending=False).iloc[0]
    duplicate_total = int(duplicates_df["duplicate_rows"].sum())
    quality_features = ", ".join(quality_df["feature"].tolist()) if not quality_df.empty else "ninguna"
    key_categories = ", ".join(key_feature_df["category"].drop_duplicates().tolist())

    findings = [
        "### Hallazgos principales",
        f"- El conjunto contiene **{total_rows:,} registros**, **{total_features} variables** y **{total_classes} clases** distribuidas en **{total_files} archivos** diarios.",
        f"- Existe un **desbalance fuerte de clases**: las mas frecuentes son {top_labels_text}. Esto implica que metricas de accuracy por si solas serian enganiosas en etapas de modelado.",
        f"- La estructura temporal del dataset es muy util para el contexto: **{most_attack_heavy.file}** es el archivo con mayor proporcion de ataques ({most_attack_heavy.attack_pct:.2f}%), mientras que **{most_diverse.file}** concentra la mayor diversidad de clases ({most_diverse.classes} clases).",
        f"- Los principales problemas de calidad estan concentrados en **{quality_features}**, con infinitos generados por divisiones sobre tasas, ademas de **{duplicate_total:,} filas duplicadas** dentro de archivos.",
        f"- Se detectaron **{len(constant_features)} columnas constantes** y **{len(redundant_df)} pares de columnas redundantes exactas**, por lo que conviene excluirlas antes de entrenar modelos.",
        f"- Las variables con mas señal combinan **{key_categories}**. En especial destacan medidas de duracion, tasas por segundo, tamanos de paquete y longitud de cabeceras en direccion forward.",
        "- Para la narrativa del informe, una buena idea es explicar las variables clave por familias: **volumen**, **temporalidad**, **tamano de paquetes**, **simetria del flujo** y **banderas TCP**.",
    ]
    return "\n".join(findings)


def run_eda(dataset_dir: str | Path = "dataset") -> dict[str, object]:
    _set_plot_style()
    dataset_path = Path(dataset_dir)

    print("Construyendo vista general del dataset...")
    overview_df, file_summary_df, label_distribution_df, file_label_matrix = build_overview(dataset_path)

    print("Evaluando calidad de datos...")
    quality_df, duplicates_df, constant_features, redundant_df = build_quality_report(dataset_path)

    print("Generando muestra balanceada para visualizacion y ranking de variables...")
    sample_df = build_balanced_sample(dataset_path)
    mi_df, sample_constant_features, sample_duplicate_features = rank_features(sample_df)
    key_feature_df = build_feature_dictionary(sample_df, mi_df, top_n=12)

    _markdown("## 1. Vista general del dataset")
    display(overview_df)
    display(file_summary_df)
    display(label_distribution_df)
    plot_label_distribution(label_distribution_df)
    plot_file_label_heatmap(file_label_matrix)

    _markdown("## 2. Calidad de datos y variables redundantes")
    display(quality_df if not quality_df.empty else pd.DataFrame({"status": ["Sin valores faltantes ni infinitos"]}))
    display(duplicates_df)
    display(pd.DataFrame({"constant_feature": constant_features or ["Ninguna"]}))
    display(redundant_df if not redundant_df.empty else pd.DataFrame({"status": ["Sin pares redundantes exactos"]}))
    plot_quality(quality_df, duplicates_df)

    _markdown("## 3. Caracteristicas clave del trafico")
    display(key_feature_df)
    plot_feature_importance(mi_df, top_n=12)
    plot_key_feature_distributions(
        sample_df,
        label_distribution_df,
        features=["Flow Duration", "Flow Packets/s", "Average Packet Size", "Destination Port"],
    )
    plot_correlation_heatmap(sample_df, key_feature_df["feature"].head(8).tolist())

    _markdown(
        build_findings_markdown(
            overview_df=overview_df,
            file_summary_df=file_summary_df,
            label_distribution_df=label_distribution_df,
            quality_df=quality_df,
            duplicates_df=duplicates_df,
            constant_features=constant_features,
            redundant_df=redundant_df,
            key_feature_df=key_feature_df,
        )
    )

    return {
        "overview": overview_df,
        "file_summary": file_summary_df,
        "label_distribution": label_distribution_df,
        "file_label_matrix": file_label_matrix,
        "quality": quality_df,
        "duplicates": duplicates_df,
        "constant_features": constant_features,
        "redundant_features": redundant_df,
        "sample": sample_df,
        "mutual_information": mi_df,
        "key_features": key_feature_df,
    }
