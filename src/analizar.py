"""
Análisis del caso Almacenes del Valle.

Produce figuras, una muestra de datos y un resumen numérico (resultados.json)
que usan el informe, el notebook y la sustentación. El notebook
analisis_agotados.ipynb repite este mismo razonamiento con comentarios
para la entrega en GitHub.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datos" / "agotados_semanal.csv"
FIGURAS = [
    ROOT / "Taller 1" / "figuras",
    ROOT / "Taller 2" / "figuras",
]
OUT = ROOT / "datos" / "resultados.json"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
    }
)


def cargar_y_limpiar(path: Path) -> tuple[pd.DataFrame, dict]:
    crudo = pd.read_csv(path)
    calidad = {
        "filas_crudas": int(len(crudo)),
        "columnas": int(crudo.shape[1]),
        "precios_nulos": int(crudo["precio_unitario"].isna().sum()),
        "duplicados": int(crudo.duplicated().sum()),
    }

    df = crudo.drop_duplicates().copy()
    # Precio faltante: se imputa con la mediana del mismo SKU (error de captura).
    df["precio_unitario"] = df.groupby("sku")["precio_unitario"].transform(
        lambda s: s.fillna(s.median())
    )
    # Inventario 10 veces mayor que la demanda estimada: error de digito. Se corrige.
    mascara_outlier = df["inventario_inicial"] > df["demanda_estimada"] * 4
    calidad["outliers_inventario"] = int(mascara_outlier.sum())
    df.loc[mascara_outlier, "inventario_inicial"] = (
        df.loc[mascara_outlier, "inventario_inicial"] / 10
    ).round().astype(int)

    df["semana"] = pd.to_datetime(df["semana"])
    df["cobertura"] = df["inventario_inicial"] / df["demanda_estimada"].clip(lower=1)
    df["tasa_faltante"] = df["unidades_faltantes"] / df["demanda_estimada"].clip(lower=1)

    # Lo que el planner sí conoce el lunes: venta de la semana anterior del mismo SKU-tienda.
    df = df.sort_values(["tienda_id", "sku", "semana"])
    df["venta_semana_anterior"] = df.groupby(["tienda_id", "sku"])["unidades_vendidas"].shift(1)
    df["agotado_semana_anterior"] = df.groupby(["tienda_id", "sku"])["agotado"].shift(1)
    df["cobertura_vs_semana_anterior"] = df["inventario_inicial"] / df["venta_semana_anterior"].clip(lower=1)

    calidad["filas_limpias"] = int(len(df))
    return df, calidad


def _tasa(grupo: pd.DataFrame) -> float:
    return float(grupo["agotado"].mean())


def resumen_negocio(df: pd.DataFrame) -> dict:
    semanas = df["semana"].nunique()
    venta = float(df["venta_cop"].sum())
    perdida = float(df["venta_perdida_cop"].sum())
    por_mes = perdida / (semanas / 4.345)  # promedio mensual aproximado

    por_categoria = (
        df.groupby("categoria")
        .agg(
            tasa_agotado=("agotado", "mean"),
            tasa_critica=("agotado_critico", "mean"),
            venta_perdida=("venta_perdida_cop", "sum"),
            dias_promedio=("dias_agotado", "mean"),
        )
        .sort_values("venta_perdida", ascending=False)
    )
    por_tienda = (
        df.groupby(["tienda", "ciudad"])
        .agg(
            tasa_agotado=("agotado", "mean"),
            venta_perdida=("venta_perdida_cop", "sum"),
        )
        .sort_values("tasa_agotado", ascending=False)
    )
    promo = df.groupby("en_promocion")["agotado"].mean()
    festivo = df.groupby("festivo")["agotado"].mean()
    retraso = df.groupby("retraso_proveedor")["agotado"].mean()

    return {
        "filas": int(len(df)),
        "semanas": int(semanas),
        "tiendas": int(df["tienda_id"].nunique()),
        "skus": int(df["sku"].nunique()),
        "desde": df["semana"].min().date().isoformat(),
        "hasta": df["semana"].max().date().isoformat(),
        "tasa_agotado": float(df["agotado"].mean()),
        "tasa_critica": float(df["agotado_critico"].mean()),
        "dias_agotado_promedio": float(df["dias_agotado"].mean()),
        "venta_cop": venta,
        "venta_perdida_cop": perdida,
        "venta_perdida_mensual_cop": float(por_mes),
        "participacion_perdida": float(perdida / (venta + perdida)),
        "tasa_con_promo": float(promo.get(1, np.nan)),
        "tasa_sin_promo": float(promo.get(0, np.nan)),
        "tasa_festivo": float(festivo.get(1, np.nan)),
        "tasa_sin_festivo": float(festivo.get(0, np.nan)),
        "tasa_con_retraso": float(retraso.get(1, np.nan)),
        "tasa_sin_retraso": float(retraso.get(0, np.nan)),
        "por_categoria": por_categoria.reset_index().to_dict(orient="records"),
        "por_tienda": [
            {
                "tienda": t,
                "ciudad": c,
                "tasa_agotado": float(r.tasa_agotado),
                "venta_perdida": float(r.venta_perdida),
            }
            for (t, c), r in por_tienda.iterrows()
        ],
    }


def bivariate(df: pd.DataFrame) -> dict:
    cols = [
        "cobertura",
        "dias_agotado",
        "en_promocion",
        "descuento_pct",
        "lead_time_dias",
        "retraso_proveedor",
        "festivo",
        "venta_perdida_cop",
        "unidades_faltantes",
    ]
    corr = df[cols].corr(numeric_only=True)
    pares = {
        "cobertura_vs_dias_agotado": float(corr.loc["cobertura", "dias_agotado"]),
        "promo_vs_dias_agotado": float(corr.loc["en_promocion", "dias_agotado"]),
        "lead_time_vs_dias_agotado": float(corr.loc["lead_time_dias", "dias_agotado"]),
        "festivo_vs_dias_agotado": float(corr.loc["festivo", "dias_agotado"]),
        "retraso_vs_dias_agotado": float(corr.loc["retraso_proveedor", "dias_agotado"]),
        "descuento_vs_dias_agotado": float(corr.loc["descuento_pct", "dias_agotado"]),
        "cobertura_vs_venta_perdida": float(corr.loc["cobertura", "venta_perdida_cop"]),
    }

    # Cobertura en quintiles para mostrar la relación sin depender solo del coeficiente.
    df = df.copy()
    df["quintil_cobertura"] = pd.qcut(
        df["cobertura"], 5, labels=["Q1 más baja", "Q2", "Q3", "Q4", "Q5 más alta"]
    )
    por_q = (
        df.groupby("quintil_cobertura", observed=True)
        .agg(tasa_agotado=("agotado", "mean"), dias=("dias_agotado", "mean"), n=("agotado", "size"))
        .reset_index()
    )

    promo_cat = (
        df.groupby(["categoria", "en_promocion"])["agotado"].mean().unstack()
    )
    promo_cat = promo_cat.rename(columns={0: "sin_promo", 1: "con_promo"})

    return {
        "correlaciones": pares,
        "matriz": corr.round(3).to_dict(),
        "por_quintil_cobertura": [
            {
                "quintil": str(r.quintil_cobertura),
                "tasa_agotado": float(r.tasa_agotado),
                "dias_promedio": float(r.dias),
                "n": int(r.n),
            }
            for r in por_q.itertuples(index=False)
        ],
        "promo_por_categoria": promo_cat.reset_index().to_dict(orient="records"),
    }


def _guardar(fig: plt.Figure, nombre: str) -> None:
    fig.tight_layout()
    for carpeta in FIGURAS:
        carpeta.mkdir(parents=True, exist_ok=True)
        fig.savefig(carpeta / nombre, dpi=140, bbox_inches="tight")
    plt.close(fig)


def graficos(df: pd.DataFrame, bi: dict) -> None:
    # 1. Tasa de agotado por categoría
    cat = (
        df.groupby("categoria")["agotado"].mean().sort_values(ascending=False) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cat.plot(kind="bar", color="#1f4e79", ax=ax)
    ax.set_title("Tasa de agotado por categoría")
    ax.set_ylabel("Semanas con al menos 1 día de agotado (%)")
    ax.set_xlabel("Categoría")
    ax.set_ylim(0, max(cat.max() * 1.15, 10))
    ax.tick_params(axis="x", rotation=0)
    _guardar(fig, "01_tasa_agotado_por_categoria.png")

    # 2. Tasa por tienda
    tienda = (
        df.groupby("tienda")["agotado"].mean().sort_values(ascending=True) * 100
    )
    fig, ax = plt.subplots(figsize=(8, 5.2))
    tienda.plot(kind="barh", color="#b85c38", ax=ax)
    ax.set_title("Tasa de agotado por tienda")
    ax.set_xlabel("Semanas con al menos 1 día de agotado (%)")
    ax.set_ylabel("")
    _guardar(fig, "02_tasa_agotado_por_tienda.png")

    # 3. Dispersión cobertura vs días de agotado (muestra para que se lea)
    muestra = df.sample(n=2500, random_state=7)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        muestra["cobertura"],
        muestra["dias_agotado"] + np.random.default_rng(1).uniform(-0.15, 0.15, len(muestra)),
        alpha=0.25,
        s=12,
        c="#1f4e79",
        edgecolors="none",
    )
    ax.set_title("Cobertura de inventario vs días de agotado")
    ax.set_xlabel("Cobertura (inventario inicial / demanda estimada)")
    ax.set_ylabel("Días de agotado en la semana")
    ax.set_xlim(0, min(3.2, muestra["cobertura"].quantile(0.995)))
    _guardar(fig, "03_dispersion_cobertura_dias.png")

    # 4. Tasa por quintil de cobertura
    q = pd.DataFrame(bi["por_quintil_cobertura"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(q["quintil"], q["tasa_agotado"] * 100, color="#1f4e79")
    ax.set_title("Tasa de agotado según quintil de cobertura")
    ax.set_ylabel("Tasa de agotado (%)")
    ax.set_xlabel("Cobertura de inventario (Q1 = más baja)")
    _guardar(fig, "04_tasa_por_quintil_cobertura.png")

    # 5. Promoción vs no promoción por categoría
    promo = (
        df.groupby(["categoria", "en_promocion"])["agotado"].mean().unstack() * 100
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    promo.plot(kind="bar", ax=ax, color=["#7d8b99", "#b85c38"])
    ax.set_title("Tasa de agotado con y sin promoción, por categoría")
    ax.set_ylabel("Tasa de agotado (%)")
    ax.set_xlabel("")
    ax.legend(["Sin promoción", "Con promoción"], frameon=False)
    ax.tick_params(axis="x", rotation=0)
    _guardar(fig, "05_promo_vs_agotado_por_categoria.png")

    # 6. Venta perdida por categoría
    perdida = (
        df.groupby("categoria")["venta_perdida_cop"].sum().sort_values(ascending=False) / 1e6
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    perdida.plot(kind="bar", color="#1f4e79", ax=ax)
    ax.set_title("Venta perdida estimada por categoría")
    ax.set_ylabel("Millones de COP")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    _guardar(fig, "06_venta_perdida_por_categoria.png")

    # 7. Serie mensual de tasa de agotado
    mensual = df.set_index("semana").resample("ME")["agotado"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.plot(mensual.index, mensual.values, marker="o", color="#1f4e79")
    ax.set_title("Tasa de agotado mensual")
    ax.set_ylabel("Tasa de agotado (%)")
    ax.set_xlabel("Mes")
    _guardar(fig, "07_tasa_agotado_mensual.png")

    # 8. Mapa de correlaciones
    cols = [
        "cobertura",
        "dias_agotado",
        "en_promocion",
        "lead_time_dias",
        "festivo",
        "descuento_pct",
        "venta_perdida_cop",
    ]
    etiquetas = [
        "Cobertura",
        "Días agotado",
        "Promoción",
        "Lead time",
        "Festivo",
        "Descuento %",
        "Venta perdida",
    ]
    matriz = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(matriz.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(etiquetas)))
    ax.set_yticks(range(len(etiquetas)))
    ax.set_xticklabels(etiquetas, rotation=40, ha="right")
    ax.set_yticklabels(etiquetas)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{matriz.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Correlación entre variables operativas")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _guardar(fig, "08_correlaciones.png")


def modelo(df: pd.DataFrame) -> dict:
    """
    Predice si la semana tendrá al menos un día de agotado, con datos disponibles el lunes,
    antes de que ocurra la semana. No usa demanda, faltantes ni días de agotado
    de la misma semana: eso sería fuga de información.
    """
    trabajo = df.dropna(
        subset=["venta_semana_anterior", "agotado_semana_anterior", "cobertura_vs_semana_anterior"]
    ).copy()
    trabajo["cobertura_vs_semana_anterior"] = trabajo["cobertura_vs_semana_anterior"].clip(0, 8)

    corte = pd.Timestamp("2026-05-04")
    train = trabajo[trabajo["semana"] < corte]
    test = trabajo[trabajo["semana"] >= corte]

    features_num = [
        "cobertura_vs_semana_anterior",
        "en_promocion",
        "lead_time_dias",
        "festivo",
        "mes",
        "agotado_semana_anterior",
    ]
    features_cat = ["categoria", "ciudad"]
    X_train = train[features_num + features_cat]
    y_train = train["agotado"]
    X_test = test[features_num + features_cat]
    y_test = test["agotado"]

    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), features_cat),
        ]
    )

    log_reg = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                LogisticRegression(max_iter=400, class_weight="balanced", random_state=42),
            ),
        ]
    )
    bosque = Pipeline(
        [
            ("pre", pre),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_leaf=20,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    log_reg.fit(X_train, y_train)
    bosque.fit(X_train, y_train)

    proba_log = log_reg.predict_proba(X_test)[:, 1]
    alerta = proba_log >= 0.50
    perdida_test = float(test.loc[test["agotado"] == 1, "venta_perdida_cop"].sum())
    perdida_detectada = float(test.loc[(test["agotado"] == 1) & alerta, "venta_perdida_cop"].sum())

    def metricas(nombre, pipe, umbral=0.50):
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        return {
            "modelo": nombre,
            "umbral": umbral,
            "auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "prevalencia_test": float(y_test.mean()),
        }

    # Umbral 0.45 en el bosque: prioriza detectar el agotado.
    # Una alerta de más cuesta revisar un pedido; un agotado no detectado cuesta la venta.
    m_log = metricas("Regresión logística", log_reg, 0.50)
    m_rf = metricas("Bosque aleatorio", bosque, 0.45)
    regla = test["en_promocion"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, regla).ravel()
    m_regla = {
        "modelo": "Regla: alertar toda promoción",
        "umbral": None,
        "auc": None,
        "accuracy": float(accuracy_score(y_test, regla)),
        "precision": float(precision_score(y_test, regla, zero_division=0)),
        "recall": float(recall_score(y_test, regla, zero_division=0)),
        "f1": float(f1_score(y_test, regla, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "prevalencia_test": float(y_test.mean()),
    }

    # Importancia del bosque (promedio de impureza), agregada al nombre original.
    rf = bosque.named_steps["clf"]
    nombres = bosque.named_steps["pre"].get_feature_names_out()
    importancias = (
        pd.Series(rf.feature_importances_, index=nombres).sort_values(ascending=False).head(8)
    )
    # Coeficientes de la logística para explicar dirección, no solo importancia.
    lr = log_reg.named_steps["clf"]
    nombres_lr = log_reg.named_steps["pre"].get_feature_names_out()
    coefs = pd.Series(lr.coef_[0], index=nombres_lr).sort_values()

    top_pos = coefs.tail(5)
    top_neg = coefs.head(5)

    return {
        "corte_entrenamiento": "semanas anteriores al 2026-05-04",
        "corte_prueba": "2026-05-04 a 2026-06-29",
        "filas_train": int(len(train)),
        "filas_test": int(len(test)),
        "variables": features_num + features_cat,
        "objetivo": "agotado (al menos 1 día sin inventario vendible en la semana)",
        "logistica": m_log,
        "bosque": m_rf,
        "regla_promocion": m_regla,
        "venta_perdida_test_cop": perdida_test,
        "venta_perdida_detectada_cop": perdida_detectada,
        "participacion_perdida_detectada": float(perdida_detectada / perdida_test) if perdida_test else None,
        "importancias_bosque": [
            {"variable": k, "importancia": float(v)} for k, v in importancias.items()
        ],
        "coef_suben_riesgo": [
            {"variable": k, "coeficiente": float(v)} for k, v in top_pos.items()
        ],
        "coef_bajan_riesgo": [
            {"variable": k, "coeficiente": float(v)} for k, v in top_neg.items()
        ],
    }


def main() -> None:
    df, calidad = cargar_y_limpiar(DATA)
    negocio = resumen_negocio(df)
    bi = bivariate(df)
    graficos(df, bi)
    mod = modelo(df)

    muestra = (
        df[
            [
                "semana",
                "tienda",
                "producto",
                "categoria",
                "en_promocion",
                "inventario_inicial",
                "demanda_estimada",
                "unidades_vendidas",
                "dias_agotado",
                "venta_perdida_cop",
            ]
        ]
        .head(12)
        .copy()
    )
    muestra["semana"] = muestra["semana"].dt.date.astype(str)
    muestra.to_csv(ROOT / "datos" / "muestra_datos.csv", index=False)
    destino_muestra = ROOT / "Taller 2" / "datos"
    destino_muestra.mkdir(parents=True, exist_ok=True)
    muestra.to_csv(destino_muestra / "muestra_datos.csv", index=False)

    payload = {
        "calidad": calidad,
        "negocio": negocio,
        "bivariado": {
            "correlaciones": bi["correlaciones"],
            "por_quintil_cobertura": bi["por_quintil_cobertura"],
            "promo_por_categoria": bi["promo_por_categoria"],
        },
        "modelo": mod,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"tasa_agotado": negocio["tasa_agotado"], "auc": mod["bosque"]["auc"]}, indent=2))
    print(f"Resumen: {OUT}")


if __name__ == "__main__":
    main()
