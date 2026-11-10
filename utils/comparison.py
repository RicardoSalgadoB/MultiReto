"""
Comparación de modelos con los DATOS DE PRUEBA.

Los modelos se ajustan con entrenamiento y se comparan por su desempeño fuera de muestra:

* Logística y probabilidades P(X > c): log-verosimilitud, Brier, BSS y AUC en prueba.
* GPD: -log verosimilitud de los excesos de prueba (filas dentro del soporte de todos
  los modelos) y cuántos excesos caen fuera del soporte.

Para cada par (A, B) se calcula 2 * (logL_B - logL_A) en prueba (> 0 favorece a B) y un
p-valor de una prueba pareada de las diferencias por observación, con errores estándar
agrupados por día objetivo (las horas de un día y las estaciones de un mismo día están
correlacionadas). Equivale a una prueba de Diebold-Mariano y, a diferencia de la
chi-cuadrada de la prueba LR clásica, es válida fuera de muestra y para modelos no anidados.
La fila "ALL" junta todas las estaciones.
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .data_preparation import (STANDARD_C, THRESHOLD_U, add_climatology_reference,
                               build_all_forecast_tables, complete_cases)
from .gpd_utils import fit_two_part_model, gpd_aic, gpd_nll_per_observation
from .validation import probability_metrics


# ----------------------------------------------------------------------------
# Prueba pareada con errores agrupados
# ----------------------------------------------------------------------------
def clustered_paired_test(diff, clusters):
    """
    H0: la media de `diff` es 0. Errores estándar agrupados por `clusters`.
    Regresa (media, error estándar, p-valor bilateral t con G - 1 gl).
    """
    diff = np.asarray(diff, dtype=float)
    ids = pd.factorize(np.asarray(clusters))[0]
    n, groups = len(diff), ids.max() + 1 if len(diff) else 0
    if n == 0 or groups < 2:
        return np.nan, np.nan, np.nan
    mean = diff.mean()
    sums = np.bincount(ids, weights=diff)
    counts = np.bincount(ids)
    variance = groups / (groups - 1) * np.sum((sums - mean * counts) ** 2) / n ** 2
    se = np.sqrt(variance)
    if se == 0:
        return mean, 0.0, 1.0 if mean == 0 else 0.0
    return mean, se, 2 * student_t.sf(abs(mean / se), groups - 1)


def _log_score(y, p):
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def _stack(tables, columns, target, period):
    parts = []
    for station, table in tables.items():
        t = table[table["period"] == period][[target, "target_date", *columns]].copy()
        t["station"] = station
        parts.append(t)
    return pd.concat(parts)


def _groups(data, pooled):
    groups = list(data.groupby("station", sort=False))
    return groups + ([("ALL", data)] if pooled else [])


# ----------------------------------------------------------------------------
# Modelos de probabilidad (logística, climatología, dos partes, ...)
# ----------------------------------------------------------------------------
def score_probability_columns(tables, columns, target="event", ref_col="ref_event",
                              period="test", pooled=True):
    """
    Métricas en `period` de cada columna de probabilidad (una columna = un modelo).
    Incluye BSS contra la referencia `ref_col` (climatología).
    """
    data = _stack(tables, list(columns) + [ref_col], target, period)
    rows = []
    for station, g in _groups(data, pooled):
        for col in columns:
            rows.append({"station": station, "model": col,
                         **probability_metrics(g[target], g[col], g[ref_col])})
    return pd.DataFrame(rows)


def compare_probability_columns(tables, pairs, target="event", period="test", pooled=True):
    """
    pairs: lista de (columna_A, columna_B). Reporta 2 * suma(logL_B - logL_A), el p-valor
    de la prueba pareada y Brier / AUC de cada modelo.
    """
    columns = sorted({c for pair in pairs for c in pair})
    data = _stack(tables, columns, target, period)
    rows = []
    for station, g in _groups(data, pooled):
        for a, b in pairs:
            diff = _log_score(g[target], g[b]) - _log_score(g[target], g[a])
            mean, _, p = clustered_paired_test(diff, g["target_date"])
            ma = probability_metrics(g[target], g[a])
            mb = probability_metrics(g[target], g[b])
            rows.append({"station": station, "comparison": f"{a} vs {b}",
                         "test_LR_stat": 2 * diff.sum(), "mean_logL_gain": mean, "p_value": p,
                         "brier_A": ma["brier"], "brier_B": mb["brier"],
                         "auc_A": ma["auc"], "auc_B": mb["auc"],
                         "better": b if mean > 0 else a})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# GPD
# ----------------------------------------------------------------------------
def summarize_gpd_families(models_by_family, data_by_station, pooled=True):
    """
    Por familia y estación: -log verosimilitud de los excesos de prueba en las filas
    dentro del soporte de TODAS las familias (comparables), excesos fuera del soporte
    por familia y AIC de entrenamiento (solo referencia).
    """
    families = list(models_by_family)
    rows = []
    for station, data in data_by_station.items():
        nll = {f: gpd_nll_per_observation(models_by_family[f][station], data) for f in families}
        common = np.all([np.isfinite(v) for v in nll.values()], axis=0)
        for f in families:
            rows.append({"station": station, "model": f, "n_test": len(data),
                         "n_common": int(common.sum()), "test_nll": nll[f][common].sum(),
                         "outside_support": int((~np.isfinite(nll[f])).sum()),
                         "aic_train": gpd_aic(models_by_family[f][station])})
    table = pd.DataFrame(rows)
    if pooled:
        total = table.groupby("model", sort=False).agg(
            n_test=("n_test", "sum"), n_common=("n_common", "sum"), test_nll=("test_nll", "sum"),
            outside_support=("outside_support", "sum"), aic_train=("aic_train", "sum")).reset_index()
        total.insert(0, "station", "ALL")
        table = pd.concat([table, total], ignore_index=True)
    table["mean_test_nll"] = table["test_nll"] / table["n_common"]
    return table


def compare_gpd_families(models_by_family, data_by_station, pairs, pooled=True):
    """
    pairs: lista de (familia_A, familia_B). Usa los excesos de prueba dentro del soporte
    de ambos modelos. 2 * suma(nll_A - nll_B) > 0 favorece a B.
    """
    per_station = {}
    for station, data in data_by_station.items():
        for a, b in pairs:
            nll_a = gpd_nll_per_observation(models_by_family[a][station], data)
            nll_b = gpd_nll_per_observation(models_by_family[b][station], data)
            per_station[(station, a, b)] = (nll_a, nll_b, data["target_date"].to_numpy())

    rows = []
    for a, b in pairs:
        stacks = {"na": [], "nb": [], "cl": []}
        for station in data_by_station:
            nll_a, nll_b, clusters = per_station[(station, a, b)]
            ok = np.isfinite(nll_a) & np.isfinite(nll_b)
            diff = nll_a[ok] - nll_b[ok]
            mean, _, p = clustered_paired_test(diff, clusters[ok])
            rows.append({"station": station, "comparison": f"{a} vs {b}",
                         "test_LR_stat": 2 * diff.sum(), "p_value": p, "n_common": int(ok.sum()),
                         "outside_A": int((~np.isfinite(nll_a)).sum()),
                         "outside_B": int((~np.isfinite(nll_b)).sum()),
                         "aic_train_A": gpd_aic(models_by_family[a][station]),
                         "aic_train_B": gpd_aic(models_by_family[b][station]),
                         "better": b if mean > 0 else a})
            stacks["na"].append(nll_a[ok]); stacks["nb"].append(nll_b[ok]); stacks["cl"].append(clusters[ok])
        if pooled:
            diff = np.concatenate(stacks["na"]) - np.concatenate(stacks["nb"])
            mean, _, p = clustered_paired_test(diff, np.concatenate(stacks["cl"]))
            sub = [r for r in rows if r["comparison"] == f"{a} vs {b}"]
            rows.append({"station": "ALL", "comparison": f"{a} vs {b}",
                         "test_LR_stat": 2 * diff.sum(), "p_value": p, "n_common": len(diff),
                         "outside_A": sum(r["outside_A"] for r in sub),
                         "outside_B": sum(r["outside_B"] for r in sub),
                         "aic_train_A": sum(r["aic_train_A"] for r in sub),
                         "aic_train_B": sum(r["aic_train_B"] for r in sub),
                         "better": b if mean > 0 else a})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Elección del modelo
# ----------------------------------------------------------------------------
def select_best_model(summary, metric, higher_is_better=False, scope="global"):
    """
    Elige el mejor modelo de una tabla de resumen (columnas 'station', 'model', metric).
    scope='global': un solo modelo para todas las estaciones (fila 'ALL'); es lo más estable.
    scope='station': un diccionario estación -> mejor modelo.
    """
    pick = (lambda s: s.idxmax()) if higher_is_better else (lambda s: s.idxmin())
    if scope == "global":
        g = summary[summary["station"] == "ALL"].set_index("model")[metric]
        return pick(g)
    g = summary[summary["station"] != "ALL"]
    return {station: pick(sub.set_index("model")[metric]) for station, sub in g.groupby("station")}


# ----------------------------------------------------------------------------
# Sensibilidad al horizonte de pronóstico
# ----------------------------------------------------------------------------
def horizon_sensitivity(frames, horizons, frequency, logit_features, gpd_covariates,
                        u=THRESHOLD_U, c=STANDARD_C, train_end=None, cluster_col=None,
                        climatology_key="target_hour"):
    """
    Repite el modelo de dos partes para varios horizontes y reporta en PRUEBA el AUC y el
    BSS de zeta (P(X > u)) y de P(X > c). Todos con las mismas variables y el mismo c.
    """
    kwargs = {"train_end": train_end} if train_end is not None else {}
    needed = list(logit_features) + [v for v in gpd_covariates if v != "logit_zeta"]
    rows = []
    for h in horizons:
        tables = build_all_forecast_tables(frames, h, frequency, u, c, **kwargs)
        tables = complete_cases(tables, needed)
        add_climatology_reference(tables, climatology_key)
        tables, _, _ = fit_two_part_model(tables, logit_features, gpd_covariates, u, c,
                                          cluster_col, output_column="p_two_part")
        zeta = score_probability_columns(tables, ["zeta"], "event", "ref_event", pooled=True)
        p_c = score_probability_columns(tables, ["p_two_part"], "event_c", "ref_event_c", pooled=True)
        zeta, p_c = zeta[zeta["station"] == "ALL"].iloc[0], p_c[p_c["station"] == "ALL"].iloc[0]
        rows.append({"horizon": h, "AUC zeta": zeta["auc"], "BSS zeta": zeta["bss"],
                     "AUC P(X>c)": p_c["auc"], "BSS P(X>c)": p_c["bss"]})
    return pd.DataFrame(rows).set_index("horizon")