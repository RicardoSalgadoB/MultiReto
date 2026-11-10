"""
Validación con los DATOS DE PRUEBA (modelos ajustados con entrenamiento).

1. Métricas probabilísticas y calibración.
2. Supuestos de la regresión logística: linealidad (Box-Tidwell y cuadrados),
   multicolinealidad (VIF), independencia (autocorrelación, Durbin-Watson, error estándar
   robusto) y observaciones influyentes (distancia de Cook).
   Los residuales y la distancia de Cook se calculan con el modelo de ENTRENAMIENTO sobre las
   filas de prueba; las pruebas que requieren reajustar (Box-Tidwell, cuadrados, error
   estándar robusto) se reajustan con las filas de prueba.
3. Diagnósticos de la GPD: gráficas QQ y prueba KS de los residuales en excesos de prueba.
4. Alertas: umbral para una tasa de aciertos, matriz de confusión y curva ROC.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, expon, kstest
from sklearn.metrics import (ConfusionMatrixDisplay, brier_score_loss, log_loss,
                             roc_auc_score, roc_curve)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from .gpd_utils import gpd_residuals
from .logit_utils import design_matrix_logit, fit_logit, predict_logit

CYCLIC_PREFIXES = ("hour_", "month_")


# ----------------------------------------------------------------------------
# 1. Métricas y calibración
# ----------------------------------------------------------------------------
def probability_metrics(y, p, p_ref=None):
    """
    n, eventos, log-loss, Brier, AUC y (si hay referencia) BSS = 1 - Brier / Brier_ref.
    """
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ok = ~(np.isnan(y) | np.isnan(p))
    y, p = y[ok], p[ok]
    out = {"n": int(ok.sum()), "events": int(y.sum())}
    if len(y) == 0:
        return {**out, "log_loss": np.nan, "brier": np.nan, "auc": np.nan, "bss": np.nan}
    out["log_loss"] = log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])
    out["brier"] = brier_score_loss(y, p)
    out["auc"] = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    if p_ref is not None:
        ref = np.asarray(p_ref, dtype=float)[ok]
        out["bss"] = 1 - out["brier"] / brier_score_loss(y, ref)
    return out


def pooled_values(tables, columns, period="test"):
    """Junta las filas de un periodo de todas las estaciones (columnas pedidas)."""
    return pd.concat([t.loc[t["period"] == period, list(columns)] for t in tables.values()])


def calibration_table(y, p, bins, min_count=30):
    """Probabilidad media predicha vs frecuencia observada por intervalo (n >= min_count)."""
    t = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)}).dropna()
    groups = t.groupby(pd.cut(t["p"], bins, include_lowest=True), observed=True)
    table = groups.agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))
    return table[table["n"] >= min_count]


def plot_calibration(series, bins, title=None, max_value=None, path=None):
    """
    series: dict etiqueta -> (y, p). Dibuja la calibración de cada uno.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    markers = ["o-", "s--", "^:", "d-."]
    top = 0
    for (label, (y, p)), marker in zip(series.items(), markers * 3):
        cal = calibration_table(y, p, bins)
        ax.plot(cal["pred"], cal["obs"], marker, label=label)
        if len(cal):
            top = max(top, cal["pred"].max(), cal["obs"].max())
    top = max_value or top * 1.1 or 1
    ax.plot([0, top], [0, top], color="gray", lw=1)
    ax.set_xlabel("Probabilidad predicha")
    ax.set_ylabel("Frecuencia observada")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


# ----------------------------------------------------------------------------
# 2. Supuestos de la regresión logística
# ----------------------------------------------------------------------------
def _continuous_features(data, features):
    """Variables continuas: ni binarias ni cíclicas (calendario)."""
    return [f for f in features
            if data[f].nunique() > 5 and not f.startswith(CYCLIC_PREFIXES)]


def vif_table(model, data):
    """VIF de cada variable (la constante se incluye en la matriz pero no se reporta)."""
    X = design_matrix_logit(model, data.dropna(subset=model["features"]))
    values = [variance_inflation_factor(X.to_numpy(float), i) for i in range(1, X.shape[1])]
    return pd.Series(values, index=X.columns[1:], name="VIF")


def box_tidwell_test(data, features, target="event"):
    """
    Prueba de Box-Tidwell: se agregan términos x*ln(x) (con x desplazada para ser positiva)
    a la logística; un p-valor < 0.05 sugiere que la relación no es lineal en el logit.
    Solo se prueban variables continuas.
    """
    data = data.dropna(subset=list(features) + [target])
    features = [f for f in features if data[f].std() > 1e-9]
    continuous = _continuous_features(data, features)
    z = (data[features] - data[features].mean()) / data[features].std()
    terms = {}
    for f in continuous:
        shifted = data[f] - data[f].min() + 1.0
        t = shifted * np.log(shifted)
        terms[f"{f}_x_ln"] = (t - t.mean()) / t.std()
    X = sm.add_constant(pd.concat([z, pd.DataFrame(terms, index=data.index)], axis=1))
    try:
        res = sm.Logit(data[target], X).fit(disp=0, maxiter=200)
        p = res.pvalues[list(terms)].to_numpy()
    except Exception:
        p = np.full(len(continuous), np.nan)
    return pd.Series(p, index=continuous, name="box_tidwell_p")


def quadratic_linearity_test(data, features, target="event", cluster_col=None):
    """
    Compara la logística lineal contra una con los cuadrados de las variables continuas
    (prueba de razón de verosimilitudes). Regresa D, gl, p y las variables cuyo cuadrado
    es significativo.
    """
    data = data.dropna(subset=list(features) + [target]).copy()
    features = [f for f in features if data[f].std() > 1e-9]
    continuous = _continuous_features(data, features)
    squares = []
    for f in continuous:
        z = (data[f] - data[f].mean()) / data[f].std()
        data[f + "^2"] = z ** 2
        squares.append(f + "^2")
    try:
        linear = fit_logit(data, features, target, cluster_col)["result"]
        quad = fit_logit(data, features + squares, target, cluster_col)["result"]
    except Exception:
        return {"D": np.nan, "df": np.nan, "p": np.nan, "significant_squares": []}
    D = 2 * (quad.llf - linear.llf)
    df = quad.df_model - linear.df_model
    significant = [s[:-2] for s in squares if s in quad.pvalues and quad.pvalues[s] < 0.05]
    return {"D": D, "df": df, "p": chi2.sf(D, df), "significant_squares": significant}


def pearson_residuals(model, data):
    p = predict_logit(model, data).clip(1e-9, 1 - 1e-9)
    return (data[model["target"]] - p) / np.sqrt(p * (1 - p))


def residual_independence(model, data, freq="h", seasonal_lag=24):
    """
    Durbin-Watson y autocorrelación de los residuales de Pearson en una malla regular
    (lag 1 y lag estacional: 24 h para series horarias, 7 días para diarias).
    """
    r = pearson_residuals(model, data.dropna(subset=model["features"]))
    grid = r.reindex(pd.date_range(r.index.min(), r.index.max(), freq=freq))
    return {"durbin_watson": durbin_watson(r.to_numpy()),
            "autocorr_lag1": grid.autocorr(1),
            "autocorr_seasonal": grid.autocorr(seasonal_lag)}


def robust_se_ratio(data, features, target="event", cluster_col="target_date"):
    """Mediana de (EE robusto por clúster) / (EE usual): cuánto subestima el EE usual."""
    try:
        naive = fit_logit(data, features, target)["result"]
        robust = fit_logit(data, features, target, cluster_col)["result"]
        return float((robust.bse / naive.bse).drop("const").median())
    except Exception:
        return np.nan


def cooks_distance(model, data):
    """Distancia de Cook de una logística: r_i^2 * h_i / (k (1 - h_i)^2)."""
    data = data.dropna(subset=model["features"])
    X = design_matrix_logit(model, data).to_numpy(float)
    p = predict_logit(model, data).clip(1e-9, 1 - 1e-9).to_numpy()
    r = pearson_residuals(model, data).to_numpy()
    w = p * (1 - p)
    inverse = np.linalg.pinv(X.T @ (X * w[:, None]))
    h = np.clip(w * np.sum((X @ inverse) * X, axis=1), 0, 1 - 1e-9)
    return pd.Series(r ** 2 * h / (X.shape[1] * (1 - h) ** 2), index=data.index, name="cook")


def validate_logit_station(model, test_table, freq="h", seasonal_lag=24, cluster_col="target_date"):
    """Todos los diagnósticos de una estación. Regresa (resumen escalar, detalle)."""
    target, feats = model["target"], model["features"]
    data = test_table.dropna(subset=feats + [target])

    vif = vif_table(model, data)
    bt = box_tidwell_test(data, feats, target)
    quad = quadratic_linearity_test(data, feats, target)
    independence = residual_independence(model, data, freq, seasonal_lag)
    cook = cooks_distance(model, data)
    r = pearson_residuals(model, data)
    large = r.abs() > 3

    summary = {
        "n": len(data), "events": int(data[target].sum()),
        "max_VIF": vif.max(), "max_VIF_variable": vif.idxmax(),
        "BT_nonlinear_vars": int((bt < 0.05).sum()),
        "quad_D": quad["D"], "quad_p": quad["p"], "quad_n_significant": len(quad["significant_squares"]),
        **independence,
        "robust_SE_ratio": robust_se_ratio(data, feats, target, cluster_col) if cluster_col in data else np.nan,
        "abs_resid_gt_3": int(large.sum()),
        "pct_of_those_events": 100 * data.loc[large, target].mean() if large.any() else np.nan,
        "cook_gt_4n": int((cook > 4 / len(data)).sum()),
        "cook_gt_1": int((cook > 1).sum()), "cook_max": cook.max(),
    }
    detail = {"vif": vif, "box_tidwell": bt, "quadratic": quad, "cook": cook, "residuals": r}
    return summary, detail


def validate_logit_all_stations(models, tables, test_period="test", freq="h",
                                seasonal_lag=24, cluster_col="target_date"):
    """Diagnósticos en los datos de prueba. Regresa (tabla resumen por estación, detalles)."""
    rows, details = [], {}
    for station, model in models.items():
        test = tables[station]
        test = test[test["period"] == test_period]
        summary, detail = validate_logit_station(model, test, freq, seasonal_lag, cluster_col)
        rows.append({"station": station, **summary})
        details[station] = detail
    return pd.DataFrame(rows).set_index("station"), details


def influence_sensitivity(tables, features, fraction=0.01, target="event", ref_col="ref_event",
                          cluster_col=None, train_period="train", test_period="test"):
    """
    ¿Cambia el desempeño en PRUEBA si se reajusta sin el `fraction` de filas de
    entrenamiento más influyentes (distancia de Cook)? Regresa métricas por estación y en conjunto.
    """
    rows, pooled = [], {"con todas": ([], [], []), "sin influyentes": ([], [], [])}
    for station, table in tables.items():
        train = table[table["period"] == train_period].dropna(subset=features + [target])
        test = table[table["period"] == test_period].dropna(subset=features + [target])
        full = fit_logit(train, features, target, cluster_col)
        cook = cooks_distance(full, train)
        keep = train.drop(cook.nlargest(max(1, int(fraction * len(train)))).index)
        reduced = fit_logit(keep, features, target, cluster_col)
        for label, m in [("con todas", full), ("sin influyentes", reduced)]:
            p = predict_logit(m, test)
            rows.append({"station": station, "model": label,
                         **probability_metrics(test[target], p, test[ref_col])})
            pooled[label][0].append(test[target].to_numpy())
            pooled[label][1].append(p.to_numpy())
            pooled[label][2].append(test[ref_col].to_numpy())
    for label, (ys, ps, refs) in pooled.items():
        rows.append({"station": "ALL", "model": label,
                     **probability_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(refs))})
    return pd.DataFrame(rows).set_index(["station", "model"])


# ----------------------------------------------------------------------------
# 3. Diagnósticos de la GPD (excesos de prueba)
# ----------------------------------------------------------------------------
def gpd_qq_diagnostics(models, data_by_station):
    """
    Residuales de la GPD en excesos nuevos: n, cuántos caen fuera del soporte, media
    (debe ser ~1) y p-valor de la prueba KS contra Exp(1).
    """
    rows = []
    for station, model in models.items():
        res = gpd_residuals(model, data_by_station[station])
        finite = res[np.isfinite(res)]
        rows.append({"station": station, "n": len(res), "outside_support": int(len(res) - len(finite)),
                     "mean_residual": finite.mean() if len(finite) else np.nan,
                     "ks_p": kstest(finite, "expon").pvalue if len(finite) > 5 else np.nan})
    return pd.DataFrame(rows).set_index("station")


def plot_gpd_qq(models_by_label, data_by_station, ncols=5, title=None, path=None):
    """
    Gráficas QQ de los residuales contra una Exp(1), una por estación.
    models_by_label: dict etiqueta -> dict estación -> modelo.
    """
    stations = list(data_by_station)
    nrows = int(np.ceil(len(stations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
    for ax, station in zip(axes.flat, stations):
        top = 1.0
        for label, models in models_by_label.items():
            res = np.sort(gpd_residuals(models[station], data_by_station[station]))
            res = res[np.isfinite(res)]
            if len(res) == 0:
                continue
            q = expon.ppf((np.arange(1, len(res) + 1) - 0.5) / len(res))
            ax.plot(q, res, "o", ms=3, alpha=0.6, label=label)
            top = max(top, q.max(), res.max())
        ax.plot([0, top], [0, top], "k--", lw=1)
        ax.set_title(station, fontsize=9)
    for ax in list(axes.flat)[len(stations):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig


# ----------------------------------------------------------------------------
# 4. Alertas
# ----------------------------------------------------------------------------
def aggregate_by_target_date(tables, p_column, event_column="event_c", period="test"):
    """Por estación y día objetivo: evento = máximo del día; probabilidad = máximo del día."""
    parts = []
    for station, table in tables.items():
        t = table[table["period"] == period]
        g = t.groupby("target_date").agg(y=(event_column, "max"), p=(p_column, "max"))
        g["station"] = station
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def threshold_for_recall(y, p, target_recall):
    """El corte más alto que todavía detecta al menos `target_recall` de los eventos."""
    y = np.asarray(y, dtype=bool)
    positives = np.sort(np.asarray(p, dtype=float)[y])[::-1]
    if len(positives) == 0:
        return np.nan
    k = int(np.ceil(target_recall * len(positives)))
    return positives[max(k, 1) - 1]


def alert_metrics(y, p, threshold):
    """Alertas, eventos, tasa de aciertos y razón de falsas alarmas con un corte."""
    y = np.asarray(y, dtype=bool)
    alert = np.asarray(p, dtype=float) >= threshold
    hits = (alert & y).sum()
    return {"threshold": threshold, "alerts": int(alert.sum()), "events": int(y.sum()),
            "hit_rate": hits / y.sum() if y.sum() else np.nan,
            "false_alarm_ratio": (alert & ~y).sum() / alert.sum() if alert.sum() else np.nan}


def confusion_counts(y, p, threshold):
    y = np.asarray(y, dtype=bool)
    alert = np.asarray(p, dtype=float) >= threshold
    return pd.DataFrame([[(~alert & ~y).sum(), (alert & ~y).sum()],
                         [(~alert & y).sum(), (alert & y).sum()]],
                        index=["real: no exceso", "real: exceso"],
                        columns=["pred: no exceso", "pred: exceso"])


def plot_confusion_matrix(y, p, threshold, title=None, ax=None):
    counts = confusion_counts(y, p, threshold).to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay(counts, display_labels=["No exceso", "Exceso"]).plot(ax=ax, colorbar=False)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    if title:
        ax.set_title(title, fontsize=10)
    return ax.figure


def plot_roc(curves, title="Curva ROC", path=None):
    """curves: dict etiqueta -> (y, p). Usa las PROBABILIDADES (no las alertas binarias)."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    for label, (y, p) in curves.items():
        y, p = np.asarray(y, float), np.asarray(p, float)
        ok = ~(np.isnan(y) | np.isnan(p))
        if 0 < y[ok].sum() < ok.sum():
            fpr, tpr, _ = roc_curve(y[ok], p[ok])
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc_score(y[ok], p[ok]):.2f})")
    ax.plot([0, 1], [0, 1], "r--", label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig