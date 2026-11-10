"""
Regresión logística para zeta = P(X_{t+h} > u | información en t).

Los modelos se guardan como diccionarios:
    {"result": resultado de statsmodels, "mean", "std", "features", "target"}
Las variables se estandarizan con la media y desviación de ENTRENAMIENTO.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_logit(data, features, target="event", cluster_col=None, maxiter=200, min_events=10):
    """
    Ajusta una logística con variables estandarizadas.

    - Quita variables constantes (por ejemplo lluvia en estaciones sin pluviómetro).
    - cluster_col: columna para errores estándar robustos por clúster (las horas de un
      mismo día no son independientes). No cambia los coeficientes ni la verosimilitud.
    """
    data = data.dropna(subset=list(features) + [target])
    events = int(data[target].sum())
    if events < min_events or events > len(data) - min_events:
        raise ValueError(f"Muy pocos eventos para ajustar la logística ({events} de {len(data)}).")

    features = [f for f in features if data[f].std() > 1e-9]
    mean, std = data[features].mean(), data[features].std()
    X = sm.add_constant((data[features] - mean) / std, has_constant="add")

    model = sm.Logit(data[target], X)
    if cluster_col is None:
        result = model.fit(disp=0, maxiter=maxiter)
    else:
        groups = pd.factorize(data[cluster_col])[0]
        result = model.fit(disp=0, maxiter=maxiter, cov_type="cluster", cov_kwds={"groups": groups})
    return {"result": result, "mean": mean, "std": std, "features": features, "target": target}


def design_matrix_logit(model, data):
    """Matriz de diseño estandarizada con la media y desviación de entrenamiento."""
    z = (data[model["features"]] - model["mean"]) / model["std"]
    return sm.add_constant(z, has_constant="add")


def predict_logit(model, data):
    """Probabilidades predichas (Series con el índice de `data`)."""
    return model["result"].predict(design_matrix_logit(model, data))


def fit_logit_all_stations(tables, features, target="event", cluster_col=None,
                           train_period="train", skip_failed=False):
    """
    Ajusta una logística por estación con las filas de entrenamiento.
    features: lista (igual para todas) o diccionario estación -> lista.
    skip_failed=True omite (con aviso) las estaciones con muy pocos eventos en lugar de fallar.
    """
    models = {}
    for station, table in tables.items():
        feats = features[station] if isinstance(features, dict) else features
        train = table[table["period"] == train_period]
        try:
            models[station] = fit_logit(train, feats, target, cluster_col)
        except Exception as error:
            if not skip_failed:
                raise RuntimeError(f"Estación {station}: {error}") from error
            print(f"Aviso: estación {station} omitida ({error})")
    return models


def fit_logit_families(tables, families, target="event", cluster_col=None, train_period="train",
                       skip_failed=False):
    """
    families: dict nombre -> lista de variables.
    Regresa dict nombre -> dict estación -> modelo.
    """
    return {name: fit_logit_all_stations(tables, feats, target, cluster_col, train_period, skip_failed)
            for name, feats in families.items()}


def add_logit_predictions(tables, models, column, fallback_column=None):
    """
    Agrega la columna `column` con la probabilidad predicha (todas las filas) a cada tabla.
    Si una estación no tiene modelo (skip_failed) se copia `fallback_column` (por ejemplo la
    climatología) o se deja NaN.
    """
    for station, table in tables.items():
        if station in models:
            table[column] = predict_logit(models[station], table)
        else:
            table[column] = table[fallback_column] if fallback_column else np.nan
    return tables


def logit_odds_table(model):
    """Coeficientes, razones de momios (RM), intervalo de confianza al 95 % y p-valores."""
    res = model["result"]
    ci = np.exp(res.conf_int())
    return pd.DataFrame({"coef": res.params, "odds_ratio": np.exp(res.params),
                         "CI_low": ci[0], "CI_high": ci[1], "p": res.pvalues})


def logit_information_criteria(families):
    """AIC y log-verosimilitud de ENTRENAMIENTO por familia y estación (referencia, no para elegir)."""
    rows = []
    for family, by_station in families.items():
        for station, model in by_station.items():
            res = model["result"]
            rows.append({"family": family, "station": station, "n_params": res.df_model + 1,
                         "llf_train": res.llf, "aic_train": res.aic})
    return pd.DataFrame(rows)