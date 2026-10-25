"""
Preparación de datos.

Carga y limpieza del CSV del SIMA, construcción de series por estación (horarias
y diarias), PCA de la meteorología y tablas de pronóstico (una fila = lo que se
sabe en el instante t para predecir el instante t + horizonte).

Flujo típico:
    raw      = load_data("data/datos_no_imputados.csv")
    hourly, report = build_hourly_station_data(raw)
    pca      = fit_meteorological_pca(hourly); apply_meteorological_pca(hourly, pca)
    tables   = build_all_forecast_tables(hourly, horizon=24, frequency="hourly")
"""

import re

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Constantes del problema
# ----------------------------------------------------------------------------
VARIABLE = "O3"            # contaminante
THRESHOLD_U = 63.0         # umbral del modelo POT (ppb)
STANDARD_C = 90.0          # norma de 1 hora (ppb)
DATA_START = "2023-01-01"
TRAIN_END = "2025-01-01"   # primer instante del periodo de prueba (se asigna por hora OBJETIVO)

MET_VARIABLES = ["TOUT", "RH", "SR", "WSR", "PRS", "RAINF"]
PCA_VARIABLES = ["TOUT", "RH", "SR", "WSR", "PRS", "RAINF_log1p", "WDR_sin", "WDR_cos"]
FORECAST_MET_COLUMNS = ["TOUT", "RH", "SR", "WSR", "PRS", "RAINF_log1p",
                        "wind_u", "wind_v", "WDR_sin", "WDR_cos"]

# Estaciones cuya lluvia debe multiplicarse por 20 (indicación del SIMA)
RAIN_X20_STATIONS = ["CE", "NE", "NO2", "NO3", "NTE2", "SE", "SO"]

FREQUENCY_STEP = {"hourly": pd.Timedelta(hours=1), "daily": pd.Timedelta(days=1)}

# Variables de persistencia (ventanas en pasos de la propia serie: horas o días)
PERSISTENCE_SPECS = {
    "hourly": {"lags": (1, 24), "max_windows": (24,), "count_windows": (24, 168)},
    "daily": {"lags": (1,), "max_windows": (7,), "count_windows": (7, 30)},
}


# ----------------------------------------------------------------------------
# Carga y limpieza
# ----------------------------------------------------------------------------
def clean_measurements(data, variable=VARIABLE):
    """Valores negativos o físicamente imposibles pasan a NaN (no se imputa nada)."""
    d = data.copy()
    numeric = [variable, "TOUT", "RH", "SR", "WSR", "PRS", "RAINF", "WDR"]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    for col in [variable, "RAINF", "SR", "WSR", "RH", "PRS"]:
        d.loc[d[col] < 0, col] = np.nan
    d.loc[d["RH"] > 105, "RH"] = np.nan
    d["RH"] = d["RH"].clip(upper=100)
    d.loc[(d["TOUT"] < -10) | (d["TOUT"] > 50), "TOUT"] = np.nan
    d.loc[(d["WDR"] < 0) | (d["WDR"] > 360), "WDR"] = np.nan
    d.loc[d["SR"] > 1.5, "SR"] = np.nan
    d.loc[(d["PRS"] < 650) | (d["PRS"] > 780), "PRS"] = np.nan
    d.loc[d["WSR"] > 150, "WSR"] = np.nan
    d.loc[d["RAINF"] > 100, "RAINF"] = np.nan
    return d


def load_data(path, start=DATA_START, end=None, apply_rain_factor=True, variable=VARIABLE):
    """
    Lee el CSV del SIMA y regresa un DataFrame con columnas
    ['Fecha y hora', 'Estacion', variable, TOUT, RH, SR, WSR, PRS, RAINF, WDR].

    apply_rain_factor : aplica el factor x20 de lluvia a RAINF_X20_STATIONS.
        Ponerlo en False si el archivo ya trae la lluvia corregida.
    """
    raw = pd.read_csv(path, low_memory=False)

    # Algunas columnas vienen con unidades, por ejemplo "O3 (ppb)"
    for col in list(raw.columns):
        if " (" in col:
            base = col.split(" (")[0]
            raw[base] = raw[base].fillna(raw[col]) if base in raw.columns else raw[col]

    needed = ["Fecha y hora", "Estacion", variable, *MET_VARIABLES, "WDR"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise KeyError(f"Faltan columnas en {path}: {missing}")

    d = raw[needed].copy()
    d["Fecha y hora"] = pd.to_datetime(d["Fecha y hora"])
    d = d[d["Fecha y hora"] >= start]
    if end is not None:
        d = d[d["Fecha y hora"] < end]

    d["RAINF"] = pd.to_numeric(d["RAINF"], errors="coerce")
    if apply_rain_factor:
        d.loc[d["Estacion"].isin(RAIN_X20_STATIONS), "RAINF"] *= 20

    return clean_measurements(d, variable).reset_index(drop=True)


# ----------------------------------------------------------------------------
# Series por estación
# ----------------------------------------------------------------------------
def calendar_features(times, hours=None):
    """
    Variables cíclicas de calendario (diccionario de arreglos).
    hours: horas a usar; si es None se toma la hora de `times`.
    """
    times = pd.DatetimeIndex(times)
    h = np.asarray(times.hour if hours is None else hours, dtype=float)
    m = np.asarray(times.month, dtype=float)
    return {
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "hour_sin2": np.sin(4 * np.pi * h / 24),   # segundo armónico
        "hour_cos2": np.cos(4 * np.pi * h / 24),
        "month_sin": np.sin(2 * np.pi * (m - 1) / 12),
        "month_cos": np.cos(2 * np.pi * (m - 1) / 12),
    }


def add_derived_variables(frame, wind_rotation_deg=0.0):
    """
    Agrega lluvia en log, dirección del viento circular y componentes u, v.
    wind_u: componente hacia el este; wind_v: hacia el norte (el viento viene de WDR).
    wind_rotation_deg: gira los ejes (por ejemplo 45 para componentes NO y SO).
    """
    s = frame.copy()
    s["RAINF_log1p"] = np.log1p(s["RAINF"])            # la lluvia casi siempre es 0
    angle = np.deg2rad(s["WDR"])
    s["WDR_sin"] = np.sin(angle)                       # la dirección es circular
    s["WDR_cos"] = np.cos(angle)
    rotated = np.deg2rad(s["WDR"] - wind_rotation_deg)
    s["wind_u"] = -s["WSR"] * np.sin(rotated)
    s["wind_v"] = -s["WSR"] * np.cos(rotated)
    return s


def build_hourly_station_data(df, wind_rotation_deg=0.0, fill_meteorology=True):
    """
    Cada estación en una malla horaria COMPLETA (una fila = una hora); así los
    desfases y ventanas móviles siempre cuentan horas reales.

    Los huecos de meteorología se llenan con la mediana de las demás estaciones
    a la misma hora (dirección: promedio circular). El ozono NUNCA se rellena.

    Regresa (dict estación -> DataFrame, DataFrame con la fracción rellenada).
    """
    hours = pd.date_range(df["Fecha y hora"].min().floor("h"),
                          df["Fecha y hora"].max().floor("h"), freq="h")
    raw = {}
    for station in sorted(df["Estacion"].unique()):
        s = df[df["Estacion"] == station].drop(columns="Estacion")
        s = s.set_index("Fecha y hora").sort_index()
        s = s[~s.index.duplicated()]
        raw[station] = s.reindex(hours)

    # Horas con toda la meteorología medida (el PCA solo usa estas)
    for s in raw.values():
        s["met_observed"] = s[MET_VARIABLES + ["WDR"]].notna().all(axis=1)

    report = {station: {} for station in raw}
    for var in MET_VARIABLES:
        city_median = pd.DataFrame({st: raw[st][var] for st in raw}).median(axis=1)
        for st in raw:
            report[st][var] = raw[st][var].isna().mean()
            if fill_meteorology:
                raw[st][var] = raw[st][var].fillna(city_median)

    sin_mean = pd.DataFrame({st: np.sin(np.deg2rad(raw[st]["WDR"])) for st in raw}).mean(axis=1)
    cos_mean = pd.DataFrame({st: np.cos(np.deg2rad(raw[st]["WDR"])) for st in raw}).mean(axis=1)
    city_direction = (np.degrees(np.arctan2(sin_mean, cos_mean)) + 360) % 360
    for st in raw:
        report[st]["WDR"] = raw[st]["WDR"].isna().mean()
        if fill_meteorology:
            raw[st]["WDR"] = raw[st]["WDR"].fillna(city_direction)

    frames = {st: add_derived_variables(raw[st], wind_rotation_deg) for st in raw}
    return frames, pd.DataFrame(report).T


def build_daily_station_data(hourly_frames, min_hours=18, variable=VARIABLE):
    """
    Serie diaria por estación: máximo diario de O3 y la fila completa (meteorología)
    de la hora en que ocurrió ese máximo. Agrega 'peak_time' y 'peak_hour'.
    Los días con menos de `min_hours` horas medidas quedan en NaN (el máximo no es confiable).
    La malla diaria es completa, así que shift(1) siempre es "el día anterior".
    """
    daily = {}
    for station, s in hourly_frames.items():
        valid = s[variable].dropna()
        day = valid.index.normalize()
        peak_time = valid.groupby(day).idxmax()
        n_hours = valid.groupby(day).size()
        peak_time = peak_time[n_hours.reindex(peak_time.index).to_numpy() >= min_hours]

        rows = s.loc[peak_time.to_numpy()].copy()
        rows["peak_time"] = peak_time.to_numpy()
        rows["peak_hour"] = pd.DatetimeIndex(peak_time.to_numpy()).hour
        rows.index = pd.DatetimeIndex(peak_time.index)

        full = pd.date_range(s.index.min().normalize(), s.index.max().normalize(), freq="D")
        rows = rows.reindex(full)
        rows["met_observed"] = rows["met_observed"].fillna(False).astype(bool)
        daily[station] = rows
    return daily


# ----------------------------------------------------------------------------
# PCA de la meteorología (se ajusta SOLO con entrenamiento)
# ----------------------------------------------------------------------------
def fit_meteorological_pca(frames, train_end=TRAIN_END, variables=PCA_VARIABLES, n_components=5):
    """PCA sobre la matriz de correlación; solo filas de entrenamiento con meteorología medida."""
    parts = []
    for s in frames.values():
        mask = (s.index < pd.Timestamp(train_end)) & s["met_observed"]
        parts.append(s.loc[mask, variables])
    train = pd.concat(parts).dropna()

    mean, std = train.mean(), train.std()
    eigenvalues, eigenvectors = np.linalg.eigh(train.corr().to_numpy())
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    k = n_components
    names = [f"PC{i + 1}" for i in range(k)]
    return {
        "variables": list(variables), "mean": mean, "std": std,
        "eigenvalues": eigenvalues[:k], "eigenvectors": eigenvectors[:, :k],
        "components": names,
        "explained_cumulative": np.cumsum(eigenvalues)[:k] / eigenvalues.sum(),
        "loadings": pd.DataFrame(eigenvectors[:, :k] * np.sqrt(eigenvalues[:k]),
                                 index=variables, columns=names),
    }


def apply_meteorological_pca(frames, pca):
    """Agrega PC1..PCk a cada serie (puntajes con varianza 1). Modifica los frames en sitio."""
    for s in frames.values():
        z = (s[pca["variables"]] - pca["mean"]) / pca["std"]
        scores = z.to_numpy() @ pca["eigenvectors"] / np.sqrt(pca["eigenvalues"])
        for i, name in enumerate(pca["components"]):
            s[name] = scores[:, i]
    return frames


# ----------------------------------------------------------------------------
# Tablas de pronóstico
# ----------------------------------------------------------------------------
def build_forecast_table(frame, horizon, frequency, u=THRESHOLD_U, c=STANDARD_C,
                         train_end=TRAIN_END, persistence=None, variable=VARIABLE):
    """
    Una fila por instante t con lo que se sabe en t y lo que ocurre en t + horizonte.

    Columnas principales
        x_target, event (x_target > u), event_c (x_target > c)
        x_t, exc_t, x_lag*, max*, n_exc*   (persistencia)
        meteorología en t (y PC1..PCk si existen)
        hour_*, month_*                    (calendario del instante OBJETIVO; en la serie
                                            diaria la hora es la del pico del día t)
        target_time, target_date, target_month, target_hour, target_month_num
        period ('train' / 'test') según el instante objetivo
    Las ventanas usan solo el pasado, así que no hay fuga de información.
    """
    persistence = persistence or PERSISTENCE_SPECS[frequency]
    step = FREQUENCY_STEP[frequency]
    x = frame[variable]
    exceed = (x > u).astype(float).where(x.notna())

    d = pd.DataFrame(index=frame.index)
    d["x_target"] = x.shift(-horizon)
    missing_target = d["x_target"].isna()
    d["event"] = np.where(missing_target, np.nan, (d["x_target"] > u).astype(float))
    d["event_c"] = np.where(missing_target, np.nan, (d["x_target"] > c).astype(float))

    d["x_t"] = x
    d["exc_t"] = exceed
    for lag in persistence["lags"]:
        d[f"x_lag{lag}"] = x.shift(lag)
    for w in persistence["max_windows"]:
        d[f"max{w}"] = x.rolling(w, min_periods=max(1, w // 2)).max()
    for w in persistence["count_windows"]:
        d[f"n_exc{w}"] = exceed.rolling(w, min_periods=max(1, w // 2)).sum()

    pc_columns = [col for col in frame.columns if re.fullmatch(r"PC\d+", col)]
    for col in [m for m in FORECAST_MET_COLUMNS if m in frame.columns] + pc_columns:
        d[col] = frame[col]

    target_time = d.index + horizon * step
    hours = frame["peak_hour"].to_numpy() if "peak_hour" in frame.columns else None
    for name, values in calendar_features(target_time, hours).items():
        d[name] = values

    d["target_time"] = target_time
    d["target_date"] = target_time.normalize()
    d["target_month"] = target_time.strftime("%Y-%m")
    d["target_hour"] = target_time.hour
    d["target_month_num"] = target_time.month
    d["period"] = np.where(target_time < pd.Timestamp(train_end), "train", "test")
    return d


def build_all_forecast_tables(frames, horizon, frequency, u=THRESHOLD_U, c=STANDARD_C,
                              train_end=TRAIN_END, persistence=None):
    return {station: build_forecast_table(frame, horizon, frequency, u, c, train_end, persistence)
            for station, frame in frames.items()}


def complete_cases(tables, columns, required=("event", "event_c", "x_target")):
    """
    Deja solo las filas con todas las columnas presentes, para que TODOS los modelos
    se comparen sobre exactamente las mismas filas.
    """
    needed = sorted(set(columns) | set(required))
    return {station: t.dropna(subset=needed).copy() for station, t in tables.items()}


def add_climatology_reference(tables, key, targets=("event", "event_c"), train_period="train"):
    """
    Probabilidad de referencia: frecuencia observada en entrenamiento por `key`
    ('target_hour' para series horarias, 'target_month_num' para diarias).
    Crea las columnas ref_<target>.
    """
    for t in tables.values():
        train = t[t["period"] == train_period]
        for target in targets:
            freq = train.groupby(key)[target].mean()
            t[f"ref_{target}"] = t[key].map(freq).fillna(train[target].mean())
    return tables


def count_events_by_period(frames, u=THRESHOLD_U, c=STANDARD_C, train_end=TRAIN_END,
                           variable=VARIABLE):
    """Número de horas (o días) por encima de u y de c, por estación y periodo."""
    rows = []
    for station, s in frames.items():
        row = {"station": station}
        is_train = s.index < pd.Timestamp(train_end)
        for label, mask in [("train", is_train), ("test", ~is_train)]:
            x = s.loc[mask, variable]
            row[f"{label} > u"] = int((x > u).sum())
            row[f"{label} > c"] = int((x > c).sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("station")