"""Carga y preparación de los datos.

Reproduce exactamente los pasos del notebook 04e_gam_gpd:
limpieza -> malla horaria por estación -> relleno de la meteorología -> tabla diaria.
Solo se usa para ENTRENAR y exportar el modelo; la API no lee los datos crudos.
"""
import io

import numpy as np
import pandas as pd

from .config import CLAVES, INICIO_PRUEBA, INICIO_VALIDACION, U, VAR

METEO = ["TOUT", "RH", "SR", "WSR", "PRS", "RAINF"]


def cargar_datos(ruta):
    """Lee el CSV del SIMA y convierte en faltantes los valores imposibles."""
    d = pd.read_csv(ruta, low_memory=False)

    # En 2024 los datos vienen en columnas con unidades, por ejemplo "O3 (ppb)"
    for col in d.columns:
        if " (" in col:
            nombre = col.split(" (")[0]
            d[nombre] = d[nombre].fillna(d[col])

    d["Fecha y hora"] = pd.to_datetime(d["Fecha y hora"])
    d = d[d["Fecha y hora"] >= "2023-01-01"].copy()

    # Factor de la lluvia indicado por el SIMA
    estaciones_x20 = ["CE", "NE", "NO2", "NO3", "NTE2", "SE", "SO"]
    d.loc[d["Estacion"].isin(estaciones_x20), "RAINF"] *= 20

    # Limpieza: valores negativos o imposibles se vuelven faltantes (rangos del Anexo A)
    for v in ["O3", "RAINF", "SR", "WSR", "RH", "PRS"]:
        d.loc[d[v] < 0, v] = np.nan
    d.loc[d["RH"] > 105, "RH"] = np.nan
    d["RH"] = d["RH"].clip(upper=100)
    d.loc[(d["TOUT"] < -10) | (d["TOUT"] > 50), "TOUT"] = np.nan
    d.loc[(d["WDR"] < 0) | (d["WDR"] > 360), "WDR"] = np.nan
    d.loc[d["SR"] > 1.5, "SR"] = np.nan
    d.loc[(d["PRS"] < 650) | (d["PRS"] > 780), "PRS"] = np.nan
    d.loc[d["WSR"] > 150, "WSR"] = np.nan
    d.loc[d["RAINF"] > 100, "RAINF"] = np.nan

    return d[["Fecha y hora", "Estacion", VAR] + METEO + ["WDR"]]


def construir_estaciones(df):
    """Una tabla horaria por estación (malla completa de horas).

    Los huecos de la meteorología se rellenan con la mediana de las demás estaciones a la
    misma hora. El ozono NUNCA se rellena.
    """
    horas = pd.date_range(df["Fecha y hora"].min(), df["Fecha y hora"].max(), freq="h")
    E = {}
    for nombre in sorted(df["Estacion"].unique()):
        s = df[df["Estacion"] == nombre].drop(columns="Estacion")
        s = s.set_index("Fecha y hora").sort_index()
        s = s[~s.index.duplicated()]
        E[nombre] = s.reindex(horas)

    for v in METEO:
        mediana_ciudad = pd.DataFrame({n: E[n][v] for n in E}).median(axis=1)
        for n in E:
            E[n][v] = E[n][v].fillna(mediana_ciudad)
    return E


def leer_pronostico(ruta):
    """Pronóstico de Open-Meteo emitido un día antes (las primeras líneas son información del punto)."""
    lineas = open(ruta, encoding="utf-8").read().splitlines()
    inicio = 0
    while not lineas[inicio].startswith("time"):
        inicio += 1
    pron = pd.read_csv(io.StringIO("\n".join(lineas[inicio:])))
    pron.columns = ["time", "TOUT", "RH", "SR", "WSR", "WDR", "PRS", "RAINF"]
    pron["time"] = pd.to_datetime(pron["time"])
    return pron.set_index("time")


def etiqueta_periodo(fechas):
    fechas = pd.DatetimeIndex(fechas)
    periodo = np.full(len(fechas), "prueba", dtype=object)
    periodo[fechas < INICIO_PRUEBA] = "validación"
    periodo[fechas < INICIO_VALIDACION] = "entrenamiento"
    return periodo


def variables_calendario(fecha_objetivo):
    """Mes (seno y coseno) y fin de semana del día que se predice."""
    f = pd.DatetimeIndex(pd.to_datetime(fecha_objetivo))
    return pd.DataFrame({
        "mes_sen": np.sin(2 * np.pi * (f.month - 1) / 12),
        "mes_cos": np.cos(2 * np.pi * (f.month - 1) / 12),
        "sabado": (f.dayofweek == 5).astype(float),
        "domingo": (f.dayofweek == 6).astype(float),
    })


def tabla_diaria(E, pron):
    """Una fila = (estación, día de hoy). El objetivo es el máximo de MAÑANA (columna `fecha`).

    A diferencia del notebook, aquí NO se borran las filas con faltantes: la API también necesita
    los días cuyo "mañana" no tiene dato (por ejemplo, el último día) para armar escenarios.
    El entrenamiento filtra las filas completas con `filas_completas`.
    """
    o3_dia = pd.DataFrame({st: E[st][VAR].resample("D").max()
                           .where(E[st][VAR].resample("D").count() >= 18) for st in CLAVES})
    ciudad_dia = o3_dia.mean(axis=1)
    t_max_pron = pron["TOUT"].resample("D").max()
    t_min_pron = pron["TOUT"].resample("D").min()

    partes = []
    for st in CLAVES:
        s = E[st]
        t = pd.DataFrame(index=o3_dia.index)
        t["estacion"] = st
        t["hoy"] = t.index
        t["o3max"] = o3_dia[st]
        t["o3media"] = s[VAR].resample("D").mean()
        t["horas63"] = (s[VAR] > U).resample("D").sum().where(t["o3max"].notna())
        t["horas63_7d"] = t["horas63"].rolling(7, min_periods=4).sum()
        t["o3max_ciudad"] = ciudad_dia
        t["T_max_hoy"] = s["TOUT"].resample("D").max()
        t["SR_max_hoy"] = s["SR"].resample("D").max()
        t["RH_media_hoy"] = s["RH"].resample("D").mean()
        t["WSR_media_hoy"] = s["WSR"].resample("D").mean()
        manana = t.index + pd.Timedelta(days=1)
        t["T_max_pron"] = t_max_pron.reindex(manana).to_numpy()
        t["T_min_pron"] = t_min_pron.reindex(manana).to_numpy()
        t["max_manana"] = o3_dia[st].shift(-1)
        cal = variables_calendario(manana)
        for c in cal.columns:
            t[c] = cal[c].to_numpy()
        t["fecha"] = manana
        partes.append(t)

    DIA = pd.concat(partes, ignore_index=True)
    DIA["dT_pron"] = DIA["T_max_pron"] - DIA["T_max_hoy"]
    DIA["rango_pron"] = DIA["T_max_pron"] - DIA["T_min_pron"]
    DIA["y63"] = (DIA["max_manana"] > U).astype(float).where(DIA["max_manana"].notna())
    DIA["periodo"] = etiqueta_periodo(DIA["fecha"])
    return DIA


def filas_completas(DIA, cont):
    """Mismo filtro que el notebook: todas las variables y el objetivo observados."""
    return DIA.dropna(subset=cont + ["T_max_hoy", "T_min_pron", "max_manana"]).reset_index(drop=True)
