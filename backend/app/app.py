"""API del dashboard de ozono (FastAPI).

Correr desde la carpeta backend/:
    uvicorn app.app:app --reload
Documentación interactiva: http://localhost:8000/docs
"""
import json
import math
import os
from datetime import date, timedelta
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from source.config import (C, CARPETA_DATOS, CARPETA_MODELOS, CLAVES, CLIMA_ENTRADA, CONT_D,
                           DESCRIPCIONES, ESTACIONES, OZONO_D, U)
from source.predict import ModeloOzono

# ---------------------------------------------------------------------------
# Carga (una sola vez, al arrancar)
# ---------------------------------------------------------------------------
MODELO = ModeloOzono(CARPETA_MODELOS)
DIARIO = pd.read_csv(CARPETA_DATOS / "diario.csv.gz", parse_dates=["hoy", "fecha"])
HORARIO = pd.read_csv(CARPETA_DATOS / "historico_o3.csv.gz", index_col="fecha_hora", parse_dates=True)
DEPENDENCIA = json.loads((CARPETA_DATOS / "dependencia.json").read_text(encoding="utf-8"))
METRICAS = json.loads((CARPETA_MODELOS / "metricas.json").read_text(encoding="utf-8"))
MEDIANAS = MODELO.gam["referencia_curvas"]["mediana"]
# Al leer el CSV, "alerta" llega como texto: se recalcula a partir de p90 y el corte
DIARIO["alerta"] = (DIARIO["p90"] >= MODELO.tau).where(DIARIO["p90"].notna())

app = FastAPI(
    title="API de ozono · SIMA (Equipo 6)",
    description="Probabilidad de que el máximo de O3 de mañana supere la norma de 90 ppb "
                "(modelo de dos partes: GAM binario + GPD).",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def limpio(x):
    """Convierte NaN y tipos de numpy a algo que JSON entienda."""
    if x is None:
        return None
    if isinstance(x, (np.floating, float)):
        return None if math.isnan(x) else round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def punto(st):
    """Formato que usa el componente Map del frontend (id, name, latitude, longitude)."""
    nombre, lat, lon = ESTACIONES[st]
    return {"id": CLAVES.index(st) + 1, "name": st, "nombre": nombre, "latitude": lat, "longitude": lon}


def validar_estacion(estacion):
    if estacion not in ESTACIONES:
        raise HTTPException(404, f"Estación desconocida: {estacion}. Opciones: {CLAVES}")


# ---------------------------------------------------------------------------
# Generales
# ---------------------------------------------------------------------------
@app.get("/", tags=["General"])
def inicio():
    return {"mensaje": "API de ozono del Equipo 6. Documentación en /docs",
            "u_ppb": U, "norma_ppb": C, "corte_alerta": MODELO.tau}


@app.get("/estaciones", tags=["General"])
def estaciones():
    """Las 15 estaciones del SIMA con sus coordenadas."""
    return [punto(st) for st in CLAVES]


@app.get("/metricas", tags=["General"])
def metricas():
    """Desempeño del modelo en validación (2025) y prueba (2026)."""
    return METRICAS


# ---------------------------------------------------------------------------
# Tab 1: datos pasados
# ---------------------------------------------------------------------------
@app.get("/historico/rango", tags=["Tab 1 · Histórico"])
def historico_rango():
    """Primer y último día con datos horarios."""
    return {"inicio": HORARIO.index.min().date().isoformat(), "fin": HORARIO.index.max().date().isoformat()}


@app.get("/historico/hora", tags=["Tab 1 · Histórico"])
def historico_hora(fecha: date, hora: int = Query(..., ge=0, le=23)):
    """Ozono de cada estación a esa hora y la predicción del modelo PARA ESE DÍA.

    El modelo es diario: predice el máximo de `fecha` con los datos del día anterior, así que la
    predicción es la misma para todas las horas del día. `exceso_real` dice si el máximo observado
    del día superó 90 ppb. Las predicciones de 2023-2024 son de entrenamiento (el modelo ya vio esos datos).
    """
    momento = pd.Timestamp(fecha) + pd.Timedelta(hours=hora)
    if momento not in HORARIO.index:
        raise HTTPException(404, "Fecha u hora fuera del rango de datos (ver /historico/rango)")
    fila = HORARIO.loc[momento]
    pred = DIARIO[DIARIO["fecha"] == pd.Timestamp(fecha)].set_index("estacion")
    puntos = []
    for st in CLAVES:
        p = pred.loc[st] if st in pred.index else None
        puntos.append({
            **punto(st),
            "value": limpio(fila[st]),                       # O3 de esa hora (ppb), null si no se midió
            "periodo": None if p is None else p["periodo"],
            "zeta": None if p is None else limpio(p["zeta"]),
            "p90": None if p is None else limpio(p["p90"]),
            "alerta": None if p is None or pd.isna(p["alerta"]) else bool(p["alerta"]),
            "max_dia": None if p is None else limpio(p["max_manana"]),
            "exceso_real": None if p is None or pd.isna(p["max_manana"]) else bool(p["max_manana"] > C),
        })
    return {"fecha": fecha.isoformat(), "hora": hora, "corte_alerta": MODELO.tau, "puntos": puntos}


@app.get("/historico/serie", tags=["Tab 1 · Histórico"])
def historico_serie(estacion: str, fecha: date, hora: int = Query(23, ge=0, le=23),
                    dias: int = Query(7, ge=1, le=60)):
    """Serie horaria de O3 de una estación en los `dias` anteriores al momento indicado."""
    validar_estacion(estacion)
    fin = pd.Timestamp(fecha) + pd.Timedelta(hours=hora)
    serie = HORARIO.loc[fin - pd.Timedelta(days=dias) + pd.Timedelta(hours=1): fin, estacion]
    return {"estacion": estacion, "norma_ppb": C, "u_ppb": U,
            "serie": [{"fecha_hora": t.isoformat(), "o3": limpio(v)} for t, v in serie.items()]}


# ---------------------------------------------------------------------------
# Tab 2: predicciones y escenarios
# ---------------------------------------------------------------------------
class Clima(BaseModel):
    """Clima del escenario. Lo que no se mande se toma del día base."""
    SR_max_hoy: float | None = Field(None, ge=0, le=1.5, description="Radiación solar máxima de hoy (kW/m²)")
    RH_media_hoy: float | None = Field(None, ge=0, le=100, description="Humedad relativa media de hoy (%)")
    WSR_media_hoy: float | None = Field(None, ge=0, le=150, description="Velocidad media del viento de hoy (km/h)")
    T_max_hoy: float | None = Field(None, ge=-10, le=50, description="Temperatura máxima de hoy (°C)")
    T_max_pron: float | None = Field(None, ge=-10, le=50, description="Máxima pronosticada para mañana (°C)")
    T_min_pron: float | None = Field(None, ge=-10, le=50, description="Mínima pronosticada para mañana (°C)")


class Escenario(BaseModel):
    fecha_base: date | None = Field(None, description="Día 'de hoy' del que se toman el ozono y el clima. "
                                                      "Por defecto, el último día con datos.")
    fecha_objetivo: date | None = Field(None, description="Día que se predice (define mes y fin de semana). "
                                                          "Por defecto, fecha_base + 1.")
    clima: Clima | None = Field(None, description="Clima para TODAS las estaciones")
    estaciones: dict[str, dict[str, float]] | None = Field(
        None, description="Cambios por estación, p. ej. {'SUR': {'o3max': 95}}")
    niveles: list[float] = Field([C], description="Niveles (ppb) para los que se calcula P(máx > nivel); >= 63")


def dia_base(fecha_base):
    if fecha_base is None:
        fecha_base = DIARIO.dropna(subset=["o3max"])["hoy"].max().date()
    filas = DIARIO[DIARIO["hoy"] == pd.Timestamp(fecha_base)].set_index("estacion")
    if filas.empty:
        raise HTTPException(404, "No hay datos para esa fecha_base")
    return fecha_base, filas


def armar_escenario(esc: Escenario):
    fecha_base, filas = dia_base(esc.fecha_base)
    objetivo = esc.fecha_objetivo or fecha_base + timedelta(days=1)
    clima = esc.clima.model_dump(exclude_none=True) if esc.clima else {}
    cambios = esc.estaciones or {}
    for st, valores in cambios.items():
        validar_estacion(st)
        malas = set(valores) - set(OZONO_D + CLIMA_ENTRADA)
        if malas:
            raise HTTPException(422, f"Variables no válidas para {st}: {sorted(malas)}")

    registros, completados = [], {}
    for st in CLAVES:
        fila = {"estacion": st, "fecha": pd.Timestamp(objetivo)}
        for v in OZONO_D + CLIMA_ENTRADA:
            valor = filas.loc[st, v] if st in filas.index else np.nan
            valor = clima.get(v, valor)
            valor = cambios.get(st, {}).get(v, valor)
            if pd.isna(valor):                       # sin dato: mediana de entrenamiento
                valor = MEDIANAS.get(v, np.nan)
                if pd.isna(valor):
                    valor = float(DIARIO[v].median())
                completados.setdefault(st, []).append(v)
            fila[v] = float(valor)
        registros.append(fila)
    return fecha_base, objetivo, pd.DataFrame(registros), completados


@app.get("/escenario/base", tags=["Tab 2 · Predicciones"])
def escenario_base(fecha: date | None = None):
    """Valores reales de un día para llenar el sidebar (por defecto, el último día con datos).

    `clima_ciudad` es el promedio de las 15 estaciones: sirve como valor inicial de los controles.
    """
    fecha_base, objetivo, datos, completados = armar_escenario(Escenario(fecha_base=fecha))
    clima_ciudad = {v: limpio(datos[v].mean()) for v in CLIMA_ENTRADA}
    return {
        "fecha_base": fecha_base.isoformat(), "fecha_objetivo": objetivo.isoformat(),
        "clima_ciudad": clima_ciudad,
        "estaciones": {r["estacion"]: {v: limpio(r[v]) for v in OZONO_D + CLIMA_ENTRADA}
                       for r in datos.to_dict("records")},
        "valores_completados": completados,
        "controles": {v: {"descripcion": DESCRIPCIONES[v][0], "unidad": DESCRIPCIONES[v][1],
                          "min": MODELO.gam["splines"][v]["limites"][0] if v in CONT_D else None,
                          "max": MODELO.gam["splines"][v]["limites"][1] if v in CONT_D else None}
                      for v in OZONO_D + CLIMA_ENTRADA},
        "nota": "El modelo no usa la dirección del viento, la presión ni la lluvia.",
    }


@app.post("/prediccion", tags=["Tab 2 · Predicciones"])
def prediccion(esc: Escenario):
    """P(máximo de mañana > 63 ppb) con el GAM y P(> 90 ppb) con la GPD, por estación.

    Si un valor queda fuera del rango con que se entrenó el modelo, se recorta al límite y se
    indica en `recortadas` (los splines no extrapolan).
    """
    if any(n < U for n in esc.niveles):
        raise HTTPException(422, f"Los niveles deben ser >= {U} ppb")
    niveles = tuple(dict.fromkeys([C] + esc.niveles))
    fecha_base, objetivo, datos, completados = armar_escenario(esc)
    pred = MODELO.predecir(datos, niveles=niveles)
    puntos = []
    for r, p in zip(datos.to_dict("records"), pred.to_dict("records")):
        st = r["estacion"]
        puntos.append({
            **punto(st),
            "value": limpio(p["p90"]),                              # para colorear el mapa
            "zeta": limpio(p["zeta"]),                              # P(máx > 63) · GAM
            "p90": limpio(p["p90"]),                                # P(máx > 90) · GAM + GPD
            "probabilidades": {f"{n:g}": limpio(p[f"p{n:g}"]) for n in niveles},
            "sigma": limpio(p["sigma"]),
            "alerta": bool(p["alerta"]),
            "entradas": {v: limpio(r[v]) for v in OZONO_D + CLIMA_ENTRADA},
            "recortadas": p["recortadas"],
            "completadas": completados.get(st, []),
        })
    return {"fecha_base": fecha_base.isoformat(), "fecha_objetivo": objetivo.isoformat(),
            "u_ppb": U, "norma_ppb": C, "corte_alerta": MODELO.tau, "xi": MODELO.xi, "puntos": puntos}


@app.get("/gam/curvas", tags=["Tab 2 · Predicciones"])
def gam_curvas(estacion: str | None = None, puntos: int = Query(100, ge=10, le=400)):
    """Curvas del GAM (efecto de cada variable en logit zeta, con las demás en su mediana).

    Las curvas son las MISMAS para todas las estaciones (es un solo GAM); la estación solo
    sube o baja la curva completa (`efecto_estacion`). Si se manda `estacion`, se agrega la curva
    en escala de probabilidad para un día típico de esa estación.
    """
    if estacion is not None:
        validar_estacion(estacion)
    curvas = MODELO.curvas(estacion, puntos)
    for v, c in curvas.items():
        c["descripcion"], c["unidad"] = DESCRIPCIONES[v]
    return {"estacion": estacion, "efecto_estacion": MODELO.gam["efecto_estacion"],
            "referencia": MODELO.gam["referencia"], "curvas": curvas}


# ---------------------------------------------------------------------------
# Tab 3: dependencia entre estaciones
# ---------------------------------------------------------------------------
@app.get("/dependencia", tags=["Tab 3 · Dependencia"])
def dependencia(tipo: Literal["chi", "chibar", "spearman"] = "chi"):
    """Matriz de dependencia entre estaciones (máximos diarios de O3, 2023-2025).

    - `chi`: chi(0.95), probabilidad de que una estación tenga un extremo dado que la otra lo tiene.
    - `chibar`: chi barra, si la dependencia se mantiene en los valores más altos.
    - `spearman`: correlación de rangos en toda la distribución.
    `pares` trae cada par una sola vez, listo para dibujar líneas.
    """
    M = DEPENDENCIA[tipo]
    est = DEPENDENCIA["estaciones"]
    pares = []
    for i, a in enumerate(est):
        for j in range(i + 1, len(est)):
            pares.append({"a": a, "b": est[j], "valor": limpio(M[i][j]),
                          "distancia_km": DEPENDENCIA["distancia_km"][i][j],
                          "dias_conjuntos": DEPENDENCIA["dias_conjuntos"][i][j]})
    return {"tipo": tipo, "q": DEPENDENCIA["q"], "periodo": DEPENDENCIA["periodo"],
            "estaciones": [punto(st) for st in est], "matriz": M, "pares": pares}
