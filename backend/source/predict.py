"""Predicción con el modelo de dos partes (GAM binario + GPD) a partir de los parámetros en JSON.

No usa statsmodels: reconstruye las bases de los splines con scipy y multiplica por los
coeficientes guardados. `entrenar.py` verifica que los resultados coincidan con statsmodels.

    P(X > c) = zeta * [1 + xi (c - u) / sigma]^(-1/xi)

    zeta  = P(máximo de mañana > u)           <- GAM con respuesta binaria
    sigma = exp(b0 + b1 logit(zeta) + spline)  <- GPD del exceso (xi común)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import splev

from .config import CAL_D, CARPETA_MODELOS, CONT_D
from .datos import variables_calendario


# ---------------------------------------------------------------------------
# Bases de los splines
# ---------------------------------------------------------------------------
def base_bspline(x, nudos, grado):
    """Base B-spline igual a statsmodels (`include_intercept=False`: se omite la primera columna)."""
    x = np.asarray(x, dtype=float)
    nudos = np.asarray(nudos, dtype=float)
    n_bases = len(nudos) - (grado + 1) - 1
    base = np.empty((len(x), n_bases))
    for i in range(n_bases):
        coefs = np.zeros(n_bases + 1)
        coefs[i + 1] = 1.0
        base[:, i] = splev(x, (nudos, coefs, grado))
    return base


def spline_restringido(x, nudos):
    """Spline cúbico restringido (Harrell, 2015): lineal antes del primer nudo y después del último."""
    x = np.asarray(x, dtype=float)
    t = np.asarray(nudos, dtype=float)
    k = len(t)
    pos = lambda z: np.clip(z, 0, None) ** 3
    cols = []
    for j in range(k - 2):
        f = (pos(x - t[j])
             - pos(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
             + pos(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        cols.append(f / (t[-1] - t[0]) ** 2)
    return np.column_stack(cols)


def prob_mayor_gpd(x, sigma, xi):
    """P(Y > x) para Y ~ GPD(sigma, xi); vale 0 fuera del soporte."""
    sigma = np.asarray(sigma, dtype=float)
    if abs(xi) < 1e-8:
        return np.exp(-x / sigma)
    t = 1 + xi * x / sigma
    resultado = np.zeros(len(sigma))
    resultado[t > 0] = t[t > 0] ** (-1 / xi)
    return resultado


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class ModeloOzono:
    def __init__(self, carpeta=CARPETA_MODELOS):
        carpeta = Path(carpeta)
        self.gam = json.loads((carpeta / "gam.json").read_text(encoding="utf-8"))
        self.gpd = json.loads((carpeta / "gpd.json").read_text(encoding="utf-8"))
        self.u = self.gpd["u"]
        self.c = self.gpd["c"]
        self.tau = self.gpd["tau"]
        self.xi = self.gpd["xi"]
        self.estaciones = self.gam["estaciones"]

    # --- preparación de las variables ---------------------------------------
    @staticmethod
    def completar_variables(datos):
        """Agrega dT_pron y rango_pron a partir de las temperaturas, si no vienen ya."""
        d = datos.copy()
        if "dT_pron" not in d or d["dT_pron"].isna().any():
            d["dT_pron"] = d["T_max_pron"] - d["T_max_hoy"]
        if "rango_pron" not in d or d["rango_pron"].isna().any():
            d["rango_pron"] = d["T_max_pron"] - d["T_min_pron"]
        if "fecha" in d and any(c not in d for c in CAL_D):
            cal = variables_calendario(d["fecha"])
            for c in CAL_D:
                d[c] = cal[c].to_numpy()
        return d

    def acotar(self, datos):
        """Los splines solo están definidos dentro del rango de los datos: se recortan los valores.

        Devuelve los datos recortados y, por fila, la lista de variables que se recortaron.
        """
        d = datos.copy()
        recortes = [[] for _ in range(len(d))]
        for v in CONT_D:
            lo, hi = self.gam["splines"][v]["limites"]
            x = d[v].to_numpy(dtype=float)
            fuera = (x < lo) | (x > hi)
            for i in np.flatnonzero(fuera):
                recortes[i].append(v)
            d[v] = np.clip(x, lo, hi)
        return d, recortes

    # --- primera parte: GAM -------------------------------------------------
    def predictor_lineal(self, datos):
        """logit(zeta) = constante + calendario + efecto de la estación + suma de las curvas."""
        d = datos
        n = len(d)
        coef = self.gam["coef_lineales"]
        eta = np.full(n, coef["const"])
        for c in CAL_D:
            eta = eta + coef[c] * d[c].to_numpy(dtype=float)
        eta = eta + d["estacion"].map(self.gam["efecto_estacion"]).to_numpy(dtype=float)
        for v in CONT_D:
            eta = eta + self.curva(v, d[v].to_numpy(dtype=float))
        return eta

    def curva(self, v, x):
        """f_v(x): contribución de la variable v al logit de zeta (sin centrar)."""
        s = self.gam["splines"][v]
        return base_bspline(x, s["nudos"], s["grado"]) @ np.asarray(s["coef"])

    def zeta(self, datos):
        return 1 / (1 + np.exp(-self.predictor_lineal(datos)))

    # --- segunda parte: GPD -------------------------------------------------
    def sigma(self, zeta):
        g = self.gpd
        lz = logit(zeta)
        cov = np.column_stack([lz, spline_restringido(lz, g["nudos_logit_zeta"])])
        Z = (cov - np.asarray(g["media"])) / np.asarray(g["desv"])
        b = np.asarray(g["b"])
        return np.exp(b[0] + Z @ b[1:])

    def prob_superar(self, nivel, zeta, sigma):
        """P(máximo de mañana > nivel) para cualquier nivel >= u."""
        if nivel < self.u:
            raise ValueError(f"El modelo solo extrapola por arriba de u = {self.u} ppb")
        return np.asarray(zeta) * prob_mayor_gpd(nivel - self.u, sigma, self.xi)

    # --- todo junto ---------------------------------------------------------
    def predecir(self, datos, niveles=(90.0,)):
        """Recibe un DataFrame con `estacion`, las variables de CONT_D (o las temperaturas crudas)
        y el calendario (o `fecha`), y devuelve zeta, sigma y P(X > nivel) por fila."""
        d = self.completar_variables(datos)
        faltan = [v for v in CONT_D + CAL_D + ["estacion"] if v not in d or d[v].isna().any()]
        if faltan:
            raise ValueError(f"Faltan valores en: {faltan}")
        d, recortes = self.acotar(d)
        z = self.zeta(d)
        s = self.sigma(z)
        salida = pd.DataFrame({"estacion": d["estacion"].to_numpy(), "zeta": z, "sigma": s})
        for nivel in niveles:
            salida[f"p{nivel:g}"] = self.prob_superar(nivel, z, s)
        salida["alerta"] = salida[f"p{self.c:g}"] >= self.tau if self.c in niveles else np.nan
        salida["recortadas"] = recortes
        return salida

    # --- curvas para graficar ----------------------------------------------
    def curvas(self, estacion=None, puntos=100):
        """Efecto de cada variable con las demás en su mediana de entrenamiento.

        - `efecto_logit`: f_v(x) centrada (igual para todas las estaciones: el GAM es uno solo).
        - `probabilidad`: P(máximo de mañana > u) para un día típico de `estacion`.
        """
        ref = self.gam["referencia_curvas"]
        resultado = {}
        for v in CONT_D:
            lo, hi = ref["malla"][v]
            malla = np.linspace(lo, hi, puntos)
            efecto = self.curva(v, malla)
            fila = {"variable": v, "x": malla.round(4).tolist(),
                    "efecto_logit": (efecto - efecto.mean()).round(5).tolist()}
            if estacion is not None:
                d = pd.DataFrame({w: np.full(puntos, ref["mediana"][w]) for w in CONT_D})
                d[v] = malla
                for c in CAL_D:
                    d[c] = ref["media_calendario"][c]
                d["estacion"] = estacion
                fila["probabilidad"] = self.zeta(d).round(5).tolist()
            resultado[v] = fila
        return resultado
