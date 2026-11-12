"""Entrena el modelo final y exporta todo lo que necesita la API.

Reproduce el notebook 04e_gam_gpd (Parte 3) y guarda:
  models/gam.json         parámetros del GAM (nudos y coeficientes de cada spline, efectos lineales)
  models/gpd.json         parámetros de la GPD (b, xi, nudos del spline de logit zeta) y el corte tau
  models/metricas.json    BSS, AUC, aciertos y falsas alarmas en 2025 y 2026
  data/diario.csv.gz      tabla estación-día con variables y predicciones (Tabs 1 y 2)
  data/historico_o3.csv.gz  ozono horario por estación (Tab 1)
  data/dependencia.json   matrices chi(0.95), chi barra y Spearman (Tab 3)

Uso (desde la carpeta backend/):
    python -m source.entrenar
    python -m source.entrenar --datos RUTA/datos_no_imputados.csv --pronostico RUTA/pronostico_dia_anterior.csv
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.gam.api import BSplines, GLMGam

from .config import (C, CAL_D, CARPETA_DATOS, CARPETA_MODELOS, CLAVES, CONT_D, ESTACIONES,
                     RUTA_DATOS_DEFECTO, RUTA_PRONOSTICOS_DEFECTO, U, VAR)
from .datos import cargar_datos, construir_estaciones, filas_completas, leer_pronostico, tabla_diaria
from .predict import ModeloOzono, logit, prob_mayor_gpd, spline_restringido

ALPHAS = [10, 100, 1000, 3000, 10000, 30000, 100000]
ESTAC_D = ["est_" + st for st in CLAVES[1:]]      # la primera estación es la referencia
LIN_GAM = CAL_D + ESTAC_D


# ---------------------------------------------------------------------------
# GAM (igual que el notebook)
# ---------------------------------------------------------------------------
def agregar_dummies(DIA):
    for st in CLAVES[1:]:
        DIA["est_" + st] = (DIA["estacion"] == st).astype(float)
    return DIA


def ajustar_gam(datos, alpha, limites_de):
    limites = [{"lower_bound": limites_de[v].min(), "upper_bound": limites_de[v].max()} for v in CONT_D]
    bases = BSplines(datos[CONT_D].to_numpy(), df=[8] * len(CONT_D), degree=[3] * len(CONT_D),
                     include_intercept=False, knot_kwds=limites)
    X = sm.add_constant(datos[LIN_GAM])
    modelo = GLMGam(datos["y63"], exog=X, smoother=bases, alpha=[alpha] * len(CONT_D),
                    family=sm.families.Binomial())
    return modelo.fit()


def predecir_gam(res, datos):
    X = sm.add_constant(datos[LIN_GAM], has_constant="add")
    return np.asarray(res.predict(exog=X, exog_smooth=datos[CONT_D].to_numpy()))


# ---------------------------------------------------------------------------
# GPD (igual que el notebook)
# ---------------------------------------------------------------------------
def log1p_entre_xi(xi, z):
    if abs(xi) < 1e-8:
        return z
    return np.log1p(xi * z) / xi


def gpd_nll(parametros, y, Z):
    beta, xi = parametros[:-1], parametros[-1]
    if xi <= -1:
        return np.inf
    sigma = np.exp(Z @ beta)
    z = y / sigma
    if np.any(1 + xi * z <= 0):
        return np.inf
    return np.sum(np.log(sigma) + (1 + xi) * log1p_entre_xi(xi, z))


def matriz_diseño(datos, covs):
    if len(covs) == 0:
        return np.ones((len(datos), 1)), None, None
    media, desv = datos[covs].mean(), datos[covs].std()
    Zc = ((datos[covs] - media) / desv).to_numpy()
    return np.column_stack([np.ones(len(datos)), Zc]), media, desv


def minimizar(f, x0):
    res1 = minimize(f, x0, method="Nelder-Mead", options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
    res2 = minimize(f, res1.x, method="BFGS", options={"gtol": 1e-6})
    return res2 if np.isfinite(res2.fun) and res2.fun <= res1.fun else res1


def ajustar_gpd(datos, covs, inicio=None):
    y = datos["y"].to_numpy()
    Z, media, desv = matriz_diseño(datos, covs)
    f = lambda p: gpd_nll(p, y, Z)
    puntos = []
    if inicio is not None:
        anteriores = dict(zip(inicio["nombres"], inicio["params"]))
        puntos.append(np.array([anteriores.get(n, 0.0) for n in ["intercepto"] + covs] + [anteriores["xi"]]))
    for xi0 in [0.1, -0.1]:
        puntos.append(np.array([np.log(y.mean() * (1 - xi0))] + [0.0] * len(covs) + [xi0]))
    mejor = None
    for x0 in puntos:
        if np.isfinite(f(x0)):
            res = minimizar(f, x0)
            if mejor is None or res.fun < mejor.fun:
                mejor = res
    return {"params": mejor.x, "nll": mejor.fun, "k": len(mejor.x), "nombres": ["intercepto"] + covs + ["xi"],
            "covs": covs, "media": media, "desv": desv}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def metricas(y, p, p_ref):
    y, p, p_ref = (np.asarray(a, dtype=float) for a in (y, p, p_ref))
    brier = brier_score_loss(y, p)
    return {"n": int(len(y)), "eventos": int(y.sum()), "auc": float(roc_auc_score(y, p)),
            "brier": float(brier), "bss": float(1 - brier / brier_score_loss(y, p_ref))}


def alertas(y, p, tau):
    y = np.asarray(y, dtype=bool)
    alerta = np.asarray(p) >= tau
    return {"aciertos": float((alerta & y).sum() / y.sum()),
            "falsas_alarmas": float((alerta & ~y).sum() / max(alerta.sum(), 1)),
            "alertas": int(alerta.sum())}


def corte_70(y, p):
    """El corte más alto que detecta al menos 70 % de los días con exceso."""
    for tau in np.sort(np.unique(p))[::-1]:
        if alertas(y, p, tau)["aciertos"] >= 0.7:
            return float(tau)
    return float(np.min(p))


# ---------------------------------------------------------------------------
# Dependencia de colas (igual que el notebook 04b)
# ---------------------------------------------------------------------------
def chi_y_chibar(u1, u2, q):
    ok = ~(np.isnan(u1) | np.isnan(u2))
    ambos = (u1[ok] > q) & (u2[ok] > q)
    p_ambos = ambos.mean()
    chi = p_ambos / (1 - q)
    chibar = 2 * np.log(1 - q) / np.log(p_ambos) - 1 if p_ambos > 0 else np.nan
    return chi, chibar, int(ambos.sum())


def distancia_km(a, b):
    lat1, lon1 = np.radians(ESTACIONES[a][1:])
    lat2, lon2 = np.radians(ESTACIONES[b][1:])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * 6371 * np.arcsin(np.sqrt(h)))


def dependencia(E, q=0.95):
    horario = pd.DataFrame({st: E[st][VAR] for st in CLAVES})
    X = horario.resample("D").max().where(horario.resample("D").count() >= 18)
    X = X[X.index < "2026-01-01"]
    Uq = X.rank() / (X.notna().sum() + 1)
    n = len(CLAVES)
    chi, chibar, conj = np.eye(n), np.eye(n), np.zeros((n, n), dtype=int)
    for i, a in enumerate(CLAVES):
        for j in range(i + 1, n):
            c, cb, nc = chi_y_chibar(Uq[a].to_numpy(), Uq[CLAVES[j]].to_numpy(), q)
            chi[i, j] = chi[j, i] = c
            chibar[i, j] = chibar[j, i] = cb
            conj[i, j] = conj[j, i] = nc
    spearman = X.corr(method="spearman").loc[CLAVES, CLAVES].to_numpy()
    redondear = lambda M: np.round(M, 4).tolist()
    return {
        "estaciones": CLAVES, "q": q, "periodo": "2023-01-01 a 2025-12-31 (máximos diarios con >= 18 h)",
        "chi": redondear(chi), "chibar": redondear(chibar), "spearman": redondear(spearman),
        "dias_conjuntos": conj.tolist(),
        "distancia_km": [[round(distancia_km(a, b), 2) for b in CLAVES] for a in CLAVES],
    }


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datos", default=str(RUTA_DATOS_DEFECTO))
    parser.add_argument("--pronostico", default=str(RUTA_PRONOSTICOS_DEFECTO))
    args = parser.parse_args()
    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_DATOS.mkdir(exist_ok=True)

    print("1/6 Cargando datos...")
    E = construir_estaciones(cargar_datos(args.datos))
    pron = leer_pronostico(args.pronostico)
    DIA_todo = agregar_dummies(tabla_diaria(E, pron))
    DIA = filas_completas(DIA_todo, CONT_D)
    DIA["y63"] = (DIA["max_manana"] > U).astype(float)
    DIA["y90"] = (DIA["max_manana"] > C).astype(float)
    ent, val = DIA[DIA["periodo"] == "entrenamiento"], DIA[DIA["periodo"] == "validación"]
    print("   Filas estación-día:", DIA.groupby("periodo").size().to_dict())

    print("2/6 Eligiendo la penalización del GAM con 2025...")
    clima63 = ent.groupby("estacion")["y63"].mean()
    eleccion = {}
    for alpha in ALPHAS:
        res = ajustar_gam(ent, alpha, DIA)
        eleccion[alpha] = metricas(val["y63"], predecir_gam(res, val), val["estacion"].map(clima63))["bss"]
        print(f"   alpha = {alpha:>6}: BSS zeta 2025 = {eleccion[alpha]:.4f}")
    ALPHA = max(eleccion, key=eleccion.get)
    m_gam = ajustar_gam(ent, ALPHA, DIA)
    print(f"   alpha elegido = {ALPHA} (grados de libertad efectivos = {m_gam.edf.sum():.1f})")

    # --- exportar el GAM ---
    params = np.asarray(m_gam.params)
    nombres_lin = ["const"] + LIN_GAM
    coef_lin = dict(zip(nombres_lin, params[:len(nombres_lin)].tolist()))
    splines, pos = {}, len(nombres_lin)
    for v, sm_v in zip(CONT_D, m_gam.model.smoother.smoothers):
        k = sm_v.dim_basis
        splines[v] = {"nudos": np.asarray(sm_v.knots).tolist(), "grado": int(sm_v.degree),
                      "coef": params[pos:pos + k].tolist(),
                      "limites": [float(DIA[v].min()), float(DIA[v].max())]}
        pos += k
    assert pos == len(params), "El número de coeficientes no cuadra"
    efecto_estacion = {CLAVES[0]: 0.0}
    efecto_estacion.update({st: coef_lin.pop("est_" + st) for st in CLAVES[1:]})
    gam_json = {
        "descripcion": "GAM con respuesta binaria: logit P(máximo de mañana > u) = const + calendario "
                       "+ efecto de la estación + suma de B-splines cúbicos penalizados",
        "u": U, "alpha": ALPHA, "gl_efectivos": float(m_gam.edf.sum()),
        "estaciones": CLAVES, "referencia": CLAVES[0], "variables": CONT_D, "calendario": CAL_D,
        "coef_lineales": coef_lin, "efecto_estacion": efecto_estacion, "splines": splines,
        "referencia_curvas": {
            "malla": {v: [float(ent[v].quantile(0.01)), float(ent[v].quantile(0.99))] for v in CONT_D},
            "mediana": {v: float(ent[v].median()) for v in CONT_D},
            "media_calendario": {c: float(ent[c].mean()) for c in CAL_D},
        },
        "eleccion_alpha": {str(a): round(b, 5) for a, b in eleccion.items()},
    }

    print("3/6 Ajustando la GPD de los excesos...")
    DIA["zeta"] = np.clip(predecir_gam(m_gam, DIA), 1e-6, 1 - 1e-6)
    DIA["logit_zeta"] = logit(DIA["zeta"])
    exc = DIA[(DIA["periodo"] == "entrenamiento") & (DIA["y63"] == 1)].copy()
    exc["y"] = exc["max_manana"] - U
    nudos_lz = np.quantile(exc["logit_zeta"], [0.05, 0.35, 0.65, 0.95])
    B = spline_restringido(exc["logit_zeta"], nudos_lz)
    covs = ["logit_zeta"] + [f"lz{j + 1}" for j in range(B.shape[1])]
    for j in range(B.shape[1]):
        exc[f"lz{j + 1}"] = B[:, j]
    g0 = ajustar_gpd(exc, [])
    recta = ajustar_gpd(exc, ["logit_zeta"], inicio=g0)
    g = ajustar_gpd(exc, covs, inicio=recta)
    print(f"   {len(exc)} excesos | xi = {g['params'][-1]:.4f} | "
          f"D(logit zeta) = {2 * (g0['nll'] - recta['nll']):.1f} | D(spline) = {2 * (recta['nll'] - g['nll']):.1f}")
    gpd_json = {
        "descripcion": "GPD del exceso del máximo diario sobre u: log(sigma) = b0 + b·z, con z las covariables "
                       "[logit zeta, spline restringido de logit zeta] estandarizadas con media y desv; xi común",
        "u": U, "c": C, "xi": float(g["params"][-1]), "b": g["params"][:-1].tolist(),
        "covariables": covs, "media": g["media"].tolist(), "desv": g["desv"].tolist(),
        "nudos_logit_zeta": nudos_lz.tolist(), "n_excesos": int(len(exc)), "tau": None,
    }

    # El corte de alerta se fija en 2025 (70 % de los días con exceso detectados)
    xi = gpd_json["xi"]
    Z = (np.column_stack([DIA["logit_zeta"], spline_restringido(DIA["logit_zeta"], nudos_lz)])
         - np.asarray(gpd_json["media"])) / np.asarray(gpd_json["desv"])
    sigma = np.exp(gpd_json["b"][0] + Z @ np.asarray(gpd_json["b"][1:]))
    DIA["p90"] = DIA["zeta"] * prob_mayor_gpd(C - U, sigma, xi)
    v25 = DIA[DIA["periodo"] == "validación"]
    gpd_json["tau"] = corte_70(v25["y90"], v25["p90"])
    print(f"   corte de alerta tau = {gpd_json['tau']:.4f}")

    (CARPETA_MODELOS / "gam.json").write_text(json.dumps(gam_json, ensure_ascii=False, indent=1), encoding="utf-8")
    (CARPETA_MODELOS / "gpd.json").write_text(json.dumps(gpd_json, ensure_ascii=False, indent=1), encoding="utf-8")

    print("4/6 Verificando que predict.py reproduzca a statsmodels...")
    modelo = ModeloOzono()
    mio = modelo.predecir(DIA, niveles=(C,))
    dif_zeta = float(np.max(np.abs(mio["zeta"].to_numpy() - predecir_gam(m_gam, DIA))))
    dif_p90 = float(np.max(np.abs(mio["p90"].to_numpy() - DIA["p90"].to_numpy())))
    print(f"   diferencia máxima: zeta = {dif_zeta:.2e}, P(>90) = {dif_p90:.2e}")
    assert dif_zeta < 1e-8 and dif_p90 < 1e-8, "predict.py no coincide con statsmodels"

    print("5/6 Calculando métricas...")
    clima90 = ent.groupby("estacion")["y90"].mean()
    tabla = {}
    for per, anio in [("validación", "2025"), ("prueba", "2026")]:
        t = DIA[DIA["periodo"] == per]
        tabla[anio] = {
            "zeta": metricas(t["y63"], t["zeta"], t["estacion"].map(clima63)),
            "p90": {**metricas(t["y90"], t["p90"], t["estacion"].map(clima90)),
                    **alertas(t["y90"], t["p90"], gpd_json["tau"])},
            "climatologia_p90": alertas(t["y90"], t["estacion"].map(clima90),
                                        corte_70(v25["y90"], v25["estacion"].map(clima90))),
        }
        print(f"   {anio}: BSS P(>90) = {tabla[anio]['p90']['bss']:.3f}, AUC = {tabla[anio]['p90']['auc']:.3f}, "
              f"aciertos = {tabla[anio]['p90']['aciertos']:.1%}, falsas alarmas = {tabla[anio]['p90']['falsas_alarmas']:.1%}")
    metricas_json = {"generado": datetime.now().isoformat(timespec="seconds"), "alpha": ALPHA,
                     "tau": gpd_json["tau"], "por_anio": tabla,
                     "nota": "Climatología = frecuencia de excedencia de entrenamiento por estación."}
    (CARPETA_MODELOS / "metricas.json").write_text(json.dumps(metricas_json, ensure_ascii=False, indent=1),
                                                   encoding="utf-8")

    print("6/6 Exportando datos para la API...")
    # Tabla diaria con TODAS las filas (también las que no tienen mañana), con predicción cuando se puede
    completas = DIA_todo.dropna(subset=["o3max", "o3media", "horas63", "horas63_7d", "o3max_ciudad",
                                        "SR_max_hoy", "RH_media_hoy", "WSR_media_hoy", "T_max_hoy",
                                        "T_max_pron", "T_min_pron"])
    pred = modelo.predecir(completas, niveles=(C,))
    DIA_todo["zeta"] = np.nan
    DIA_todo["p90"] = np.nan
    DIA_todo.loc[completas.index, "zeta"] = pred["zeta"].to_numpy()
    DIA_todo.loc[completas.index, "p90"] = pred["p90"].to_numpy()
    DIA_todo["alerta"] = (DIA_todo["p90"] >= gpd_json["tau"]).where(DIA_todo["p90"].notna())
    columnas = ["estacion", "hoy", "fecha", "periodo", "o3max", "o3media", "horas63", "horas63_7d",
                "o3max_ciudad", "SR_max_hoy", "RH_media_hoy", "WSR_media_hoy", "T_max_hoy", "T_max_pron",
                "T_min_pron", "max_manana", "zeta", "p90", "alerta"]
    salida = DIA_todo[columnas].copy()
    numericas = salida.select_dtypes("number").columns
    salida[numericas] = salida[numericas].round(5)
    salida.to_csv(CARPETA_DATOS / "diario.csv.gz", index=False)

    horario = pd.DataFrame({st: E[st][VAR] for st in CLAVES})
    horario.index.name = "fecha_hora"
    horario.to_csv(CARPETA_DATOS / "historico_o3.csv.gz")

    (CARPETA_DATOS / "dependencia.json").write_text(json.dumps(dependencia(E)), encoding="utf-8")
    print("Listo. Archivos en", CARPETA_MODELOS, "y", CARPETA_DATOS)


if __name__ == "__main__":
    main()
