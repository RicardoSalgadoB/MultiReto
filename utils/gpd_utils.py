"""
Distribución de Pareto generalizada (GPD): declustering, índice extremal,
verosimilitud con covariables y modelo de dos partes.

    P(X_{t+h} > c | info en t) = zeta(t) * P(Exceso > c - u | covariables en t)
        zeta  : logística (logit_utils)
        exceso: GPD con log(sigma) = Z @ beta  y  xi constante
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

from .data_preparation import STANDARD_C, THRESHOLD_U, VARIABLE, calendar_features
from .logit_utils import fit_logit_all_stations, predict_logit

# ----------------------------------------------------------------------------
# Declustering e índice extremal
# ----------------------------------------------------------------------------
def decluster_runs(x, u, r):
    """
    Método de rachas. x: Series con índice regular (puede tener NaN); u: umbral;
    r: dos excesos separados por MÁS de r pasos pertenecen a episodios distintos.
    Regresa un renglón por episodio: start, end, size, peak, peak_time.
    """
    values = x.to_numpy(dtype=float)
    positions = np.flatnonzero(values > u)           # NaN > u es False
    if len(positions) == 0:
        return pd.DataFrame(columns=["start", "end", "size", "peak", "peak_time"])

    cluster_id = np.cumsum(np.r_[True, np.diff(positions) > r]) - 1
    df = pd.DataFrame({"cluster": cluster_id, "value": values[positions],
                       "time": x.index[positions]})
    groups = df.groupby("cluster")
    peak_rows = groups["value"].idxmax().to_numpy()
    return pd.DataFrame({
        "start": groups["time"].first().to_numpy(),
        "end": groups["time"].last().to_numpy(),
        "size": groups.size().to_numpy(),
        "peak": df.loc[peak_rows, "value"].to_numpy(),
        "peak_time": df.loc[peak_rows, "time"].to_numpy(),
    })


def extremal_index_runs(x, u, r):
    """Estimador de rachas: (# episodios) / (# excesos)."""
    n_exceedances = int((x > u).sum())
    return len(decluster_runs(x, u, r)) / n_exceedances if n_exceedances else np.nan


def extremal_index_intervals(x, u):
    """Estimador de intervalos de Ferro y Segers (2003); no requiere elegir r."""
    positions = np.flatnonzero((x > u).to_numpy())
    n = len(positions)
    if n < 3:
        return np.nan
    t = np.diff(positions)
    if t.max() <= 2:
        theta = 2 * t.sum() ** 2 / ((n - 1) * (t ** 2).sum())
    else:
        theta = 2 * (t - 1).sum() ** 2 / ((n - 1) * ((t - 1) * (t - 2)).sum())
    return min(1.0, theta)


def summarize_episodes(frames, u=THRESHOLD_U, r=24, train_end=None, variable=VARIABLE,
                       unit="horas"):
    """Tabla por estación: excesos, episodios, theta (rachas e intervalos) y duración."""
    rows = []
    for station, s in frames.items():
        x = s[variable] if train_end is None else s.loc[s.index < pd.Timestamp(train_end), variable]
        episodes = decluster_runs(x, u, r)
        if len(episodes) == 0:
            continue
        step = x.index[1] - x.index[0]                  # un paso de la serie (1 h o 1 día)
        duration = (pd.to_datetime(episodes["end"]) - pd.to_datetime(episodes["start"])) / step + 1
        n_exceedances = int((x > u).sum())
        rows.append({"station": station, "exceedances": n_exceedances,
                     "episodes": len(episodes),
                     "theta_runs": len(episodes) / n_exceedances,
                     "theta_intervals": extremal_index_intervals(x, u),
                     "exceedances per episode": episodes["size"].mean(),
                     f"mean duration ({unit})": duration.mean(),
                     f"max duration ({unit})": duration.max()})
    return pd.DataFrame(rows).set_index("station")


# ----------------------------------------------------------------------------
# Datos de excesos
# ----------------------------------------------------------------------------
def build_peak_data(frame, u, r, covariates, train_end=None, variable=VARIABLE):
    """
    Picos de episodios (declustering) con las covariables medidas EN EL PICO.
    Sirve para describir qué hace más altos los picos (parte 1A).
    """
    episodes = decluster_runs(frame[variable], u, r)
    peak_times = pd.DatetimeIndex(episodes["peak_time"])
    base_cols = [c for c in covariates if c in frame.columns]
    peaks = frame.loc[peak_times, base_cols].copy()

    hours = frame.loc[peak_times, "peak_hour"].to_numpy() if "peak_hour" in frame.columns else None
    for name, values in calendar_features(peak_times, hours).items():
        if name in covariates:
            peaks[name] = values
    peaks["excess"] = episodes["peak"].to_numpy() - u
    peaks["cluster_size"] = episodes["size"].to_numpy()
    cutoff = pd.Timestamp(train_end) if train_end is not None else None
    peaks["period"] = np.where(peaks.index < cutoff, "train", "test") if cutoff is not None else "train"
    peaks["target_date"] = peaks.index.normalize()
    return peaks.dropna(subset=list(covariates) + ["excess"])


def build_peak_data_all_stations(frames, u, r, covariates, train_end=None):
    return {st: build_peak_data(f, u, r, covariates, train_end) for st, f in frames.items()}


def build_exceedance_data(table, u=THRESHOLD_U):
    """
    Excesos alineados con el pronóstico: filas donde x_target > u, con las covariables
    medidas en t (no declusterizado: se condiciona a lo que se sabe en t).
    """
    exceedances = table[table["event"] == 1].copy()
    exceedances["excess"] = exceedances["x_target"] - u
    return exceedances


def add_zeta(table, logit_model, clip=1e-6):
    """Agrega zeta (probabilidad de la logística) y logit_zeta como columnas."""
    out = table.copy()
    zeta = predict_logit(logit_model, out).clip(clip, 1 - clip)
    out["zeta"] = zeta
    out["logit_zeta"] = np.log(zeta / (1 - zeta))
    return out


# ----------------------------------------------------------------------------
# Verosimilitud y ajuste
# ----------------------------------------------------------------------------
def log1p_over_xi(xi, z):
    """(1/xi) * log(1 + xi*z), con el límite z cuando xi -> 0."""
    if abs(xi) < 1e-8:
        return z
    return np.log1p(xi * z) / xi


def gpd_nll(params, y, Z):
    """-log verosimilitud. params = [beta_0..beta_p, xi]; Z incluye la columna de unos."""
    beta, xi = params[:-1], params[-1]
    if xi <= -1:
        return np.inf
    sigma = np.exp(Z @ beta)
    z = y / sigma
    if np.any(1 + xi * z <= 0):                  # exceso fuera del soporte
        return np.inf
    return np.sum(np.log(sigma) + (1 + xi) * log1p_over_xi(xi, z))


def design_matrix(data, covariates, mean=None, std=None):
    """Columna de unos + covariables estandarizadas. Regresa (Z, mean, std)."""
    if len(covariates) == 0:
        return np.ones((len(data), 1)), None, None
    if mean is None:
        mean, std = data[covariates].mean(), data[covariates].std()
    z = ((data[covariates] - mean) / std).to_numpy(dtype=float)
    return np.column_stack([np.ones(len(data)), z]), mean, std


def _minimize(f, x0):
    """Nelder-Mead para acercarse y BFGS para afinar."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first = minimize(f, x0, method="Nelder-Mead",
                         options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-9})
        second = minimize(f, first.x, method="BFGS", options={"gtol": 1e-6})
    if np.isfinite(second.fun) and second.fun <= first.fun:
        return second
    return first


def fit_gpd(data, covariates, response="excess", warm_start=None, min_obs=10):
    """
    GPD con log(sigma) lineal en las covariables y xi constante.

    warm_start: modelo más chico cuyos parámetros sirven de punto de partida; así el
    modelo grande no queda con peor verosimilitud que el chico. Se prueban además
    arranques con xi = 0.1 y xi = -0.1 y se conserva el mejor.
    """
    data = data.dropna(subset=list(covariates) + [response])
    covariates = [c for c in covariates if data[c].std() > 1e-9]
    k = len(covariates) + 2
    if len(data) < max(min_obs, 3 * k):
        raise ValueError(f"Muy pocos excesos ({len(data)}) para {k} parámetros.")

    y = data[response].to_numpy(dtype=float)
    Z, mean, std = design_matrix(data, covariates)
    f = lambda p: gpd_nll(p, y, Z)

    starts = []
    if warm_start is not None:
        prev = dict(zip(warm_start["names"], warm_start["params"]))
        starts.append(np.array([prev.get(n, 0.0) for n in ["intercept"] + covariates] + [prev["xi"]]))
    for xi0 in (0.1, -0.1):
        starts.append(np.r_[np.log(y.mean() * (1 - xi0)), np.zeros(len(covariates)), xi0])

    best = None
    for x0 in starts:
        if not np.isfinite(f(x0)):
            continue
        res = _minimize(f, x0)
        if best is None or res.fun < best.fun:
            best = res
    if best is None:
        raise RuntimeError("No se encontró un punto inicial válido para la GPD.")

    return {"params": best.x, "nll": best.fun, "k": len(best.x), "n": len(y),
            "names": ["intercept"] + covariates + ["xi"], "covariates": covariates,
            "mean": mean, "std": std, "Z": Z, "y": y, "response": response,
            "index": data.index}


def fit_gpd_all_stations(data_by_station, covariates, warm_start_models=None):
    """
    Un modelo por estación. covariates: lista o diccionario estación -> lista.
    Si una estación tiene muy pocos excesos para el modelo pedido se usa el
    estacionario y se marca model['fallback'] = True.
    """
    models = {}
    for station, data in data_by_station.items():
        covs = covariates[station] if isinstance(covariates, dict) else covariates
        warm = warm_start_models[station] if warm_start_models else None
        try:
            model = fit_gpd(data, covs, warm_start=warm)
            model["fallback"] = False
        except ValueError:
            model = fit_gpd(data, [], warm_start=None)
            model["fallback"] = True
        models[station] = model
    return models


def fit_gpd_families(data_by_station, families, baseline="stationary"):
    """
    families: dict nombre -> lista de covariables (debe incluir `baseline` con []).
    El modelo estacionario ajustado primero sirve de arranque para los demás.
    Regresa dict nombre -> dict estación -> modelo.
    """
    fitted = {baseline: fit_gpd_all_stations(data_by_station, families[baseline])}
    for name, covs in families.items():
        if name != baseline:
            fitted[name] = fit_gpd_all_stations(data_by_station, covs, fitted[baseline])
    return {name: fitted[name] for name in families}


# ----------------------------------------------------------------------------
# Predicción, resumen y diagnósticos
# ----------------------------------------------------------------------------
def gpd_sigma(model, data):
    """Escala sigma para cada fila de `data` (usa media y desviación de entrenamiento)."""
    Z, _, _ = design_matrix(data, model["covariates"], model["mean"], model["std"])
    return np.exp(Z @ model["params"][:-1])


def gpd_survival(x, sigma, xi):
    """P(Exceso > x) para una GPD(sigma, xi); vale 0 fuera del soporte."""
    sigma = np.asarray(sigma, dtype=float)
    if abs(xi) < 1e-8:
        return np.exp(-x / sigma)
    t = 1 + xi * x / sigma
    out = np.zeros(sigma.shape)
    out[t > 0] = t[t > 0] ** (-1 / xi)
    return out


def gpd_aic(model):
    return 2 * model["nll"] + 2 * model["k"]


def gpd_nll_per_observation(model, data):
    """
    -log verosimilitud de cada exceso de `data` (por ejemplo, de prueba) con los
    parámetros ya ajustados. Es infinito si el exceso cae fuera del soporte.
    """
    y = data[model["response"]].to_numpy(dtype=float)
    sigma = gpd_sigma(model, data)
    xi = model["params"][-1]
    z = y / sigma
    out = np.full(len(y), np.inf)
    inside = 1 + xi * z > 0
    out[inside] = np.log(sigma[inside]) + (1 + xi) * log1p_over_xi(xi, z[inside])
    return out


def gpd_residuals(model, data=None):
    """
    Residuales que siguen una Exp(1) si el modelo es correcto. Con `data` se evalúan
    excesos nuevos; los que caen fuera del soporte regresan NaN.
    """
    xi = model["params"][-1]
    if data is None:
        y, sigma = model["y"], np.exp(model["Z"] @ model["params"][:-1])
    else:
        y, sigma = data[model["response"]].to_numpy(dtype=float), gpd_sigma(model, data)
    z = y / sigma
    out = np.full(len(y), np.nan)
    inside = 1 + xi * z > 0
    out[inside] = log1p_over_xi(xi, z[inside])
    return out


def numerical_hessian(f, x, eps=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps, eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej)
                                 - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps ** 2)
    return H


def gpd_summary(model):
    """Coeficientes, errores estándar (Hessiana numérica), z, p y exp(coef)."""
    f = lambda p: gpd_nll(p, model["y"], model["Z"])
    covariance = np.linalg.pinv(numerical_hessian(f, model["params"]))
    se = np.sqrt(np.clip(np.diag(covariance), 0, None))
    with np.errstate(all="ignore"):
        z = model["params"] / se
    table = pd.DataFrame({"coef": model["params"], "se": se, "z": z,
                          "p": 2 * norm.sf(np.abs(z))}, index=model["names"])
    table["exp(coef)"] = np.exp(table["coef"])
    table.loc["xi", "exp(coef)"] = np.nan             # solo tiene sentido para los beta
    return table


def excess_scenarios(model, data, u, c, quantile=0.95, low=0.1, high=0.9):
    """
    Escenarios: una covariable en su percentil bajo/alto y las demás en la mediana.
    Regresa sigma, el cuantil del pico (ppb) y P(pico > c).
    """
    xi = model["params"][-1]
    covs = [cv for cv in model["covariates"]]
    base = {cv: data[cv].median() for cv in covs}
    rows = [dict(base, scenario="todas en la mediana")]
    for cv in covs:
        for q in (low, high):
            row = dict(base, scenario=f"{cv} en el percentil {int(q * 100)}")
            row[cv] = data[cv].quantile(q)
            rows.append(row)
    table = pd.DataFrame(rows).set_index("scenario")
    sigma = gpd_sigma(model, table)
    table["sigma"] = sigma
    if abs(xi) < 1e-8:
        table[f"peak quantile {quantile:.0%} (ppb)"] = u - sigma * np.log(1 - quantile)
    else:
        table[f"peak quantile {quantile:.0%} (ppb)"] = u + sigma / xi * ((1 - quantile) ** (-xi) - 1)
    table[f"P(peak > {c:.0f})"] = gpd_survival(c - u, sigma, xi)
    return table


# ----------------------------------------------------------------------------
# Modelo de dos partes
# ----------------------------------------------------------------------------
def two_part_probability(table, gpd_model, c=STANDARD_C, u=THRESHOLD_U, zeta_column="zeta"):
    """P(X_{t+h} > c) = zeta * P(Exceso > c - u). La tabla debe traer zeta (add_zeta)."""
    if c <= u:
        raise ValueError("c debe ser mayor que u.")
    sigma = gpd_sigma(gpd_model, table)
    return table[zeta_column].to_numpy() * gpd_survival(c - u, sigma, gpd_model["params"][-1])


def fit_two_part_model(tables, logit_features, gpd_covariates, u=THRESHOLD_U, c=STANDARD_C,
                       cluster_col=None, output_column="p_two_part"):
    """
    Ajusta el modelo completo con las filas de entrenamiento:
      1) logística por estación -> zeta
      2) GPD sobre los excesos de entrenamiento (con logit_zeta si está en gpd_covariates)
      3) columna `output_column` = P(X > c) para TODAS las filas
    Regresa (tablas con zeta y la probabilidad, modelos logit, modelos GPD).
    """
    logit_models = fit_logit_all_stations(tables, logit_features, cluster_col=cluster_col)
    out = {st: add_zeta(t, logit_models[st]) for st, t in tables.items()}
    exceedances = {st: build_exceedance_data(t[t["period"] == "train"], u) for st, t in out.items()}
    gpd_models = fit_gpd_all_stations(exceedances, gpd_covariates)
    for st, t in out.items():
        t[output_column] = two_part_probability(t, gpd_models[st], c, u)
    return out, logit_models, gpd_models


# ----------------------------------------------------------------------------
# Sensibilidad a r
# ----------------------------------------------------------------------------
def r_sensitivity(x, u, r_values):
    """Número de episodios, theta, xi y sigma de una GPD sin covariables para cada r."""
    n_exceedances = int((x > u).sum())
    rows = []
    for r in r_values:
        episodes = decluster_runs(x, u, r)
        row = {"r": r, "clusters": len(episodes),
               "theta": len(episodes) / n_exceedances if n_exceedances else np.nan,
               "xi": np.nan, "sigma": np.nan}
        try:
            model = fit_gpd(pd.DataFrame({"excess": episodes["peak"].to_numpy(dtype=float) - u}), [])
            row["xi"], row["sigma"] = model["params"][-1], np.exp(model["params"][0])
        except (ValueError, RuntimeError):
            pass
        rows.append(row)
    return pd.DataFrame(rows)


def r_sensitivity_all_stations(frames, u, r_values, train_end=None, variable=VARIABLE):
    out = {}
    for station, s in frames.items():
        x = s[variable] if train_end is None else s.loc[s.index < pd.Timestamp(train_end), variable]
        out[station] = r_sensitivity(x, u, r_values)
    return out


def plot_r_sensitivity(tables, r_values, chosen_r=None, unit="horas", title=None, path=None):
    """Cuatro paneles (episodios, theta, xi, sigma) con una línea por estación y la mediana."""
    columns = ["clusters", "theta", "xi", "sigma"]
    labels = ["Número de episodios", "Índice extremal θ", "Forma ξ", "Escala σ"]
    stacked = pd.concat(tables.values())
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, col, label in zip(axes.flat, columns, labels):
        for t in tables.values():
            ax.plot(t["r"], t[col], color="0.75", lw=1)
        median = stacked.groupby("r")[col].median()
        ax.plot(median.index, median.values, "o-", color="black", lw=2, label="mediana")
        if chosen_r is not None:
            ax.axvline(chosen_r, color="tab:red", ls="--", lw=1, label=f"r elegido = {chosen_r}")
        ax.set_xticks(list(r_values))
        ax.set_xlabel(f"r ({unit})")
        ax.set_title(label, fontsize=10)
    axes[0, 0].legend(fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
    return fig