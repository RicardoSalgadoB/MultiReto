import numpy as np
import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass
from scipy.stats import chi2
from statsmodels.gam.api import GLMGam, BSplines
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn import metrics

# U - THRESHOLD
U = 63.0

R = 5

C = 90.0

VARIABLE = "O3"

def load_data(filename):
    # Load data
    data = pd.read_csv(filename)

    # Select relevant columns
    data = data[["Fecha y hora", "Estacion", VARIABLE, "PRS", "RAINF", "RH", "SR", "TOUT", "WSR", "WDR"]]
    
    # Convert dates to datetime
    data["Fecha y hora"] = pd.to_datetime(data["Fecha y hora"])
    
    # Filter data
    data = data[(data["Fecha y hora"] >= "2023-01-01 00:00:00") & (data["Fecha y hora"] < "2026-01-01 00:00:00")]
    
    # Reset index
    data.set_index("Fecha y hora", inplace=True)
    
    return data


def group_data_by_station(data):
    stations_names = list(data["Estacion"].unique())
    stations_data = {}
    for station in stations_names:
        d = data[data["Estacion"]==station].drop(columns="Estacion")
        d_var = d.dropna(subset=[VARIABLE])
        daily_max_idx = d_var.groupby(d_var.index.date)[VARIABLE].idxmax()
        stations_data[station] = d.loc[daily_max_idx]
    return stations_data, stations_names


def feature_engineering(station_data, station_names):
    for station in station_names:
        d = station_data[station]
        
        # Wind feat. eng.
        d["wind_u"] = -d["WSR"] * np.sin(np.deg2rad(d["WDR"]-45))   # Restar 45° para que los componentes sean NO y SO
        d["wind_v"] = -d["WSR"] * np.cos(np.deg2rad(d["WDR"]-45))
        
        # Hour feat. eng.
        d["hour_sin"] = np.sin(2 * np.pi * d.index.hour / 24)
        d["hour_cos"] = np.cos(2 * np.pi * d.index.hour / 24)
        
        # Month feat. eng.
        d["month_sin"] = np.sin(2 * np.pi * d.index.month / 12)
        d["month_cos"] = np.cos(2 * np.pi * d.index.month / 12)
        
        station_data[station] = d
        
    return station_data


def split_data(station_data, station_names):
    train_data = {}
    test_data = {}
    for station in station_names:
        data = station_data[station]
        train = data[(data.index < "2025-01-01 00:00:00")]
        test = data[(data.index >= "2025-01-01 00:00:00")]
        train_data[station] = train
        test_data[station] = test
    return train_data, test_data


def build_logit_data(x, met, u, horizon=1):
    d = pd.DataFrame(index=x.index)
    d["y"] = (x.shift(-horizon) > u).astype(float)
    d.loc[x.shift(-horizon).isna(), "y"] = np.nan

    # --- persistence-type features (the strongest predictors) ---
    # d["x_t"] = np.log1p(x)                                   # current level
    d["x_t"] = x                                   # current level
    d["exc_t"] = (x > u).astype(float)             # exceeded now?
    d["x_lag1"] = x.shift(1)
    d["x_lag24"] = x.shift(24)
    # d["max24"] = np.log1p(np.log1p(x.rolling(24).max()))            # recent peak
    d["max24"] = x.rolling(24).max()            # recent peak
    # d["n_exc7"] = np.log1p(np.log1p((x > u).astype(float).rolling(7).sum()))   # recent episode activity
    d["n_exc7"] = (x > u).astype(float).rolling(7).sum()   # recent episode activity

    # --- meteorology ---
    for c in ["WSR", "RH", "PRS", "SR"]:
        d[c] = met[c]
    d["TOUT"] = met["TOUT"]
    d["wind_u"] = met["wind_u"]
    d["wind_v"] = met["wind_v"]
    
    # --- temporal ---
    d["hour_sin"], d["hour_cos"] = met["hour_sin"], met["hour_cos"]
    d["month_sin"], d["month_cos"] = met["month_sin"], met["month_cos"]
    return d.dropna()


def build_logit_station_data(station_data, station_names):
    logit_station_data = {}
    for station in station_names:
        x = station_data[station][VARIABLE]
        met = station_data[station].drop(columns=[VARIABLE])
        logit_station_data[station] = build_logit_data(x, met, U)
    return logit_station_data


# =============================================================================
# GAM (Generalized Additive Model) for the exceedance probability  P(X_{t+h} > u)
#
#   logit P(y=1) = b0 + sum_j b_j * z_j  +  sum_k f_k(x_k)
#
# * linear terms  : features in LINEAR_FEATS (binary / count / cyclic harmonics)
#                   -> standardised and entered as ordinary parametric terms
# * smooth terms  : all other features -> penalised cubic B-splines
#                   (statsmodels.gam.GLMGam, Binomial family, logit link)
# * the penalty weight (alpha) is chosen per model by minimising the AIC
#   computed with the *effective* degrees of freedom (edf)
# =============================================================================

# Features that enter linearly (everything else listed in `feats` is smoothed)
LINEAR_FEATS = ("exc_t", "n_exc7", "hour_sin", "hour_cos", "month_sin", "month_cos")
GAM_DF = 6                                  # basis dimension of every smooth (5 after centring)
GAM_DEGREE = 3                              # cubic B-splines
ALPHA_GRID = tuple(np.logspace(-1, 4, 6))   # candidate penalty weights


@dataclass
class GAMFit:
    """Everything needed to evaluate / predict / validate one fitted GAM."""
    res: object                 # statsmodels GLMGamResults (or GLMResults if no smooth term)
    smoother: object            # fitted BSplines object (None if no smooth term)
    feats: list
    smooth_feats: list
    linear_feats: list
    mu: pd.Series               # mean of the linear features (training)
    sd: pd.Series               # std  of the linear features (training)
    lo: pd.Series               # training min of the smooth features (for clipping)
    hi: pd.Series               # training max of the smooth features (for clipping)
    alpha: float                # penalty weight used
    df_basis: int
    degree: int
    n: int
    ll_null: float              # log-likelihood of the intercept-only model

    # ---- quantities used by the LR tests / AIC -------------------------------
    @property
    def llf(self):
        """Un-penalised binomial log-likelihood."""
        return float(self.res.llf)

    @property
    def edf(self):
        """Total effective degrees of freedom (intercept included)."""
        if hasattr(self.res, "edf"):
            return float(np.sum(self.res.edf))
        return float(len(self.res.params))

    @property
    def df_model(self):
        """Effective degrees of freedom excluding the intercept."""
        return self.edf - 1.0

    @property
    def aic(self):
        return -2.0 * self.llf + 2.0 * self.edf

    @property
    def llr(self):
        """LR statistic against the intercept-only model."""
        return max(2.0 * (self.llf - self.ll_null), 0.0)

    @property
    def llr_pvalue(self):
        return float(chi2.sf(self.llr, max(self.df_model, 1e-8)))

    # ---- design matrices -----------------------------------------------------
    def design(self, df):
        """Returns (parametric exog DataFrame, smooth exog ndarray or None)."""
        const = pd.Series(1.0, index=df.index, name="const")
        if self.linear_feats:
            lin = (df[self.linear_feats] - self.mu) / self.sd
            exog = pd.concat([const, lin], axis=1)
        else:
            exog = const.to_frame()
        if self.smooth_feats:
            # B-splines cannot be evaluated outside the training knots -> clip
            xs = df[self.smooth_feats].clip(lower=self.lo, upper=self.hi, axis=1)
            return exog, xs.to_numpy()
        return exog, None

    def predict(self, df):
        """P(y=1) for every row of df (NaN where a needed feature is missing)."""
        out = pd.Series(np.nan, index=df.index, dtype=float)
        ok = df[self.feats].notna().all(axis=1)
        if not ok.any():
            return out
        exog, xs = self.design(df.loc[ok])
        if xs is None:
            out.loc[ok] = np.asarray(self.res.predict(exog))
        else:
            out.loc[ok] = np.asarray(self.res.predict(exog=exog, exog_smooth=xs))
        return out


def _split_features(train, feats, df, linear_feats):
    """Smooth every feature not forced to be linear (and with enough distinct values)."""
    smooth = [f for f in feats
              if f not in linear_feats and train[f].nunique() > df + 1]
    linear = [f for f in feats if f not in smooth]
    return smooth, linear


def _fit_once(y, exog, xs, smooth_feats, df, degree, alpha):
    if not smooth_feats:
        return sm.GLM(y, exog, family=sm.families.Binomial()).fit(), None
    smoother = BSplines(xs, df=[df] * len(smooth_feats), degree=[degree] * len(smooth_feats),
                        variable_names=smooth_feats)
    model = GLMGam(y, exog=exog, smoother=smoother, alpha=[alpha] * len(smooth_feats),
                   family=sm.families.Binomial())
    return model.fit(), smoother


def fit_gam(train, feats, df=GAM_DF, degree=GAM_DEGREE, alpha=None,
            alpha_grid=ALPHA_GRID, linear_feats=LINEAR_FEATS):
    """
    Fit a binomial GAM for train["y"].

    alpha=None  -> choose the penalty weight over `alpha_grid` by minimum AIC (edf based)
    alpha=float -> use that penalty weight (handy for like-for-like nested comparisons)
    """
    feats = list(feats)
    y = train["y"].to_numpy()
    smooth_feats, lin_feats = _split_features(train, feats, df, tuple(linear_feats))

    mu = train[lin_feats].mean() if lin_feats else pd.Series(dtype=float)
    sd = train[lin_feats].std() if lin_feats else pd.Series(dtype=float)
    lo = train[smooth_feats].min() if smooth_feats else pd.Series(dtype=float)
    hi = train[smooth_feats].max() if smooth_feats else pd.Series(dtype=float)

    p = y.mean()
    ll_null = len(y) * (p * np.log(p) + (1 - p) * np.log(1 - p)) if 0 < p < 1 else 0.0

    proto = GAMFit(res=None, smoother=None, feats=feats, smooth_feats=smooth_feats,
                   linear_feats=lin_feats, mu=mu, sd=sd, lo=lo, hi=hi, alpha=np.nan,
                   df_basis=df, degree=degree, n=len(y), ll_null=ll_null)
    exog, xs = proto.design(train)

    if not smooth_feats:
        candidates = [np.nan]
    elif alpha is not None:
        candidates = [alpha]
    else:
        candidates = list(alpha_grid)

    best = None
    for a in candidates:
        try:
            res, smoother = _fit_once(y, exog, xs, smooth_feats, df, degree, a)
        except Exception:                    # e.g. (quasi-)separation for a tiny penalty
            continue
        fit = GAMFit(**{**proto.__dict__, "res": res, "smoother": smoother, "alpha": a})
        if not np.all(np.isfinite([fit.llf, fit.edf, fit.aic])):
            continue                         # IRLS diverged (near-separation) -> discard
        if best is None or fit.aic < best.aic:
            best = fit
    if best is None:
        raise RuntimeError("GAM could not be fitted for any penalty weight")
    return best


def predict_gam(model, df):
    return model.predict(df)


def fit_gam_all_stations(logit_station_data, station_names, features, **gam_kwargs):
    """`features` can be a list (same for all stations) or a dict {station: list}."""
    models = {}
    for station in station_names:
        feats = features[station] if isinstance(features, dict) else features
        models[station] = fit_gam(logit_station_data[station], feats, **gam_kwargs)
    return models


def runs_decluster(x, u, r):
    """
    x : pd.Series with a regular DatetimeIndex (NaNs allowed)
    u : threshold
    r : run length. Exceedances separated by >= r non-exceedances
        start a new cluster.
    Returns a DataFrame with one row per cluster.
    """
    x = x.copy()
    pos = np.flatnonzero((x > u).to_numpy())        # positions of exceedances
    if len(pos) == 0:
        return pd.DataFrame(columns=["start", "end", "size", "peak", "peak_time"])

    gaps = np.diff(pos)
    new_cluster = np.r_[True, gaps > r]             # gap-1 >= r  <=>  gap > r
    cluster_id = np.cumsum(new_cluster) - 1

    df = pd.DataFrame({
        "pos": pos,
        "time": x.index[pos],
        "value": x.to_numpy()[pos],
        "cluster": cluster_id,
    })

    out = df.groupby("cluster").apply(lambda g: pd.Series({
        "start": g["time"].iloc[0],
        "end": g["time"].iloc[-1],
        "size": len(g),                              # number of exceedances in cluster
        "peak": g["value"].max(),
        "peak_time": g.loc[g["value"].idxmax(), "time"],
    }))
    return out.reset_index(drop=True)


def build_exceedance_data(station_data, station_names, cols):
    exceedance_station_data = {}
    for station in station_names:
        d = station_data[station]
        x = d[VARIABLE]
        clusters = runs_decluster(x, U, R)
        d = d.loc[clusters["peak_time"], cols].copy()
        d["y"] = clusters["peak"].to_numpy() - U
        exceedance_station_data[station] = d.dropna()
    return exceedance_station_data


def augment_exceedance_data(exceedance_station_data, gam_models, logit_station_data, station_names):
    """Adds zeta = P(exceedance) predicted by the GAM as a covariate of the GPD scale."""
    for station in station_names:
        d = exceedance_station_data[station]
        lg_d = logit_station_data[station]
        gam = gam_models[station]
        
        d_idx = set(d.index)
        lg_d_idx = set(lg_d.index)
        joint_idx = list(d_idx.intersection(lg_d_idx))
        
        print(len(joint_idx), "/", len(d_idx))
        d = d.loc[joint_idx, ]
        lg_d = lg_d.loc[joint_idx, ]
        d["zeta"] = predict_gam(gam, lg_d)
        exceedance_station_data[station] = d
    return exceedance_station_data


def gpd_nll(params, y, Z):
    """params = [beta_0..beta_p, xi]; Z includes the intercept column."""
    beta, xi = params[:-1], params[-1]
    sigma = np.exp(Z @ beta)
    if abs(xi) < 1e-6:                          # Gumbel limit (exponential excesses)
        return np.sum(np.log(sigma) + y / sigma)
    t = 1 + xi * y / sigma
    if np.any(t <= 0):                          # outside the support
        return 1e10
    return np.sum(np.log(sigma) + (1 + 1 / xi) * np.log(t))


def fit_cov_gpd(data, covs, xi0=0.1):
    y = data["y"].to_numpy()
    mu, sd = data[covs].mean(), data[covs].std()
    Zc = ((data[covs] - mu) / sd).to_numpy() if covs else np.empty((len(data), 0))
    Z = np.column_stack([np.ones(len(data)), Zc])

    # start: intercept = log of the mean excess, other betas = 0
    x0 = np.r_[np.log(y.mean()), np.zeros(Z.shape[1] - 1), xi0]

    res = minimize(gpd_nll, x0, args=(y, Z), method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-8})
    res = minimize(gpd_nll, res.x, args=(y, Z), method="BFGS")   # polish

    return {"params": res.x, "nll": res.fun, "k": len(res.x), "n": len(y),
            "names": ["intercept"] + covs + ["xi"],
            "mu": mu, "sd": sd, "Z": Z, "y": y, "covs": covs}
    
    
def fit_cov_gpd_all_stations(station_data, station_names, covs_per_station, xi0=0.1):
    models = {}
    for station in station_names:
        if isinstance(covs_per_station, dict):
            covs = covs_per_station[station]
        else:
            covs = covs_per_station
        d = station_data[station]
        model = fit_cov_gpd(d, covs, xi0)
        models[station] = model.copy()
    return models


def num_hessian(f, x, eps=1e-4):
    k = len(x)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = eps, eps
            H[i, j] = H[j, i] = (f(x + ei + ej) - f(x + ei - ej)
                                 - f(x - ei + ej) + f(x - ei - ej)) / (4 * eps**2)
    return H


def summary(m):
    f = lambda p: gpd_nll(p, m["y"], m["Z"])
    cov = np.linalg.inv(num_hessian(f, m["params"]))
    se = np.sqrt(np.diag(cov))
    z = m["params"] / se
    from scipy.stats import norm
    out = pd.DataFrame({"coef": m["params"], "se": se, "z": z,
                        "p": 2 * (1 - norm.cdf(np.abs(z)))}, index=m["names"])
    out["exp(coef)"] = np.exp(out["coef"])        # meaningful for the betas only
    return out


def prepare_test_data(data):
    x = data[VARIABLE]
    data["x_t"] = x                                   # current level
    data["exc_t"] = (x > U).astype(float)             # exceeded now?
    data["x_lag1"] = x.shift(1)
    data["x_lag24"] = x.shift(24)
    data["max24"] = x.rolling(24).max()               # recent peak
    data["n_exc7"] = (x > U).astype(float).rolling(7).sum()
    return data


def prob_exceed(c, zeta, sigma, xi, u):
    """P(X > c) = zeta * (1 + xi (c-u)/sigma)^(-1/xi) for c > u."""
    if c <= u:
        return np.ones_like(zeta)
    t = np.maximum(1 + xi * (c - u) / sigma, 0)     # t=0 -> prob 0 (bounded tail)
    surv = np.exp(-(c - u) / sigma) if abs(xi) < 1e-6 else t ** (-1 / xi)
    # print("Zeta:", zeta.round(3))
    # print("Surv:", surv)
    return zeta * surv


def two_part_prob(c, data, gam_model, gpd_model, u):
    zeta = predict_gam(gam_model, data).to_numpy()
    # print(zeta)
    data["zeta"] = zeta

    covs = gpd_model["covs"]
    # print(data[covs])
    # print(gpd_model["mu"])
    # print(gpd_model["sd"])
    Zs = ((data[covs] - gpd_model["mu"]) / (1*gpd_model["sd"])).to_numpy()
    # print(Zs)
    # print(gpd_model["params"][1:-1])
    # print(Zs @ gpd_model["params"][1:-1])
    sigma = np.exp(gpd_model["params"][0] + Zs @ gpd_model["params"][1:-1])
    # print(sigma)
    xi = gpd_model["params"][-1]
    return prob_exceed(c, zeta, sigma, xi, u)


def main():
    # General data preparation
    data = load_data("data/datos_corregidos.csv")
    data2 = load_data("data/datos_no_imputados.csv")
    station_data, STATIONS = group_data_by_station(data)
    station_data2, STATIONS = group_data_by_station(data2)
    station_data = feature_engineering(station_data, STATIONS)
    station_data2 = feature_engineering(station_data2, STATIONS)
    station_train_data, _ = split_data(station_data, STATIONS)
    _, station_test_data = split_data(station_data2, STATIONS)
    
    # MODEL 1: Generalized Additive Model (GAM)
    logit_station_data = build_logit_station_data(station_train_data, STATIONS)
    persistence_feats = ["x_t", "exc_t", "max24", "n_exc7"]
    meteorological_feats = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_feats = ["hour_sin", "hour_cos"]
    logit_feats = {
        "CE": persistence_feats + meteorological_feats + time_feats,
        "NE": persistence_feats + meteorological_feats + time_feats,
        "NE2": persistence_feats + time_feats,
        "NE3": persistence_feats + meteorological_feats + time_feats,
        "NO": persistence_feats + meteorological_feats + time_feats,
        "NO2": persistence_feats + meteorological_feats + time_feats,
        "NO3": persistence_feats + time_feats,
        "NTE": persistence_feats + meteorological_feats + time_feats,
        "NTE2": persistence_feats + meteorological_feats + time_feats,
        "SE": persistence_feats + meteorological_feats + time_feats,
        "SE2": persistence_feats + meteorological_feats + time_feats,
        "SE3": persistence_feats + meteorological_feats + time_feats,
        "SO": persistence_feats + time_feats,
        "SO2": persistence_feats + meteorological_feats + time_feats,
        "SUR": persistence_feats + meteorological_feats + time_feats,
    }
    gam_models = fit_gam_all_stations(logit_station_data, STATIONS, logit_feats)

    # MODEL 2: GPD
    cols = ["WSR", "TOUT", "RH", "PRS", "SR", "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_u", "wind_v"]
    exceedance_station_data = build_exceedance_data(station_train_data, STATIONS, cols)
    exceedance_station_data = augment_exceedance_data(exceedance_station_data, gam_models, logit_station_data, STATIONS)
    met_covs = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_covs = ["hour_sin", "hour_cos"]
    zeta_covs = ["zeta"]
    covs = {
        "CE": met_covs+time_covs+zeta_covs,
        "NE": time_covs+zeta_covs,
        "NE2": zeta_covs,
        "NE3": zeta_covs,
        "NO": time_covs+zeta_covs,
        "NO2": zeta_covs,
        "NO3": zeta_covs,
        "NTE": zeta_covs,
        "NTE2": met_covs+time_covs+zeta_covs,
        "SE": met_covs,
        "SE2": time_covs+zeta_covs,
        "SE3": time_covs+zeta_covs,
        "SO": zeta_covs,
        "SO2": zeta_covs,
        "SUR": met_covs+time_covs+zeta_covs,
    }
    gpd_models = fit_cov_gpd_all_stations(exceedance_station_data, STATIONS, covs)
    
    # Final Summary
    for station in STATIONS:
        print("\n-----", station, "------")
        model = gpd_models[station]
        print(summary(model))
        
    # Example: predictions
    # test = station_test_data["SUR"]
    # test = prepare_test_data(test)
    # test = test[
    #     (test.index >= "2025-11-17 00:00:00")
    #     & (test.index <= "2025-11-17 23:00:00")
    # ]
    # gam_model = gam_models["SUR"]
    # gpd_model = gpd_models["SUR"]
    # p_c = two_part_prob(C, test, gam_model, gpd_model, U)
    # print(p_c.round(3))
    
    # Complete predictions
    preds_station = {}
    threshold = 0.44    # 0.6 - 0.65 (recomendado, corregidos) / ~0.15 (no imputados) / 0.02 - 6 horas / 0.25 - 3 horas
    for station in STATIONS:
        test = station_test_data[station]
        test = prepare_test_data(test)
        preds = pd.DataFrame(index=test.index)
        preds["x"] = test[VARIABLE]
        preds["real_exc"] = (preds["x"] >= C)
        gam_model = gam_models[station]
        gpd_model = gpd_models[station]
        p_c = two_part_prob(C, test, gam_model, gpd_model, U)
        preds["p"] = p_c
        # preds["p"] = preds["p"].shift(-6)
        preds["pred_exc"] = (preds["p"] >= threshold)
        preds_station[station] = preds

    confusion_matrix = metrics.confusion_matrix(preds_station["SUR"]["real_exc"], preds_station["SUR"]["pred_exc"])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["No Exceso", "Exceso"])
    cm_display.plot()
    plt.show()
    
    fpr, tpr, _ = metrics.roc_curve(preds_station["SUR"]["real_exc"], preds_station["SUR"]["pred_exc"])
    roc_auc = metrics.auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'(AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Two Models')
    plt.legend()
    plt.show()
     

if __name__ == "__main__":
    main()