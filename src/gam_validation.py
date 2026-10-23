import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox

try:
    from gam_daily import (
        load_data,
        group_data_by_station,
        feature_engineering,
        split_data,
        build_logit_station_data,
        fit_gam,
        fit_gam_all_stations
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from gam_daily import (
        load_data,
        group_data_by_station,
        feature_engineering,
        split_data,
        build_logit_station_data,
        fit_gam,
        fit_gam_all_stations
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _term_columns(model):
    """
    Maps every term of the GAM to the columns of the full design matrix
    (parametric columns first, then one block of B-spline columns per smooth).
    """
    names = list(model.res.model.exog_names)
    linear = {f: [names.index(f)] for f in model.linear_feats}
    smooth = {
        f: [i for i, n in enumerate(names) if re.fullmatch(rf"{re.escape(f)}_s\d+", n)]
        for f in model.smooth_feats
    }
    return names, linear, smooth


def hosmer_lemeshow(y, p, groups=10):
    """Hosmer-Lemeshow goodness-of-fit test on `groups` risk bins (df = groups - 2)."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), groups, labels=False)
    g = d.groupby("bin").agg(n=("y", "size"), observed=("y", "sum"), expected=("p", "sum"))
    var = g["expected"] * (1 - g["expected"] / g["n"])
    stat = float((((g["observed"] - g["expected"]) ** 2) / var.replace(0, np.nan)).sum())
    dof = len(g) - 2
    p_value = float(chi2.sf(stat, dof)) if dof > 0 else np.nan
    return stat, dof, p_value, g


# -----------------------------------------------------------------------------
# Main validation routine
# -----------------------------------------------------------------------------
def validate_gam(model, train, make_plots=False, hl_groups=10, basis_ratio_flag=0.9, min_nl_df=0.5):
    """
    Statistical validations for a fitted GAM (GAMFit from model_daily.fit_gam):

    1. Smooth terms (replaces Box-Tidwell):
         - Wald test of each smooth (is the effect different from zero?)
         - LR test smooth vs. linear term (is the non-linearity needed?). If the
           smooth has been penalised down to ~1 edf it is flagged as "effectively
           linear" instead of reporting a p-value
         - edf vs. basis dimension (is the basis large enough? edf close to the
           basis dimension means the smooth is being constrained by the basis)
    2. Multicollinearity / concurvity:
         - VIF of the (standardised) raw features
         - Observed concurvity of every term (R^2 of the term's fitted
           contribution regressed on all the other columns of the design)
    3. Influential observations: Cook's distance and leverage built from the
       *penalised* hat matrix, and Pearson residuals
    4. Independence of errors: Durbin-Watson and Ljung-Box on Pearson residuals
    5. Goodness of fit / calibration (binary response, so deviance alone is not
       informative): Hosmer-Lemeshow, AUC, Brier score, deviance explained
    """
    results = {}
    res = model.res
    y = train["y"].to_numpy()
    n = len(train)
    names, linear_cols, smooth_cols = _term_columns(model)
    beta = np.asarray(res.params)
    X = np.asarray(res.model.exog)

    # -------------------------------------------------------------------------
    # 1. Smooth terms: significance, non-linearity and basis size
    # -------------------------------------------------------------------------
    edf_coef = np.asarray(res.edf) if hasattr(res, "edf") else np.ones(len(beta))
    rows = []
    for i, feat in enumerate(model.smooth_feats):
        cols = smooth_cols[feat]
        term_edf = float(edf_coef[cols].sum())
        basis_dim = len(cols)

        wald = res.test_significance(i)
        wald_stat = float(np.squeeze(wald.statistic))
        wald_p = float(np.squeeze(wald.pvalue))

        # Is the smooth really needed? Compare against the same model with `feat`
        # entering linearly (same penalty weight -> like-for-like, nested).
        reduced = fit_gam(
            train, model.feats, df=model.df_basis, degree=model.degree,
            alpha=model.alpha, linear_feats=tuple(model.linear_feats) + (feat,)
        )
        nl_stat = max(2.0 * (model.llf - reduced.llf), 0.0)
        nl_df = model.edf - reduced.edf
        # If the penalty already shrank the smooth to (almost) a straight line the
        # extra df is ~0 and a chi2 p-value is meaningless -> report it as linear.
        effectively_linear = nl_df < min_nl_df
        nl_p = np.nan if effectively_linear else float(chi2.sf(nl_stat, nl_df))

        rows.append({
            "Feature": feat,
            "EDF": term_edf,
            "Basis dim": basis_dim,
            "EDF / basis": term_edf / basis_dim,
            "Wald chi2": wald_stat,
            "Wald p-value": wald_p,
            "Nonlinearity LR": nl_stat,
            "Nonlinearity df": nl_df,
            "Nonlinearity p-value": nl_p,
            "Effectively linear?": effectively_linear,
            "Basis too small?": (term_edf / basis_dim) > basis_ratio_flag,
        })
    results["smooth_terms"] = pd.DataFrame(rows)

    # Parametric (linear) terms
    par_names = ["const"] + list(model.linear_feats)
    results["parametric_terms"] = pd.DataFrame({
        "coef": res.params.loc[par_names],
        "se": res.bse.loc[par_names],
        "p-value": res.pvalues.loc[par_names],
    })

    # -------------------------------------------------------------------------
    # 2. Multicollinearity (VIF) and concurvity
    # -------------------------------------------------------------------------
    Z = (train[model.feats] - train[model.feats].mean()) / train[model.feats].std()
    Z = sm.add_constant(Z)
    results["vif"] = pd.DataFrame({
        "Feature": Z.columns,
        "VIF": [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]
    })

    conc = []
    all_terms = {**linear_cols, **smooth_cols}
    for feat in model.feats:
        cols = all_terms[feat]
        f_j = X[:, cols] @ beta[cols]
        f_j = f_j - f_j.mean()
        others = np.delete(X, cols, axis=1)           # includes the intercept
        denom = float(f_j @ f_j)
        if denom < 1e-12:
            conc.append(np.nan)
            continue
        coef, *_ = np.linalg.lstsq(others, f_j, rcond=None)
        resid_j = f_j - others @ coef
        conc.append(1.0 - float(resid_j @ resid_j) / denom)
    results["concurvity"] = pd.DataFrame({
        "Feature": model.feats,
        "Type": ["smooth" if f in model.smooth_feats else "linear" for f in model.feats],
        "Observed concurvity": conc,
    })

    # -------------------------------------------------------------------------
    # 3. Influential observations (penalised hat matrix)
    # -------------------------------------------------------------------------
    residuals = np.asarray(res.resid_pearson)
    leverage = np.asarray(res.get_hat_matrix_diag())      # diag of the penalised hat matrix
    p_eff = model.edf
    cooks_d = residuals ** 2 * leverage / (p_eff * (1.0 - leverage) ** 2)   # dispersion = 1 (binomial)

    cooks_threshold = 4 / n
    lev_threshold = 2 * p_eff / n
    cook_idx = np.where(cooks_d > cooks_threshold)[0]
    pearson_idx = np.where(np.abs(residuals) > 3)[0]
    leverage_idx = np.where(leverage > lev_threshold)[0]
    outliers_idx = np.where((cooks_d > cooks_threshold) | (np.abs(residuals) > 3))[0]

    results["cooks_distance"] = cooks_d
    results["leverage"] = leverage
    results["residuals"] = residuals
    results["flagged_outliers_indices"] = train.index[outliers_idx].tolist()
    results["flagged_cook_indices"] = train.index[cook_idx].tolist()
    results["flagged_pearson_indices"] = train.index[pearson_idx].tolist()
    results["flagged_leverage_indices"] = train.index[leverage_idx].tolist()

    # -------------------------------------------------------------------------
    # 4. Independence of errors
    # -------------------------------------------------------------------------
    results["durbin_watson"] = durbin_watson(residuals)
    results["lag1_autocorr"] = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
    lb = acorr_ljungbox(residuals, lags=[10])
    results["ljung_box"] = {"stat": float(lb["lb_stat"].iloc[0]), "p_value": float(lb["lb_pvalue"].iloc[0]), "lag": 10}

    # -------------------------------------------------------------------------
    # 5. Goodness of fit / calibration
    # -------------------------------------------------------------------------
    p_hat = np.asarray(res.fittedvalues)
    hl_stat, hl_df, hl_p, hl_table = hosmer_lemeshow(y, p_hat, hl_groups)
    results["hosmer_lemeshow"] = {"stat": hl_stat, "df": hl_df, "p_value": hl_p, "table": hl_table}
    results["gof"] = {
        "auc": float(roc_auc_score(y, p_hat)) if 0 < y.mean() < 1 else np.nan,
        "brier": float(np.mean((y - p_hat) ** 2)),
        "deviance_explained": float(1.0 - res.deviance / res.null_deviance),
        "aic": model.aic,
        "edf": model.edf,
        "alpha": model.alpha,
    }

    # -------------------------------------------------------------------------
    # Visual diagnostics
    # -------------------------------------------------------------------------
    if make_plots:
        # A) Estimated smooth effects (logit scale, +/- 1.96 se)
        k = len(model.smooth_feats)
        if k:
            ncols = min(k, 3)
            nrows = int(np.ceil(k / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.6 * nrows), squeeze=False)
            for i, feat in enumerate(model.smooth_feats):
                ax = axes[i // ncols][i % ncols]
                fit_i, se_i = res.partial_values(i, include_constant=False)
                fit_i, se_i = np.asarray(fit_i), np.asarray(se_i)
                x = train[feat].to_numpy()
                order = np.argsort(x)
                ax.plot(x[order], fit_i[order], color="C0")
                ax.fill_between(x[order], (fit_i - 1.96 * se_i)[order], (fit_i + 1.96 * se_i)[order],
                                color="C0", alpha=0.25)
                ax.axhline(0, color="k", lw=0.5)
                ax.plot(x, np.full(len(x), ax.get_ylim()[0]), "|", color="k", alpha=0.3)  # rug
                ax.set_title(f"s({feat})  edf={results['smooth_terms'].loc[i, 'EDF']:.2f}")
                ax.set_xlabel(feat)
                ax.set_ylabel("Effect on log-odds")
            for j in range(k, nrows * ncols):
                axes[j // ncols][j % ncols].axis("off")
            plt.tight_layout()
            plt.show()

        # B) Influence, residuals, calibration
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        axes[0].stem(range(n), cooks_d, markerfmt=" ", basefmt="k-")
        axes[0].axhline(y=cooks_threshold, color='r', linestyle='--', label=f'Threshold (4/n = {cooks_threshold:.4f})')
        axes[0].set_title("Cook's Distance")
        axes[0].set_xlabel("Observation Index")
        axes[0].set_ylabel("Cook's D")
        axes[0].legend()

        axes[1].scatter(train.index, residuals, alpha=0.6, edgecolors='k', s=20)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].axhline(y=3, color='g', linestyle=':', label='±3 Std Dev Bounds')
        axes[1].axhline(y=-3, color='g', linestyle=':')
        axes[1].set_title(f"Pearson Residuals vs Index (DW Stat: {results['durbin_watson']:.2f})")
        axes[1].set_xlabel("Train Index")
        axes[1].set_ylabel("Pearson Residuals")
        axes[1].legend()

        pred = hl_table["expected"] / hl_table["n"]
        obs = hl_table["observed"] / hl_table["n"]
        axes[2].plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        axes[2].plot(pred, obs, "o-", label="GAM (risk deciles)")
        axes[2].set_title(f"Calibration (Hosmer-Lemeshow p = {hl_p:.3f})")
        axes[2].set_xlabel("Mean predicted probability")
        axes[2].set_ylabel("Observed frequency")
        axes[2].legend()

        plt.tight_layout()
        plt.show()

    return results


def validate_all_gam_models(station_data, gam_models, station_names):
    validations = {}
    for station in station_names:
        train = station_data[station]
        validations[station] = validate_gam(gam_models[station], train, make_plots=(station == "SUR"))
    return validations


def main():
    # General data preparation
    data = load_data("data/datos_corregidos.csv")
    station_data, STATIONS = group_data_by_station(data)
    station_data = feature_engineering(station_data, STATIONS)
    station_train_data, station_test_data = split_data(station_data, STATIONS)

    # MODEL 1: Generalized Additive Model
    logit_station_data = build_logit_station_data(station_train_data, STATIONS)
    gam_feats = ["x_t", "exc_t", "max24", "n_exc7", "WSR", "TOUT", "RH", "wind_u", "wind_v"]
    gam_models = fit_gam_all_stations(logit_station_data, STATIONS, gam_feats)

    # 2. Run validations
    gam_validations = validate_all_gam_models(logit_station_data, gam_models, STATIONS)

    # 3. Print test summaries
    v = gam_validations["SUR"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    print("--- Smooth terms ---")
    print("(Wald p < 0.05: effect is significant | Nonlinearity p < 0.05: smooth is better than a linear term |")
    print(" Effectively linear: penalty shrank the smooth to ~a straight line | EDF/basis > 0.9: basis may be too small, increase GAM_DF)")
    print(v["smooth_terms"].round(3).to_string(index=False))

    print("\n--- Parametric terms ---")
    print(v["parametric_terms"].round(3))

    print("\n--- Variance Inflation Factors (VIF > 5-10 indicates high multicollinearity) ---")
    print(v["vif"].round(3))

    print("\n--- Observed concurvity (> 0.8 indicates a term is almost explained by the others; "
          "flexible smooths make this measure conservative) ---")
    print(v["concurvity"].round(3).to_string(index=False))

    print(f"\n--- Durbin-Watson Statistic: {v['durbin_watson']:.4f} ---")
    print("(Values near 2.0 indicate independent errors; < 1.5 suggests positive autocorrelation)")
    lb = v["ljung_box"]
    print(f"Lag-1 autocorrelation of residuals: {v['lag1_autocorr']:.3f}")
    print(f"Ljung-Box (lag {lb['lag']}): stat = {lb['stat']:.3f}, p-value = {lb['p_value']:.3f}")

    hl = v["hosmer_lemeshow"]
    print(f"\n--- Hosmer-Lemeshow: stat = {hl['stat']:.3f}, df = {hl['df']}, p-value = {hl['p_value']:.3f} ---")
    print("(p < 0.05 suggests poor calibration)")
    g = v["gof"]
    print(f"AUC = {g['auc']:.3f} | Brier = {g['brier']:.4f} | Deviance explained = {g['deviance_explained']:.3f} "
          f"| AIC = {g['aic']:.1f} | EDF = {g['edf']:.2f} | penalty alpha = {g['alpha']:.3g}")

    print(f"\nFlagged Outlier Indices: {len(v['flagged_outliers_indices'])} rows found.")
    print(f"Flagged Outlier Indices (Cook): {len(v['flagged_cook_indices'])} rows found.")
    print(f"Flagged Outlier Indices (Pearson): {len(v['flagged_pearson_indices'])} rows found.")
    print(f"Flagged High-Leverage Indices: {len(v['flagged_leverage_indices'])} rows found.")


if __name__ == "__main__":
    main()