import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import expon

try:
    from model_daily import (
        load_data,
        group_data_by_station,
        feature_engineering,
        split_data,
        build_logit_station_data,
        fit_logit_all_stations,
        build_exceedance_data,
        augment_exceedance_data,
        fit_cov_gpd_all_stations
    )
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from model_daily import (
        load_data,
        group_data_by_station,
        feature_engineering,
        split_data,
        build_logit_station_data,
        fit_logit_all_stations,
        build_exceedance_data,
        augment_exceedance_data,
        fit_cov_gpd_all_stations
    )

def validate_logit(res, train, feats, mu, sd, make_plots=False):
    """
    Carries out statistical validations for a fitted statsmodels Logit result:
    1. Linearity in Log-Odds (Box-Tidwell)
    2. Multicollinearity (VIF)
    3. Influential Outliers (Cook's Distance & Studentized/Pearson Residuals)
    4. Independence of Errors (Durbin-Watson & Residuals vs Index plot)
    """
    results = {}
    
    # Standardize features as done in fit_logit
    Z_df = (train[feats] - mu) / sd
    Z_with_const = sm.add_constant(Z_df)
    
    # -------------------------------------------------------------------------
    # 1. Linearity in Log-Odds (Box-Tidwell Test)
    # -------------------------------------------------------------------------
    # Box-Tidwell tests interaction terms: X * ln(X). 
    # Since Z can be negative/zero, shift feature values strictly above zero: X_pos = X - min(X) + 1
    bt_data = Z_df.copy()
    bt_terms = []
    
    for col in feats:
        x_shift = train[col] - train[col].min() + 1.0
        x_ln_x = x_shift * np.log(x_shift)
        bt_term_name = f"{col}_x_ln"
        bt_data[bt_term_name] = x_ln_x
        bt_terms.append(bt_term_name)
        
    bt_design = sm.add_constant(bt_data)
    if train["y"].shape[0] > 50:
        bt_model = sm.Logit(train["y"], bt_design).fit(disp=0)
    else:
        bt_model = sm.Logit(train["y"], bt_design).fit_regularized(method="l1", disp=0)
        
    
    # Extract p-values for interaction terms (p < 0.05 indicates non-linearity)
    bt_pvalues = bt_model.pvalues[bt_terms]
    results['box_tidwell_pvalues'] = bt_pvalues
    
    # -------------------------------------------------------------------------
    # 2. Multicollinearity (Variance Inflation Factor - VIF)
    # -------------------------------------------------------------------------
    # VIF is calculated on the design matrix (including constant)
    vif_df = pd.DataFrame({
        'Feature': Z_with_const.columns,
        'VIF': [variance_inflation_factor(Z_with_const.values, i) 
                for i in range(Z_with_const.shape[1])]
    })
    results['vif'] = vif_df
    
    # -------------------------------------------------------------------------
    # 3. Influential Outliers (Cook's Distance & Studentized Residuals)
    # -------------------------------------------------------------------------
    influence = res.get_influence()
    cooks_d = influence.cooks_distance[0]
    
    # Person / Studentized Residuals
    residuals = res.resid_pearson
    
    n = len(train)
    cooks_threshold = 4 / n
    cook_idx = np.where(cooks_d > cooks_threshold)[0]
    pearson_idx = np.where(np.abs(residuals) > 3)[0]
    outliers_idx = np.where((cooks_d > cooks_threshold) | (np.abs(residuals) > 3))[0]
    
    results['cooks_distance'] = cooks_d
    results['residuals'] = residuals
    results['flagged_outliers_indices'] = train.index[outliers_idx].tolist()
    results['flagged_cook_indices'] = train.index[cook_idx].tolist()
    results['flagged_pearson_indices'] = train.index[pearson_idx].tolist()
    
    # -------------------------------------------------------------------------
    # 4. Independence of Errors (Durbin-Watson & Residual Plot)
    # -------------------------------------------------------------------------
    dw_stat = durbin_watson(residuals)
    results['durbin_watson'] = dw_stat

    # -------------------------------------------------------------------------
    # Visual Diagnostics
    # -------------------------------------------------------------------------
    if make_plots:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot A: Cook's Distance
        axes[0].stem(range(n), cooks_d, markerfmt=" ", basefmt="k-")
        axes[0].axhline(y=cooks_threshold, color='r', linestyle='--', label=f'Threshold (4/n = {cooks_threshold:.4f})')
        axes[0].set_title("Cook's Distance")
        axes[0].set_xlabel("Observation Index")
        axes[0].set_ylabel("Cook's D")
        axes[0].legend()
        
        # Plot B: Residuals vs Index (Error Independence)
        axes[1].scatter(train.index, residuals, alpha=0.6, edgecolors='k', s=20)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].axhline(y=3, color='g', linestyle=':', label='±3 Std Dev Bounds')
        axes[1].axhline(y=-3, color='g', linestyle=':')
        axes[1].set_title(f"Pearson Residuals vs Index (DW Stat: {dw_stat:.2f})")
        axes[1].set_xlabel("Train Index")
        axes[1].set_ylabel("Pearson Residuals")
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
    
    return results


def validate_all_logit_models(station_data, logit_models, station_names):
    validations = {}
    for station in station_names:
        train = station_data[station]
        res, mu, sd, feats = logit_models[station]
        validation = validate_logit(res, train, feats, mu, sd, station=="SUR")
        validations[station] = validation
    return validations


def residuals(m):
    beta, xi = m["params"][:-1], m["params"][-1]
    sigma = np.exp(m["Z"] @ beta)
    return np.log1p(xi * m["y"] / sigma) / xi


def main():
    # General data preparation
    data = load_data("data/datos_corregidos.csv")
    station_data, STATIONS = group_data_by_station(data)
    station_data = feature_engineering(station_data, STATIONS)
    station_train_data, station_test_data = split_data(station_data, STATIONS)
    
    # MODEL 1: Logistic Regression
    logit_station_data = build_logit_station_data(station_train_data, STATIONS)
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
    logit_models = fit_logit_all_stations(logit_station_data, STATIONS, logit_feats)

    # 2. Run validations
    logit_validations = validate_all_logit_models(logit_station_data, logit_models, STATIONS)

    # 3. Print test summaries
    print("--- Box-Tidwell Linearity Test (p < 0.05 implies non-linearity) ---")
    print(logit_validations["SUR"]['box_tidwell_pvalues'].round(3))

    print("\n--- Variance Inflation Factors (VIF > 5-10 indicates high multicollinearity) ---")
    print(logit_validations["SUR"]['vif'].round(3))

    print(f"\n--- Durbin-Watson Statistic: {logit_validations["SUR"]['durbin_watson']:.4f} ---")
    print("(Values near 2.0 indicate independent errors; < 1.5 suggests positive autocorrelation)")

    print(f"\nFlagged Outlier Indices: {len(logit_validations["SUR"]['flagged_outliers_indices'])} rows found.")
    print(f"Flagged Outlier Indices (Cook): {len(logit_validations["SUR"]['flagged_cook_indices'])} rows found.")
    print(f"Flagged Outlier Indices (Pearson): {len(logit_validations["SUR"]['flagged_pearson_indices'])} rows found.")
    
    # MODEL 2: GPD Model
    cols = ["WSR", "TOUT", "RH", "PRS", "SR", "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_u", "wind_v"]
    exceedance_station_data = build_exceedance_data(station_train_data, STATIONS, cols)
    exceedance_station_data = augment_exceedance_data(exceedance_station_data, logit_models, logit_station_data, STATIONS)
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
    gpd_stat_models = fit_cov_gpd_all_stations(exceedance_station_data, STATIONS, [])
    
    for j, station in enumerate(STATIONS):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        m0 = gpd_stat_models[station]
        m1 = gpd_models[station]
        for m, lab in [(m0, "stationary"), (m1, "covariates")]:
            e = np.sort(residuals(m))
            q = expon.ppf((np.arange(1, len(e) + 1) - 0.5) / len(e))
            ax[0].plot(q, e, "o", ms=3, label=lab)
        lim = [0, max(q.max(), e.max())]
        ax[0].plot(lim, lim, "k--"); ax[0].set_xlabel("Exp(1) quantiles")
        ax[0].set_ylabel("Residual quantiles"); ax[0].legend()
        ax[1].hist(residuals(m1), bins=20, density=True)
        ax[1].plot(np.linspace(0, 5), expon.pdf(np.linspace(0, 5)))
        ax[0].grid()
        fig.suptitle(station)
        
        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    main()