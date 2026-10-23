import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from gam_daily import (
        load_data,
        group_data_by_station,
        feature_engineering,
        split_data,
        build_logit_station_data,
        fit_gam_all_stations,
        build_exceedance_data,
        augment_exceedance_data,
        fit_cov_gpd_all_stations
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
        fit_gam_all_stations,
        build_exceedance_data,
        augment_exceedance_data,
        fit_cov_gpd_all_stations
    )


GAM_MODEL_NAMES = (
    "( Persistence )",
    "( Persistence + Meteorological )",
    "( Persistence + Time )",
    "( Persistence + Meteorological + Time )",
)

GPD_MODEL_NAMES = (
    "( Stationary )",
    "( Meteorological )",
    "( Meteorological + Zeta )",
    "( Time )",
    "( Time + Zeta )",
    "( Meteorological + Time )",
    "( Meteorological + Time + Zeta )",
    "( Zeta )",
)


def lr_test_gpd(m_small, m_big):
    lr_stat = 2 * (m_small["nll"] - m_big["nll"])
    df = m_big["k"] - m_small["k"]
    p_value = 1 - chi2.cdf(lr_stat, df)
    return lr_stat, df, p_value


# Minimum difference in effective d.f. for which a chi-square reference is meaningful.
# (when the penalty shrinks the extra smooth terms to ~0 edf the LR test is uninformative)
MIN_LR_DF = 0.5


def lr_test_gam(restricted, full, min_df=MIN_LR_DF):
    """
    Likelihood Ratio Test between two nested GAMs (GAMFit objects from model_daily.fit_gam).

    * statistic : 2 * (llf_full - llf_restricted), using the un-penalised binomial log-likelihood
    * d.f.      : difference of the *effective* degrees of freedom (edf) of both models
    * p-value   : chi-square approximation with those d.f.

    With penalised smooths the chi-square reference is approximate (Wood, 2017, sec. 6.12),
    and nestedness is only approximate when each model selects its own penalty weight.
    The p-value is NaN if the d.f. difference is smaller than `min_df`.
    """
    lr_stat = max(2.0 * (full.llf - restricted.llf), 0.0)
    df = full.edf - restricted.edf
    p_value = float(chi2.sf(lr_stat, df)) if df >= min_df else np.nan
    return lr_stat, df, p_value


def compare_gam_to_null(models):
    """Each GAM against the intercept-only model (LR test + AIC)."""
    null_comparisons = {
        "Model Name": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "EDF": [],
        "AIC": [],
    }
    for model, name in zip(models, GAM_MODEL_NAMES):
        null_comparisons["Model Name"].append(name)
        null_comparisons["LR-test Statistic"].append(model.llr)
        null_comparisons["P-value"].append(model.llr_pvalue)
        null_comparisons["Degrees of Freedom"].append(model.df_model)
        null_comparisons["EDF"].append(model.edf)
        null_comparisons["AIC"].append(model.aic)
    return pd.DataFrame(null_comparisons)


def compare_gams(models):
    comparisons = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 3))
    comparisons_d = {
        "Comparison": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "EDF_small": [],
        "EDF_big": [],
        "AIC_small": [],
        "AIC_big": [],
    }
    for small, big in comparisons:
        model_small, small_name = models[small], GAM_MODEL_NAMES[small]
        model_big, big_name = models[big], GAM_MODEL_NAMES[big]
        lr_stat, df, p_value = lr_test_gam(model_small, model_big)
        comparisons_d["Comparison"].append(f"{small_name} vs {big_name}")
        comparisons_d["LR-test Statistic"].append(lr_stat)
        comparisons_d["P-value"].append(p_value)
        comparisons_d["Degrees of Freedom"].append(df)
        comparisons_d["EDF_small"].append(model_small.edf)
        comparisons_d["EDF_big"].append(model_big.edf)
        comparisons_d["AIC_small"].append(model_small.aic)
        comparisons_d["AIC_big"].append(model_big.aic)
    return pd.DataFrame(comparisons_d)


def compare_gam_all_stations(
    model_family_1,
    model_family_2,
    model_family_3,
    model_family_4,
    station_names
):
    null_comparisons = pd.DataFrame({
        "Station": [],
        "Model Name": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "EDF": [],
        "AIC": [],
    })
    comparisons = pd.DataFrame({
        "Station": [],
        "Comparison": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "EDF_small": [],
        "EDF_big": [],
        "AIC_small": [],
        "AIC_big": [],
    })
    for station in station_names:
        models = [
            model_family_1[station],
            model_family_2[station],
            model_family_3[station],
            model_family_4[station]
        ]
        
        # Null comparisons
        null_comp = compare_gam_to_null(models)
        null_comp["Station"] = station
        null_comparisons = pd.concat([null_comparisons, null_comp])
        
        # Comparisons against one another
        comp = compare_gams(models)
        comp["Station"] = station
        comparisons = pd.concat([comparisons, comp])
        
    return null_comparisons, comparisons


def compare_gpd(models):
    comparisons = (
        (0, 1), 
        (0, 2), (1, 2), (7, 2),
        (0, 3), 
        (0, 4), (3, 4), (7, 4),
        (0, 5), (1, 5), (3, 5), 
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6), (7, 6),
        (0, 7)
    )
    comparisons_d = {
            "Comparison": [],
            "LR-test Statistic": [],
            "P-value": [],
            "Degrees of Freedom": [],
            "AIC_small": [],
            "AIC_big": [],
        }
    for small, big in comparisons:
        model_small, small_name = models[small], GPD_MODEL_NAMES[small]
        model_big, big_name = models[big], GPD_MODEL_NAMES[big]
        lr_stat, df, p_value = lr_test_gpd(model_small, model_big)
        comparisons_d["Comparison"].append(f"{small_name} vs {big_name}")
        comparisons_d["LR-test Statistic"].append(lr_stat)
        comparisons_d["P-value"].append(p_value)
        comparisons_d["Degrees of Freedom"].append(df)
        aic_small = 2 * model_small["nll"] + 2 * model_small["k"]
        aic_big = 2 * model_big["nll"] + 2 * model_big["k"]
        comparisons_d["AIC_small"].append(aic_small)
        comparisons_d["AIC_big"].append(aic_big)
    return pd.DataFrame(comparisons_d)


def compare_gpd_all_stations(
    model_family_1,
    model_family_2,
    model_family_3,
    model_family_4,
    model_family_5,
    model_family_6,
    model_family_7,
    model_family_8,
    station_names
):
    comparisons = pd.DataFrame({
        "Station": [],
        "Comparison": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "AIC_small": [],
        "AIC_big": [],
    })
    for station in station_names:
        models = [
            model_family_1[station],
            model_family_2[station],
            model_family_3[station],
            model_family_4[station],
            model_family_5[station],
            model_family_6[station],
            model_family_7[station],
            model_family_8[station],
        ]

        comp = compare_gpd(models)
        comp["Station"] = station
        comparisons = pd.concat([comparisons, comp])
        
    return comparisons


def main():
# General data preparation
    data = load_data("data/datos_corregidos.csv")
    station_data, STATIONS = group_data_by_station(data)
    station_data = feature_engineering(station_data, STATIONS)
    station_train_data, station_test_data = split_data(station_data, STATIONS)
    
    # MODEL 1: Generalized Additive Model (GAM)
    logit_station_data = build_logit_station_data(station_train_data, STATIONS)
    persistence_feats = ["x_t", "exc_t", "max24", "n_exc7"]
    meteorological_feats = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_feats = ["hour_sin", "hour_cos"]
    gam_persistence_models = fit_gam_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats
    )
    gam_met_models = fit_gam_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats + meteorological_feats
    )
    gam_time_models = fit_gam_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats + time_feats
    )
    gam_met_time_models = fit_gam_all_stations(
        logit_station_data,
        STATIONS, 
        persistence_feats + meteorological_feats + time_feats
    )
    
    # GAM comparisons
    null_comp_gam_df, comp_gam_df = compare_gam_all_stations(
        gam_persistence_models,
        gam_met_models,
        gam_time_models,
        gam_met_time_models,
        STATIONS
    )
    null_comp_gam_df.to_csv('data/comparisons/null_gam_hourly.csv', index=False)
    comp_gam_df.to_csv('data/comparisons/gam_hourly.csv', index=False)
    
    
    # MODEL 2: GPD
    cols = ["WSR", "TOUT", "RH", "PRS", "SR", "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_u", "wind_v"]
    exceedance_station_data = build_exceedance_data(station_train_data, STATIONS, cols)
    # pers_exceedance_data = augment_exceedance_data(exceedance_station_data, gam_persistence_models, logit_station_data, STATIONS)
    # met_exceedance_data = augment_exceedance_data(exceedance_station_data, gam_met_models, logit_station_data, STATIONS)
    # time_exceedance_data = augment_exceedance_data(exceedance_station_data, gam_time_models, logit_station_data, STATIONS)
    met_time_exceedance_data = augment_exceedance_data(exceedance_station_data, gam_met_time_models, logit_station_data, STATIONS)
    stationary_covs = []
    met_covs = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_covs = ["hour_sin", "hour_cos"]
    zeta_covs = ["zeta"]
    gpd_stationary_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, stationary_covs)
    gpd_met_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs)
    gpd_met_zeta_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs + zeta_covs)
    gpd_time_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, time_covs)
    gpd_time_zeta_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, time_covs + zeta_covs)
    gpd_met_time_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs + time_covs)
    gpd_met_time_zeta_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs + time_covs + zeta_covs)
    gpd_zeta_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, zeta_covs)
    
    
    # GPD comparisons
    comp_gpd_df = compare_gpd_all_stations(
        gpd_stationary_models,
        gpd_met_models,
        gpd_met_zeta_models,
        gpd_time_models,
        gpd_time_zeta_models,
        gpd_met_time_models,
        gpd_met_time_zeta_models,
        gpd_zeta_models,
        STATIONS
    )
    comp_gpd_df.to_csv('data/comparisons/gpd_gam_hourly.csv', index=False)
    

if __name__ == '__main__':
    main()