import numpy as np
import pandas as pd
from scipy.stats import chi2

try:
    from model_hourly import (
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
    from model_hourly import (
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


LOGREG_MODEL_NAMES = (
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
)


def lr_test_gpd(m_small, m_big):
    lr_stat = 2 * (m_small["nll"] - m_big["nll"])
    df = m_big["k"] - m_small["k"]
    p_value = 1 - chi2.cdf(lr_stat, df)
    return lr_stat, df, p_value


def lr_test_logreg(restricted_res, full_res):
    """
    Performs a Likelihood Ratio Test comparing two nested statsmodels Logit results.
    """
    # Likelihood Ratio Statistic
    lr_stat = 2 * (full_res.llf - restricted_res.llf)
    
    # Degrees of freedom difference
    df = full_res.df_model - restricted_res.df_model
    
    # p-value from Chi-Squared distribution
    p_value = chi2.sf(lr_stat, df)
    
    return lr_stat, df, p_value


def compare_logreg_to_null(models):
    null_comparisons = {
        "Model Name": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "AIC": [],
    }
    for model, name in zip(models, LOGREG_MODEL_NAMES):
        null_comparisons["Model Name"].append(name)
        null_comparisons["LR-test Statistic"].append(model[0].llr)
        null_comparisons["P-value"].append(model[0].llr_pvalue)
        null_comparisons["Degrees of Freedom"].append(model[0].df_model)
        null_comparisons["AIC"].append(model[0].aic)
    return pd.DataFrame(null_comparisons)


def compare_logregs(models):
    comparisons = ((0, 1), (0, 2), (1, 3), (2, 3), (0, 3))
    comparisons_d = {
        "Comparison": [],
        "LR-test Statistic": [],
        "P-value": [],
        "Degrees of Freedom": [],
        "AIC_small": [],
        "AIC_big": [],
    }
    for small, big in comparisons:
        model_small, small_name = models[small], LOGREG_MODEL_NAMES[small]
        model_big, big_name = models[big], LOGREG_MODEL_NAMES[big]
        lr_stat, df, p_value = lr_test_logreg(model_small[0], model_big[0])
        comparisons_d["Comparison"].append(f"{small_name} vs {big_name}")
        comparisons_d["LR-test Statistic"].append(lr_stat)
        comparisons_d["P-value"].append(p_value)
        comparisons_d["Degrees of Freedom"].append(df)
        comparisons_d["AIC_small"].append(model_small[0].aic)
        comparisons_d["AIC_big"].append(model_big[0].aic)
    return pd.DataFrame(comparisons_d)


def compare_logreg_all_stations(
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
        "AIC": [],
    })
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
            model_family_4[station]
        ]
        
        # Null comparisons
        null_comp = compare_logreg_to_null(models)
        null_comp["Station"] = station
        null_comparisons = pd.concat([null_comparisons, null_comp])
        
        # Comparisons against one another
        comp = compare_logregs(models)
        comp["Station"] = station
        comparisons = pd.concat([comparisons, comp])
        
    return null_comparisons, comparisons


def compare_gpd(models):
    comparisons = (
        (0, 1), 
        (0, 2), (1, 2), 
        (0, 3), 
         (0, 4), (3, 4),
        (0, 5), (1, 5), (3, 5), 
        (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), (5, 6),
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
        ]

        comp = compare_gpd(models)
        comp["Station"] = station
        comparisons = pd.concat([comparisons, comp])
        
    return comparisons


def main():
# General data preparation
    data = load_data("data/datos_no_imputados.csv")
    station_data, STATIONS = group_data_by_station(data)
    station_data = feature_engineering(station_data, STATIONS)
    station_train_data, station_test_data = split_data(station_data, STATIONS)
    
    # MODEL 1: Logistic Regression
    logit_station_data = build_logit_station_data(station_train_data, STATIONS)
    persistence_feats = ["x_t", "exc_t", "max24", "n_exc7"]
    meteorological_feats = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_feats = ["hour_sin", "hour_cos"]
    logit_persistence_models = fit_logit_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats
    )
    logit_met_models = fit_logit_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats + meteorological_feats
    )
    logit_time_models = fit_logit_all_stations(
        logit_station_data, 
        STATIONS, 
        persistence_feats + time_feats
    )
    logit_met_time_models = fit_logit_all_stations(
        logit_station_data,
        STATIONS, 
        persistence_feats + meteorological_feats + time_feats
    )
    
    # Logistic regression comparisons
    null_comp_logreg_df, comp_logreg_df = compare_logreg_all_stations(
        logit_persistence_models,
        logit_met_models,
        logit_time_models,
        logit_met_time_models,
        STATIONS
    )
    null_comp_logreg_df.to_csv('data/comparisons/null_logreg.csv', index=False)
    comp_logreg_df.to_csv('data/comparisons/logreg.csv', index=False)
    
    
    # MODEL 2: GPD
    cols = ["WSR", "TOUT", "RH", "PRS", "SR", "hour_sin", "hour_cos", "month_sin", "month_cos", "wind_u", "wind_v"]
    exceedance_station_data = build_exceedance_data(station_train_data, STATIONS, cols)
    pers_exceedance_data = augment_exceedance_data(exceedance_station_data, logit_persistence_models, logit_station_data, STATIONS)
    met_exceedance_data = augment_exceedance_data(exceedance_station_data, logit_met_models, logit_station_data, STATIONS)
    time_exceedance_data = augment_exceedance_data(exceedance_station_data, logit_time_models, logit_station_data, STATIONS)
    met_time_exceedance_data = augment_exceedance_data(exceedance_station_data, logit_met_time_models, logit_station_data, STATIONS)
    stationary_covs = []
    met_covs = ["WSR", "TOUT", "RH", "wind_u", "wind_v"]
    time_covs = ["hour_sin", "hour_cos"]
    zeta_covs = ["zeta"]
    gpd_stationary_models = fit_cov_gpd_all_stations(pers_exceedance_data, STATIONS, stationary_covs)
    gpd_met_models = fit_cov_gpd_all_stations(met_exceedance_data, STATIONS, met_covs)
    gpd_met_zeta_models = fit_cov_gpd_all_stations(met_exceedance_data, STATIONS, met_covs + zeta_covs)
    gpd_time_models = fit_cov_gpd_all_stations(time_exceedance_data, STATIONS, time_covs)
    gpd_time_zeta_models = fit_cov_gpd_all_stations(time_exceedance_data, STATIONS, time_covs + zeta_covs)
    gpd_met_time_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs + time_covs)
    gpd_met_time_zeta_models = fit_cov_gpd_all_stations(met_time_exceedance_data, STATIONS, met_covs + time_covs + zeta_covs)
    
    
    # GPD comparisons
    comp_gpd_df = compare_gpd_all_stations(
        gpd_stationary_models,
        gpd_met_models,
        gpd_met_zeta_models,
        gpd_time_models,
        gpd_time_zeta_models,
        gpd_met_time_models,
        gpd_met_time_zeta_models,
        STATIONS
    )
    comp_gpd_df.to_csv('data/comparisons/gpd.csv', index=False)
    

if __name__ == '__main__':
    main()