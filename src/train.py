import sys
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import add_maxis
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import add_maxis


U_THRESHOLDS = {
    "PM2.5": 38.0,
    "O3": 63.0,
    "SO2": 20.2
}


def augment_w_resonances(df):
    # YET TO BE IMPLEMENTED #
    return df


def get_data(filename, standardize=True):
    df = pd.read_csv(filename)
        
    # Filter to et all data after 2023
    df["Fecha y hora"] = pd.to_datetime(df["Fecha y hora"])
    df_filtered = df["2023-01-01 00:00:00" <= df["Fecha y hora"]]
    df_filtered["Estacion"].value_counts()
    
    # Split datetime column
    df_filtered["Hora"] = df_filtered["Fecha y hora"].dt.hour
    df_filtered["Anio"] = df_filtered["Fecha y hora"].dt.year - 2023
    df_filtered["Mes"] = df_filtered["Fecha y hora"].dt.month
    
    df_augmented = augment_w_resonances(df_filtered)
    
    # Split train & test 
    df_val = df_augmented[
        "2026-01-01 00:00:00" <= df_augmented["Fecha y hora"]
    ]
    df_test = df_augmented[
        "2025-01-01 00:00:00" <= df_augmented["Fecha y hora"] < "2026-01-01 00:00:00"
    ]
    df_train = df_augmented[
        df_augmented["Fecha y hora"] < "2025-01-01 00:00:00"
    ]
    
    # Standardize
    if standardize:
        scaler = StandardScaler()
        numerical_features = [
            'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10',
            'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR',
            'Hora', 'Anio', 'Mes'
        ]
        augmented_features = []
        features = numerical_features + augmented_features

        df_train[features] = scaler.fit_transform(df_train[features])
        df_test[features] = scaler.transform(df_test[features])
        df_val[features] = scaler.transform(df_val[features])
        
        means = dict(zip(scaler.feature_names_in_, scaler.mean_))
        stds = dict(zip(scaler.feature_names_in_, scaler.scale_))
        
        U_THRESHOLDS_STD = {
            pollutant: (threshold - means[pollutant]) / stds[pollutant]
            for pollutant, threshold in U_THRESHOLDS.items()
        }
        
        U_THRESHOLDS = U_THRESHOLDS_STD
        
    # Get excess dataframes
    hourly_pollutants = ["O3", "SO3"]
    indexes = ["Estacion", "Fecha y hora"]
    target_cols = hourly_pollutants + indexes
    condition = (df_train["O3"] > U_THRESHOLDS["O3"]) | (df_train["SO2"] > U_THRESHOLDS["SO2"])
    
    df_train_ex = df_train[target_cols].where(condition)
    df_test_ex = df_test[target_cols].where(condition)
    df_val_ex = df_val[target_cols].where(condition)
    
    return df_train, df_train_ex, df_test, df_test_ex, df_val, df_val_ex


def rearrange_data(df):
    """
    Receives a pandas dataframe, rearranges it and returns it as numpy array
    """
    df_pivoted = df.pivot(
        index="Fecha y hora", 
        columns="Estacion",
        values=[
            'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10',
            'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR',
            'Hora', 'Anio', 'Mes'
        ]
    )
    
    num_intervals = df_pivoted.index.nunique()
    num_stations = df_pivoted.columns.levels[1].nunique()
    num_features = 18
    
    array = df_pivoted.to_numpy().reshape(num_intervals, num_features, num_stations)
    array = np.moveaxis(array, 1, 0)
    array = np.moveaxis(array, 1, -1)
    
    return array
    

def rearrange_excesses(df, pollutant):
    """
    Receives a pandas dataframe, rearranges it and returns it as numpy array
    """
    df_pivoted = df.pivot(
        index="Fecha y hora", 
        columns="Estacion",
        values=[pollutant]
    )
    
    num_intervals = df_pivoted.index.nunique()
    num_stations = df_pivoted.columns.levels[1].nunique()
    
    array = df_pivoted.to_numpy().reshape(num_intervals, num_stations)
    array = np.moveaxis(array, 1, 0)
    
    return array


def make_dict(data, excesses):
    data = {
        "vecs": data[:, :, :-1],
        "sets": excesses[:, :, :-1],
        "y": excesses[:, :, 1:]
    }
    return data

def model_for_(pollutant):
    # 1) GET THE DATA
    df_train, df_train_ex, df_test, df_test_ex, df_val, df_val_ex = get_data('../data/datos_imputados.csv', False)
    
    # 2) REARRANGE THE DATA & TURN IT INTO ARRAYS
    train_arr = rearrange_data(df_train)
    test_arr = rearrange_data(df_test)
    val_arr = rearrange_data(df_val)
    train_excesses = rearrange_excesses(df_train_ex, pollutant)
    train_excesses = rearrange_excesses(df_train_ex, pollutant)
    train_excesses = rearrange_excesses(df_train_ex, pollutant)
    
    # 3) SEND DATA TO TORCH (& DEVICE)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    train_tensor = torch.tensor(train_arr, dtype=torch.float64, device=device)
    train_exc_tensor = torch.tensor(train_excesses, dtype=torch.float64, device=device)
    test_tensor = torch.tensor(test_arr, dtype=torch.float64, device=device)
    test_exc_tensor = torch.tensor(train_excesses, dtype=torch.float64, device=device)
    val_tensor = torch.tensor(val_arr, dtype=torch.float64, device=device)
    val_exc_tensor = torch.tensor(train_excesses, dtype=torch.float64, device=device)
    
    # 4) CREATE THE MODEL
    model = Combiner(
        ds_params=params_dict["ds_params"],
        gcn_params=params_dict["gcn_params"],
        graph_params=params_dict["graph_params"]
    )
    
    # 5) REARRANGE THE DATA FOR MODEL TRAINING
    train_data = make_dict(train_tensor, train_exc_tensor)
    test_data = make_dict(test_tensor, test_exc_tensor)
    val_data = make_dict(val_tensor, val_exc_tensor)
    add_maxis(train_data, test_data, val_data)
    
    # 5) TRAIN THE MODEL
    
    
    # 6) SAVE THE MODEL
    
    # (7) EVALUATE RESULTS
    
    pass


def main():
    


if __name__ == "__main__":
    main()