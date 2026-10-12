import sys
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import (
        add_maxis, 
        training_loop,
        TrainConfig
    )
    from utils.haversine_distance import haversine_matrix
    
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import (
        add_maxis, 
        training_loop,
        TrainConfig
    )
    from utils.haversine_distance import haversine_matrix


U_THRESHOLDS = {
    "PM2.5": 38.0,
    "O3": 63.0,
    "SO2": 20.2
}

STATION_COORDS = torch.tensor([
    [25.665278, -100.243611],  # 1
    [25.744722, -100.253056],  # 2
    [25.675833, -100.338333],  # 3
    [25.762778, -100.369167],  # 4
    [25.675556, -100.458056],  # 5
    [25.800278, -100.584444],  # 6
    [25.798611, -100.327222],  # 7
    [25.777222, -100.188056],  # 8
    [25.645833, -100.095278],  # 9
    [25.665000, -100.412778],  # 10
    [25.601111,  -99.999167],  # 11
    [25.616944, -100.273889],  # 12
    [25.729722, -100.310000],  # 13
    [25.790556, -100.078333],  # 14
    [25.768333, -100.463611],  # 15
], dtype=torch.float64)


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
    
    return df_filtered
    
def split_train_test(df, pollutant, u_threshold, standardize=True):
    # Split train & test 
    df_val = df[
        "2026-01-01 00:00:00" <= df["Fecha y hora"]
    ]
    df_test = df[
        ("2025-01-01 00:00:00" <= df["Fecha y hora"])
        & (df["Fecha y hora"] < "2026-01-01 00:00:00")
    ]
    df_train = df[
        df["Fecha y hora"] < "2025-01-01 00:00:00"
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
        
        u_threshold = (u_threshold - means[pollutant]) / stds[pollutant]
        
    # Get excess dataframes
    hourly_pollutants = [pollutant]
    indexes = ["Estacion", "Fecha y hora"]
    target_cols = hourly_pollutants + indexes
    condition_train = (df_train[pollutant] > u_threshold)
    condition_test = (df_test[pollutant] > u_threshold)
    condition_val = (df_val[pollutant] > u_threshold)
    
    df_train_ex = df_train[target_cols].copy()
    df_test_ex = df_test[target_cols].copy()
    df_val_ex = df_val[target_cols].copy()
    
    df_train_ex[pollutant] = df_train[pollutant].where(condition_train)
    df_test_ex[pollutant] = df_test[pollutant].where(condition_test)
    df_val_ex[pollutant] = df_val[pollutant].where(condition_val)
    
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
        values=pollutant
    )
    
    num_intervals = df_pivoted.index.nunique()
    num_stations = df_pivoted.columns.nunique()
    
    array = df_pivoted.to_numpy().reshape(num_intervals, num_stations, 1)
    array = np.moveaxis(array, -1, 0)
    array = np.moveaxis(array, 1, 2)
    
    return array


def make_dict(data, excesses):
    """
    The tensors returned in the dataset have the following shapes:
    (days/batches, num_features, num_stations, hours)
    """
    data = data.reshape([18, 15, 24, -1])
    excesses = excesses.reshape([1, 15, 24, -1])

    data = data.permute([3, 0, 1, 2])
    excesses = excesses.permute([3, 0, 1, 2])
    
    dict = {
        "vecs": data[:-1, :, :, :],
        "sets": excesses[:-1, :, :, :],
        "y": excesses[1:, :, :, :]
    }
    
    print("V", dict["vecs"].shape)
    print("S", dict["sets"].shape)
    print("Y", dict["y"].shape)
    return dict


def prepare_hourly_data(filename, pollutant, threshold):
    # 1) GET THE DATA
    df = get_data(filename, False)
    df = augment_w_resonances(df)
    df_train, df_train_ex, df_test, df_test_ex, df_val, df_val_ex = split_train_test(df, pollutant, threshold, False)
    
    # 2) REARRANGE THE DATA & TURN IT INTO ARRAYS
    train_arr = rearrange_data(df_train)
    test_arr = rearrange_data(df_test)
    val_arr = rearrange_data(df_val)
    train_excesses = rearrange_excesses(df_train_ex, pollutant)
    test_excesses = rearrange_excesses(df_test_ex, pollutant)
    val_excesses = rearrange_excesses(df_val_ex, pollutant)
    
    # 3) SEND DATA TO TORCH (& DEVICE)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    train_tensor = torch.tensor(train_arr, dtype=torch.float32, device=device)
    train_exc_tensor = torch.tensor(train_excesses, dtype=torch.float32, device=device)
    test_tensor = torch.tensor(test_arr, dtype=torch.float32, device=device)
    test_exc_tensor = torch.tensor(test_excesses, dtype=torch.float32, device=device)
    val_tensor = torch.tensor(val_arr, dtype=torch.float32, device=device)
    val_exc_tensor = torch.tensor(val_excesses, dtype=torch.float32, device=device)
    
    # 4) REARRANGE THE DATA FOR MODEL TRAINING
    train_data = make_dict(train_tensor, train_exc_tensor)
    test_data = make_dict(test_tensor, test_exc_tensor)
    val_data = make_dict(val_tensor, val_exc_tensor)
    add_maxis(train_data, test_data, val_data)
    
    return train_data, test_data, val_data


def main():
    train_data_o3, test_data_o3, val_data_o3 = prepare_hourly_data('data/datos_imputados.csv', "O3", 38.0)
    
    # CREATE MODEL
    model_o3 = Combiner(
        ds_params = params_dict["ds_params"],
        gcn_params = params_dict["gcn_params"],
        graph_params = params_dict["graph_params"],
    )
    
    train_config = TrainConfig()
    
    training_loop(
        model = model_o3, 
        train_data = train_data_o3, 
        val_data = test_data_o3, 
        D = haversine_matrix(STATION_COORDS), 
        config = train_config
    )


if __name__ == "__main__":
    main()