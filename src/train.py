import sys
import numpy as np
import pandas as pd
import torch
import pickle

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
    [25.675833, -100.338333],  # CE
    [25.744722, -100.253056],  # NE
    [25.777222, -100.188056],  # NE2
    [25.790556, -100.078333],  # NE3
    [25.762778, -100.369167],  # NO
    [25.800278, -100.584444],  # NO2
    [25.768333, -100.463611],  # NO3
    [25.798611, -100.327222],  # NTE
    [25.729722, -100.310000],  # NTE2
    [25.665278, -100.243611],  # SE
    [25.645833, -100.095278],  # SE2
    [25.601111,  -99.999167],  # SE3
    [25.675556, -100.458056],  # SO
    [25.665000, -100.412778],  # SO2
    [25.616944, -100.273889],  # SUR
], dtype=torch.float64)

NUM_VEC_FEATURES = 17


def augment_w_resonances(df):
    # YET TO BE IMPLEMENTED #
    return df


def get_data(filename):
    df = pd.read_csv(filename)
        
    # Filter to et all data after 2023
    df["Fecha y hora"] = pd.to_datetime(df["Fecha y hora"])
    df_filtered = df["2023-01-01 00:00:00" <= df["Fecha y hora"]]
    df_filtered["Estacion"].value_counts()
    
    # Split datetime column
    df_filtered["Hora"] = df_filtered["Fecha y hora"].dt.hour
    df_filtered["Dia"] = df_filtered["Fecha y hora"].dt.day
    df_filtered["Anio"] = df_filtered["Fecha y hora"].dt.year - 2023
    df_filtered["Mes"] = df_filtered["Fecha y hora"].dt.month
    
    return df_filtered


def catalog_excesses(df, pollutant, u_threshold):
    df["excess"] = df[pollutant] - u_threshold
    condition = (df["excess"] > 0)
    df["excess"] = df["excess"].where(condition)
    return df

    
def split_train_test(df, standardize=True, scaler=None):
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
        save_scaler = False
        if scaler is None:
            save_scaler = True
            scaler = StandardScaler()
            
        numerical_features = [
            'CO', 'NO', 'NO2', 'NOX', 'O3', 'PM10',
            'PM2.5', 'PRS', 'RAINF', 'RH', 'SO2', 'SR', 'TOUT', 'WSR', 'WDR',
            'Hora', 'Mes'
        ]   # 'excess' NO se estandariza: la GPD vive en las unidades originales del exceso (Y - u >= 0)
        augmented_features = []
        features = numerical_features + augmented_features

        df_train[features] = scaler.fit_transform(df_train[features])
        df_test[features] = scaler.transform(df_test[features])
        df_val[features] = scaler.transform(df_val[features])
        
        if save_scaler:
            with open("models/scaler.pkl", "wb") as outfile:
                pickle.dump(scaler, outfile)
                
        
    # Get excess dataframes
    target_cols = ["Estacion", "Fecha y hora", "excess"]
    
    df_train_ex = df_train[target_cols].copy()
    df_test_ex = df_test[target_cols].copy()
    df_val_ex = df_val[target_cols].copy()
    
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
            'Hora', 'Mes'
        ]
    )
    
    num_intervals = df_pivoted.index.nunique()
    num_stations = df_pivoted.columns.levels[1].nunique()
    num_features = NUM_VEC_FEATURES
    
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
        values="excess"
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
    data = data.reshape(NUM_VEC_FEATURES, 15, -1, 24)
    excesses = excesses.reshape(1, 15, -1, 24)

    data = data.permute([2, 0, 1, 3])
    excesses = excesses.permute([2, 0, 1, 3])
    
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
    df = get_data(filename)
    df = augment_w_resonances(df)
    df = catalog_excesses(df, pollutant, threshold)
    df_train, df_train_ex, df_test, df_test_ex, df_val, df_val_ex = split_train_test(
        df, standardize=True, scaler=None
    )
    
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


def main(pollutant=None):
    """Entrena el modelo de un contaminante: `python src/train.py SO2 _hourly` (por defecto O3)."""
    if pollutant is None:
            pollutant = sys.argv[1] if len(sys.argv) > 1 else "O3"
           
    if len(sys.argv) > 2:
        extra = sys.argv[2]
    else:
        extra = ""
        
    train_data, test_data, val_data = prepare_hourly_data(
        filename='data/datos_imputados.csv', 
        pollutant=pollutant, 
        threshold=U_THRESHOLDS[pollutant]
    )
    
    # CREATE MODEL
    model = Combiner(
        ds_params = params_dict["ds_params"],
        gcn_params = params_dict["gcn_params"],
        graph_params = params_dict["graph_params"],
    )
    # Arranca sigma ~ media de los excesos de train y normaliza los excesos de entrada al DeepSet
    model.init_from_train_data(train_data)
    
    # TRAIN THE MODEL (TrainConfig usa "cuda" si hay GPU disponible)
    train_config = TrainConfig()
    
    training_loop(
        model = model, 
        train_data = train_data,
        val_data = test_data, 
        D = haversine_matrix(STATION_COORDS), 
        config = train_config
    )
    
    # SAVE THE MODEL
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), f"models/model_{pollutant}{extra}.pth")


if __name__ == "__main__":
    main()
