import sys
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from sklearn.preprocessing import StandardScaler

try:
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict


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
    
    # Standarize
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
    
    return df_train, df_test, df_val
    

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


def main():
    # 1) GET THE DATA
    df_train, df_test, df_val = get_data('../data/datos_imputados.csv')
    
    # 2) REARRANGE THE DATA & TURN IT INTO ARRAYS
    train_arr = rearrange_data(df_train)
    test_arr = rearrange_data(df_test)
    val_arr = rearrange_data(df_val)
    
    # 3) SEND DATA TO TORCH (& DEVICE)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
        
    train_tensor = torch.tensor(train_arr, dtype=torch.float64, device=device)
    test_tensor = torch.tensor(test_arr, dtype=torch.float64, device=device)
    val_tensor = torch.tensor(val_arr, dtype=torch.float64, device=device)
    
    # 4) CREATE THE MODEL
    model = Combiner(
        ds_params=params_dict["ds_params"],
        gcn_params=params_dict["gcn_params"],
        graph_params=params_dict["graph_params"]
    )
    
    # 5) REARRANGE THE DATA FOR MODEL TRAINING
    
    # 5) TRAIN THE MODEL
    
    
    # 6) SAVE THE MODEL
    
    # (7) EVALUATE RESULTS
    
    pass


if __name__ == "__main__":
    main()