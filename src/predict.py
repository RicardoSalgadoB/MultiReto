import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import predict
    from utils.haversine_distance import haversine_matrix
    from src.train import prepare_hourly_data, STATION_COORDS
    
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from utils.model_architecture import Combiner
    from utils.model_params import params_dict
    from utils.train_utils import predict
    from utils.haversine_distance import haversine_matrix
    from src.train import prepare_hourly_data, STATION_COORDS
    

STATION_NAMES = [
    "SE",
    "NE",
    "CE",
    "CO",
    "SO",
    "NO2",
    "NTE",
    "NE2",
    "SE2",
    "S02",
    "SE3",
    "SUR",
    "NTE2",
    "NE3",
    "NO3",
]

    
def main():
    # LOAD DATA
    train_data_so2_hourly, test_data_so2_hourly, val_data_so2_hourly = prepare_hourly_data('data/datos_imputados.csv', "SO2", 20.2)
    
    # LOAD MODEL
    model_so2_hourly = Combiner(
        ds_params = params_dict["ds_params"],
        gcn_params = params_dict["gcn_params"],
        graph_params = params_dict["graph_params"],
    )
    model_so2_hourly.load_state_dict(torch.load("models/model_o3.pth", map_location="cpu"))
    
    # DISTANCES
    D = haversine_matrix(STATION_COORDS)
    
    # PREDICT
    predictions_so2_hourly = predict(model_so2_hourly, train_data_so2_hourly, D)
    
    # SAVE PREDICTIONS
    dates = pd.date_range(
        start="2023-01-02",
        periods=predictions_so2_hourly.shape[0],
        freq="D"
    )

    df = pd.DataFrame({
        "Fecha": np.repeat(dates, predictions_so2_hourly.shape[1]),
        "Estacion": np.tile(np.arange(predictions_so2_hourly.shape[1]), predictions_so2_hourly.shape[0]),
        "XI": predictions_so2_hourly[:, :, 0].ravel(),
        "SIGMA": predictions_so2_hourly[:, :, 1].ravel()
    })
    
    df["Estacion"] = df["Estacion"].replace(
        {i: STATION_NAMES[i] for i in range(len(STATION_NAMES))}
    )
    
    df.to_csv("data/predictions_o3.csv", index=False)
    
    
if __name__ == '__main__':
    main()