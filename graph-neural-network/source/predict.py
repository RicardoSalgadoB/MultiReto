import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import genpareto

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
    "CE",
    "NE",
    "NE2",
    "NE3",
    "NO",
    "NO2",
    "NO3",
    "NTE",
    "NTE2",
    "SE",
    "SE2",
    "SE3",
    "SO",
    "SO2",
    "SUR",
]

U_THRESHOLDS = {
    "PM2.5": 38.0,
    "O3": 63.0,
    "SO2": 20.2
}

X_THRESHOLDS = {
    "PM2.5": 3.0,
    "O3": 27.0,
    "SO2": 54.8
}

    
def main(pollutant=None):
    """Genera (xi, sigma) por dia y estacion: `python src/predict.py SO2 _hourly` (por defecto O3)."""
    if pollutant is None:
        pollutant = sys.argv[1] if len(sys.argv) > 1 else "O3"
       
    if len(sys.argv) > 2:
        extra = sys.argv[2]
    else:
        extra = ""
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # LOAD DATA
    train_data, test_data, val_data = prepare_hourly_data(
        'data/datos_imputados.csv', pollutant, U_THRESHOLDS[pollutant])
    
    # LOAD MODEL
    model = Combiner(
        ds_params = params_dict["ds_params"],
        gcn_params = params_dict["gcn_params"],
        graph_params = params_dict["graph_params"],
    )
    model.load_state_dict(torch.load(f"models/model_{pollutant}.pth", map_location="cpu"))
    
    # DISTANCES
    D = haversine_matrix(STATION_COORDS)
    
    # PREDICT
    predictions = predict(model, train_data, D, device=device)
    
    # SAVE PREDICTIONS
    dates = pd.date_range(
        start="2023-01-02",
        periods=predictions.shape[0],
        freq="D"
    )

    df = pd.DataFrame({
        "Fecha": np.repeat(dates, predictions.shape[1]),
        "Estacion": np.tile(np.arange(predictions.shape[1]), predictions.shape[0]),
        "XI": predictions[:, :, 0].ravel(),
        "SIGMA": predictions[:, :, 1].ravel()
    })
    
    df["Estacion"] = df["Estacion"].replace(
        {i: STATION_NAMES[i] for i in range(len(STATION_NAMES))}
    )
    
    df["prob_over_threshold"] = 1 - genpareto.cdf(X_THRESHOLDS[pollutant], c = df["XI"], scale = df["SIGMA"])
    
    print(f"data/predictions_{pollutant}{extra}.csv")
    df.to_csv(f"data/predictions_{pollutant}{extra}.csv", index=False)
    
    
if __name__ == '__main__':
    main()