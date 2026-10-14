params_dict = {
   "ds_params": {
       "ndim": 1,
       "hdim": 8,
       "sodim": 8,
       "odim": 4,
       "nlayers": 1,
       "snlayers": 1
   },
   "gcn_params": {
       "in_features": 4 + 1 + 17,
       "hidden_dim": 32,
       "out_features": 2,
       "num_nodes": 24*15,  # num_intervals * num_locations
       "n_outputs": 15,
       "nlayers": 2,    # 3 capas GCN en total. Con 12 el modelo sobreajustaba y era inestable (ver chequeo)
       "use_bias": True
   },
   "graph_params": {
       "num_hours": 24      # num_intervals
   }
}