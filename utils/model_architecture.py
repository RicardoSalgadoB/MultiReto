import itertools
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils.haversine_distance import compute_adj_matrix
from utils.graph_constructor import Oceanus

class DeepSet(nn.Module):
    """Implementation of Deep Set.
    This implement the LOCAL FEATURE EXTRACTION MODULE
    
    Academically borrowed from Wilson et al. (2022) (https://github.com/TylerPWilson/deepGPD/blob/main/model.py).
    """
    def __init__(self, ndim=1, hdim=16, sodim=16, odim=8, nlayers=1, snlayers=1) -> None:
        """
        ndim: the dimension of each set element
        hdim: the hidden dimension of each element processing network
        sodim: dimension of the set representation
        odim: output dimension
        nlayers: number of layers of element processors
        snlayers: number of layers in set processors
        """
        super().__init__()
        
        self.ndim = ndim
        self.hdim = hdim
        self.sodim = sodim
        self.odim = odim
        self.nlayers = nlayers
        self.snlayers = snlayers
        
        # Middle layers of element processor
        elm_mids = list(itertools.chain(
            *[[nn.Linear(hdim, hdim), nn.ELU()] for i in range(self.nlayers)]
        ))
        # Element processing stack
        self.elements = nn.Sequential(
            nn.Linear(ndim, hdim),
            nn.ELU(),
            *elm_mids,
            nn.Linear(hdim, sodim)
        )
        
        # Middle layers for fully connected processor
        fc_mids = list(itertools.chain(
            *[[nn.Linear(sodim, sodim), nn.ELU()] for i in range(self.snlayers)]
        ))
        # Fully connected processor
        self.fc = nn.Sequential(
            nn.Linear(sodim, sodim),
            nn.ELU(),
            *fc_mids,
            nn.Linear(sodim, odim)
        )
        
    def forward(self, x):   # X-shape: (l*t, ndim_s)
        n_sets = x.shape[0]
        set_size = x.shape[1]
        
        # Store information on NaN inputs
        pre_mask = torch.isnan(x)
        out = x.reshape([n_sets * set_size, 1])     # (l * t * ndim_s)
        new_mask = ~torch.isnan(out)
        
        # Extract a representation of each set element
        new_out = torch.zeros([n_sets * set_size, self.sodim]).to(x.device) # (l * t * ndim_s, sodim)
        el_out = self.elements(out[new_mask][:, np.newaxis])
        new_out[ torch.squeeze(new_mask) ] += el_out
        out = new_out    # (l * t * ndim_s, sodim)
        
        # Construct an initial set representation for each set by averaging the representations of its elements
        mask = (~torch.isnan(out))
        out = out.reshape([n_sets, set_size, self.sodim])       # (l * t, ndim_s, sodim)
        mask = mask.reshape([n_sets, set_size, self.sodim])     # (l * t, ndim_s, sodim)
        counts = torch.sum( (~pre_mask)*1.0, dim=1 )
        counts[counts == 0] = 1
        sums = torch.sum(out, dim=1)                            # (l * t, sodim)
        avgs = sums / counts[:, np.newaxis]                     
        out = avgs                                              # (l * t, sodim)
        mask = (torch.sum(mask, dim=(1, 2)) > 0) * 1.0          # (l * t)
        
        # Use fully connected processor to compute higher lever features from initial set representation
        out = self.fc(out)                                      # (l * t, odim)
        return torch.squeeze(out, dim=1), mask[:, np.newaxis]   # (l * t, d_set) ; (l * t, 1)
    
    
class GCNLayer(nn.Module):
    """
    Implements a single layer of the Graph Convolutional Neural Network.
    
    Academically borrowed & modified from https://github.com/senadkurtisi/pytorch-GCN/tree/main
    """
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(1, out_features))))
        else:
            self.bias = None
            self.register_parameter('bias', None)
        # ELU (como en deepGPD) en lugar de ReLU: no se "mueren" las neuronas.
        self.activation = nn.ELU()
        # Conexion residual cuando las dimensiones coinciden (como en deepGPD): evita que la
        # señal se desvanezca al apilar muchas capas
        self.residual = (in_features == out_features)
        
        # Initialize weights    
        self.initialize_weights()
        
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        
    def forward(self, h_in, L):     # N: Num. nodes / M: Num. features
        h = h_in @ self.weight      # Shapes: (N, M_in) @ (M_in, M_out) -> (N, M_out)
        L = L.unsqueeze(0)
        h = L @ h   # (N, N) @ (N, M_out) -> (N, M_out)
        if self.bias is not None:
            h = h + self.bias       # (N, M_out) + (N, N)@(1, M_out) -> (N, M_out)
        h = self.activation(h)
        if self.residual:
            h = h + h_in
        return h
        
    
class GCN(nn.Module):
    """GCN implementation
    Academically borrowed from https://github.com/senadkurtisi/pytorch-GCN/tree/main
    """
    def __init__(
        self, 
        in_features, 
        hidden_dim, 
        out_features=2, 
        num_nodes=360,
        n_outputs=15,
        nlayers=24, 
        use_bias=True
    ) -> None:
        """
        in_features: d_set + d_mask + ndim_v
        hidden_dim: ...
        out_features: 2 for 2 ks per node
        nlayers: Default 24 for 24 hours in a day.
        use_bias: Defaults to true
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = in_features
            
        # Neural Network
        self.layers =  nn.ModuleList(
            [GCNLayer(in_features, hidden_dim, bias=use_bias)]
          + [GCNLayer(hidden_dim, hidden_dim, bias=use_bias) for i in range(nlayers)]
        )
        # Cabeza LINEAL (sin activacion), igual que la ultima capa del CNN de deepGPD: una ReLU
        # aqui deja las salidas k1,k2 >= 0 y puede "matar" la red.
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, out_features, bias=False),
        )
        self.readout = nn.Linear(num_nodes, n_outputs)
        # Sesgo por canal (k1 = log sigma, k2). El `readout` comparte su sesgo entre los 2 canales,
        # asi que este permite arrancar cada parametro en una escala razonable (ver Combiner.init_from_train_data)
        self.out_bias = nn.Parameter(torch.zeros(out_features))
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        
        
    def forward(self, h: torch.Tensor, L: torch.Tensor):
        """
        h: data for each layer (l * t, d_set + d_mask + ndim_v)
        D: distance matrix
        """
        L = L.type(torch.float32)
        for layer in self.layers:
            h = layer(h, L)
        h = self.node_head(h)
        h = self.readout(h.transpose(-1, -2)).transpose(-1, -2)
        return h + self.out_bias


class Constrainer(nn.Module):
    """Processes the output of NN to make sure it satisfies the generalized Pareto constraints.
    This implement the EXTREME VALUE MODELING MODULE.
    
    Academically borrowed from Wilson et al. (2022) (https://github.com/TylerPWilson/deepGPD/blob/main/model.py).

    sigma = exp(k1),  xi = exp(k2) - sigma / max_estacion   (deepGPD)

    Diferencias respecto a deepGPD, necesarias para que el entrenamiento no diverja:
      - k1 y k2 se acotan con tanh: exp() sin tope llegaba a 1e19 e inf/NaN.
      - xi se mantiene por encima de `xi_floor` (-1) con una softplus: para xi < -1 la
        verosimilitud GPD no esta acotada y la perdida "mejora" sin que el modelo sea bueno.
        La softplus solo SUBE xi, asi que 1 + xi*y/sigma > 0 se sigue cumpliendo.
      - `maxis` es un escalar por estacion (maximo de los excesos de entrenamiento, como en
        deepGPD), no un promedio de 1/max por hora.
    """
    def __init__(self, sigma_bound=7.0, xi_bound=2.0, xi_floor=-1.0, beta=20.0) -> None:
        super().__init__()
        self.sigma_bound = sigma_bound
        self.xi_bound = xi_bound
        self.xi_floor = xi_floor
        self.beta = beta
        
    def forward(self, all_ks, maxis=None):      # All ks shape: (B, L, 2)
        k1 = self.sigma_bound * torch.tanh(all_ks[..., 0:1] / self.sigma_bound)   # (B, L, 1)
        k2 = self.xi_bound * torch.tanh(all_ks[..., 1:2] / self.xi_bound)         # (B, L, 1)
        sigma = torch.exp(k1)

        L = all_ks.shape[1]
        m = maxis[0].reshape(L, -1).amax(dim=-1, keepdim=True)   # (L, 1) maximo por estacion
        xi = torch.exp(k2) - sigma / (m + 1e-6)                  # (B, L, 1)
        xi = self.xi_floor + F.softplus(xi - self.xi_floor, beta=self.beta)

        return torch.cat((xi, sigma), dim=-1)   # (B, L, 2)


class Combiner(nn.Module):
    """Graph Convolutional Neural Network for predicting GPD distributions parameters for each node/station.
    Combines each module into one pipeline.
    
    Academically borrowed from Wilson et al. (2022) (https://github.com/TylerPWilson/deepGPD/blob/main/model.py).
    """
    def __init__(self, graph_params: dict, ds_params: dict, gcn_params: dict) -> None:
        """
        ds_params (dict): Parameters for the DeepSet Network
        gcn_params (dict): Parameters for the Graph Convolutional Neural Network
        """
        super().__init__()
        
        # Create Graph Constructor
        self.graph_constructor = Oceanus(**graph_params)
        
        x = 64
        # Create MLP to modify freely the weights.
        self.sequential = nn.Sequential(
            nn.Linear(gcn_params["in_features"], x),
            nn.Dropout(), # para que no explote
            
            nn.Linear(x,x),
            nn.ReLU(),
            nn.Linear(x,x),
            nn.ReLU(),
            nn.Linear(x,x),
            nn.ReLU(),
            nn.Dropout(),
            
            nn.Linear(x, gcn_params["in_features"]),
            nn.Dropout()
        )

        # Create GCN
        self.gcn = GCN(**gcn_params)
        
        # Create DeepSet Network
        self.ds = DeepSet(**ds_params)
            
        # Create constrainer to enforce GPD constraints
        self.constrainer = Constrainer()
        
        # Escala (en unidades del contaminante) con la que se normalizan los excesos que entran
        # al DeepSet. Se fija con init_from_train_data y se guarda en el state_dict.
        self.register_buffer("set_scale", torch.tensor(1.0))
        
        # Save the final output for the model
        self.odim = gcn_params["out_features"]
        
    
    @torch.no_grad()
    def init_from_train_data(self, train_data: dict, xi0: float = 0.3) -> None:
        """Ajusta la escala de entrada y el punto de arranque de (sigma, xi) con los excesos de train.

        Los excesos y se dejan en unidades originales (ppb, ug/m3) porque la GPD vive en esas
        unidades. Aqui solo se arranca sigma ~ media de los excesos (en vez de exp(0)=1) y se
        normalizan los excesos que entran al DeepSet.
        """
        y = train_data["y"]
        v = y[torch.isfinite(y) & (y >= 0)]
        mean = float(v.mean()) if v.numel() > 0 else 1.0
        self.set_scale.fill_(mean)
        bias = torch.tensor([math.log(mean), math.log(xi0)], dtype=self.gcn.out_bias.dtype)
        self.gcn.out_bias.copy_(bias[: self.gcn.out_bias.numel()].to(self.gcn.out_bias.device))
    
    def forward(self, vecs: torch.Tensor, sets: torch.Tensor, D: torch.Tensor, maxis=None):
        """
        Args:
            vecs (torch.Tensor): Features per node per hour (shape: num. features, num. locations, num. hours)
            sets (torch.Tensor): Excesses in the last time window
            D (torch.Tensor): Distance matrix calculated with haversine distance
            maxis: ... . Defaults to None.

        Returns:
            torch.Tensor: GPD parameters for each location
        """
        # Save shape information    Assuming l to be location index and t to be time index
        if vecs is not None:
            bsize, ndim_v, l, t = vecs.shape    # Shape: (ndim_v, l, t)
        else:
            bsize, ndim_s, l, t = sets.shape   # Shape: (ndim_s, l, t) - ndim_s = 1
            
        # print(vecs[:2, :, 10, :5])
            
        # Compute set representations
        if sets is not None:
            ndim_s = sets.shape[1]
            sets = sets / self.set_scale                            # excesos en escala ~1 (NaN se conserva)
            sets = sets.reshape([bsize, ndim_s, l*t])               # (ndim_s, l*t)
            sets = sets.permute([0, 2, 1])                          
            sets = sets.reshape([-1, ndim_s])                       # (l*t, ndim_s)
            set_out, mask = self.ds(sets)                           # (l*t, d_set), (l*t, d_mask)
            set_out = torch.cat([set_out, mask], dim=1)             # (l*t, d_set + d_mask)
            cur_feat_dim = set_out.shape[1]
            set_out = set_out.reshape([bsize, l*t, cur_feat_dim])   # (l*t, d_set + d_mask)
            set_out = set_out.permute([0, 2, 1])
            set_out = set_out.reshape([bsize, -1, l, t])           # (d_set + d_mask, l, t)
            
        # Concatenate vector predictors and set representations
        if sets is not None and vecs is not None:
            out = torch.cat([set_out, vecs], dim=1)                 # (d_set + d_mask + ndim_v, l, t)
        elif vecs is not None:
            out = vecs
        else:
            out = sets
            
        # Construct graph
        L = self.graph_constructor(D)
            
        # Pass through GCN
        # print(out[:2, :, 10, :5])
        out_3d = out.reshape([bsize, -1, l*t])           # (d_set + d_mask + ndim_v, l * t)
        out_3d = out_3d.permute([0, 2, 1])                  # (l * t, d_set + d_mask + ndim_v) 
        
        out_3d = self.sequential(out_3d)
        
        out = self.gcn(out_3d, L)                  # (l * t, d_set + d_mask + ndim_v) -> (l, 2)
        # print(out[:2, 10, :])
        
        # Constrain output
        final_out = self.constrainer(out[:, :, :], maxis=maxis) # (l, 2)
        
        final_out = final_out.reshape([bsize, l, self.odim]) # (l, 2)
        return final_out