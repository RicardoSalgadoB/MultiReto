import itertools
import numpy as np
import torch
from torch import nn

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
        self.snlayers = nlayers
        
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
        self.activation = nn.Sigmoid()
        
        # Initialize weights    
        self.initialize_weights()
        
    
    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        
        
    def forward(self, h, L):        # N: Num. nodes / M: Num. features
        h = h @ self.weight         # Shapes: (N, M_in) @ (M_in, M_out) -> (N, M_out)
        h = torch.sparse.mm(L, h)   # (N, N) @ (N, M_out) -> (N, M_out)
        if self.bias is not None:
            h += self.bias          # (N, M_out) + (N, N)@(1, M_out) -> (N, M_out)
        return self.activation(h)
        
    
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
            
        # Graph Construction Parameters
        self.sigma_sq = nn.Parameter(torch.FloatTensor(1.0))
        self.epsilon = nn.Parameter(torch.FloatTensor(0.0))
            
        # Neural Network
        self.layers =  nn.ModuleList(
            [GCNLayer(in_features, hidden_dim, bias=use_bias)]
          + [GCNLayer(hidden_dim, hidden_dim, bias=use_bias) for i in range(nlayers)]
        )
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, out_features, bias=False),
            nn.Sigmoid(),
        )
        self.readout = nn.Linear(num_nodes, n_outputs)
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = in_features
        
        
    def forward(self, h: torch.Tensor, L: torch.Tensor):
        """
        h: data for each layer (l * t, d_set + d_mask + ndim_v)
        D: distance matrix
        """
        for layer in self.layers:
            h = layer(h, L)
        h = self.node_head(h)
        h = self.readout(h.transpose(-1, -2)).transpose(-1, -2)
        return h


class Constrainer(nn.Module):
    """Processes the output of NN to make sure it satisfies the generalized Pareto constraints.
    This implement the EXTREME VALUE MODELING MODULE.
    
    Academically borrowed from Wilson et al. (2022) (https://github.com/TylerPWilson/deepGPD/blob/main/model.py).
    """
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, all_ks, maxis=None):      # All ks shape: (l*t, 2)
        # Could be vectorized
        outs = list()
        for batch in range(all_ks.shape[0]):
            ks = all_ks[batch, : , :]                               # (l*t, 2)
            k1, k2 = torch.exp(ks[:, 0]), torch.exp(ks[:, 1]) 
            sigma = k1[:, np.newaxis]
            k2 = k2[:, np.newaxis]
            xi = k2 - sigma / (maxis[batch, :, np.newaxis] + 1e6)
            outs.append(torch.cat((xi, sigma), axis=1))
        return torch.stack(outs, dim=0)
    

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

        # Create GCN
        self.gcn = GCN(**gcn_params)
        
        # Create DeepSet Network
        self.ds = DeepSet(**ds_params)
            
        # Create constrainer to enforce GPD constraints
        self.constrainer = Constrainer()
        
        # Save the final output for the model
        self.odim = gcn_params["out_features"]
        
    
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
            
        # Compute set representations
        if sets is not None:
            ndim_s = sets.shape[1]
            sets = sets.reshape([bsize, ndim_s, l*t])               # (ndim_s, l*t)
            sets = sets.permute([0, 2, 1])                          
            sets = sets.reshape([-1, ndim_s])                       # (l*t, ndim_s)
            set_out, mask = self.ds(sets)                           # (l*t, d_set), (l*t, d_mask)
            set_out = torch.cat([set_out, mask], dim=1)             # (l*t, d_set + d_mask)
            cur_feat_dim = set_out.shape[1]
            set_out = set_out.reshape([bsize, l*t, cur_feat_dim])   # (l*t, d_set + d_mask)
            set_out = sets.permute([0, 2, 1])
            set_out = sets.reshape([bsize, ndim_s, l, t])           # (d_set + d_mask, l, t)
            
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
        out.reshape([bsize, -1, l*t])           # (d_set + d_mask + ndim_v, l * t)
        out.permute([0, 2, 1])                  # (l * t, d_set + d_mask + ndim_v) 
        out = self.gcn(out, L)                  # (l * t, d_set + d_mask + ndim_v) -> (l, 2)

        out_dim = out.shape[2]                      # 2 - error here maybe
        out = out.reshape([bsize, out_dim, l*t])    # (2, l)
        out = out.permute([0, 2, 1])                # (l, 2)
        
        # Constrain output
        final_out = self.constrainer(out[:, :, :], maxis=maxis) # (l, 2)
        
        final_out = final_out.reshape([bsize, l, self.odim]) # (l, 2)
        return final_out