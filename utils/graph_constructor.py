import torch
from torch import nn

class NormalizedLaplacianMatrixGenerator(nn.Module):
    def __init__(self, num_hours):
        super().__init__()
        
        # Graph Construction Parameters
        self.num_hours = num_hours
        self.sigma_sq = nn.Parameter(torch.FloatTensor(1.0))
        self.epsilon = nn.Parameter(torch.FloatTensor(0.0))
        
    def compute_big_matrix(self, A_small) -> torch.Tensor:
        pass
                
    def forward(self, D: torch.Tensor):
        # Adjacency Matrix
        A_small = torch.exp( -D**2 / self.sigma_sq )
        mask = A_small <= self.epsilon
        A_small[mask] = 0.0
        
        # Compute big matrix
        A = self.compute_big_matrix(A_small) 
        
        # Laplacian matrix
        degree_vector = torch.sum(A, dim=1)
        D = torch.diag(degree_vector)
        L = D - A
        
        # Normalization
        zeta_max = torch.max(torch.linalg.eigvalsh(L))
        I_N = torch.eye(L.shape[0])
        
        return 2*L / zeta_max - I_N

    
class Oceanus(NormalizedLaplacianMatrixGenerator):
    """
    Only one connection between each node from different time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        diag_blocks = torch.kron(I_N, A_small)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
          + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(off_diag_mask, torch.eye(d, device=device, dtype=dtype))
        
        # Combine and return
        return diag_blocks + off_diag_blocks
    

class Coeus(NormalizedLaplacianMatrixGenerator):
    """
    Connection between all adjacent nodes in time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        diag_blocks = torch.kron(I_N, A_small)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
            + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(off_diag_mask, A_small)
        
        # Combine and return
        return diag_blocks + off_diag_blocks
    

class Crius(NormalizedLaplacianMatrixGenerator):
    """
    Weighted onnection between all adjacent nodes in time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.weight = nn.Parameter(torch.FloatTensor(0.5))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        diag_blocks = torch.kron(I_N, A_small)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
            + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(off_diag_mask, A_small*self.weight)
        
        # Combine and return
        return diag_blocks + off_diag_blocks
    
    
class Hyperion(NormalizedLaplacianMatrixGenerator):
    """
    No connection between nodes in adjacent time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        return torch.kron(I_N, A_small)
    
    
class Iapetus(NormalizedLaplacianMatrixGenerator):
    """
    Random connection between nodes all through the time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.p = nn.Parameter(torch.FloatTensor(0.2))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        big_matrix = torch.kron(I_N, A_small)
        
        # Affect random connections
        mask = (torch.rand_like(big_matrix) < self.p)
        alterations = (2*torch.rand_like(big_matrix) - 1)/2
        big_matrix = torch.clamp(big_matrix[mask]+alterations[mask], min=0.0, max=1.0)
        return big_matrix