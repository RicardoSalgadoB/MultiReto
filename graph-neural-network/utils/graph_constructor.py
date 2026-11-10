import torch
from torch import nn

class NormalizedLaplacianMatrixGenerator(nn.Module):
    def __init__(self, num_hours):
        super().__init__()
        
        # Graph Construction Parameters
        self.num_hours = num_hours
        self.epsilon = nn.Parameter(torch.tensor(0.0))
        
    def compute_big_matrix(self, A_small) -> torch.Tensor:
        pass
                
    def forward(self, D: torch.Tensor):
        # Adjacency Matrix
        sigma_sq = torch.median( D**2 )
        A_small = torch.exp( -D**2 / sigma_sq )
        mask = A_small <= self.epsilon
        A_small = A_small.masked_fill(mask, 0.0)
        
        # Compute big matrix
        A = self.compute_big_matrix(A_small) 
        
        # Laplacian matrix
        degree_vector = torch.sum(A, dim=1)
        D_matrix = torch.diag(degree_vector)
        L = D_matrix - A
        
        # Normalization
        zeta_max = torch.max(torch.linalg.eigvalsh(L))
        I_N = torch.eye(L.shape[0], dtype=L.dtype, device=L.device)
        
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
        I_d = torch.eye(d, device=device, dtype=dtype)
        diag_blocks = torch.kron(A_small, I_N)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
          + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(I_d, off_diag_mask)
        
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
        diag_blocks = torch.kron(A_small, I_N)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
            + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(A_small, off_diag_mask)
        
        # Combine and return
        return diag_blocks + off_diag_blocks
    

class Crius(NormalizedLaplacianMatrixGenerator):
    """
    Weighted connection between all adjacent nodes in time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.weight = nn.Parameter(torch.tensor(0.5))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        diag_blocks = torch.kron(A_small, I_N)
        
        # Off-diagonal
        off_diag_mask = (
            torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=1)
            + torch.diag(torch.ones(N-1, device=device, dtype=dtype), diagonal=-1)
        )
        off_diag_blocks = torch.kron(A_small*self.weight, off_diag_mask)
        
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
        return torch.kron(A_small, I_N)
    
    
class Iapetus(NormalizedLaplacianMatrixGenerator):
    """
    Random connection between nodes all through the time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.p = nn.Parameter(torch.tensor(0.2))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        N = self.num_hours
        d = A_small.shape[0]
        device, dtype = A_small.device, A_small.dtype
        
        # Main diagonal
        I_N = torch.eye(N, device=device, dtype=dtype)
        big_matrix = torch.kron(A_small, I_N)
        
        # Affect random connections
        mask = (torch.rand_like(big_matrix) < self.p)
        alterations = (2*torch.rand_like(big_matrix) - 1)/2
        big_matrix = torch.clamp(big_matrix+alterations.where(mask, 0.0), min=0.0, max=1.0)
        return big_matrix