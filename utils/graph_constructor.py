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
        rows = []
        for i in self.num_hours:
            col = []
            for j in self.num_hours:
                if i == j:
                    col.append(A_small)
                elif i-1 == j or i+1 == j:
                    col.append(torch.eye(A_small.shape[0]))
                else:
                    col.append(torch.zeros_like(A_small))
            rows.append(torch.cat(col, dim=0))
        return torch.cat(rows, dim=1)
    

class Coeus(NormalizedLaplacianMatrixGenerator):
    """
    Connection between all adjacent nodes in time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in self.num_hours:
            col = []
            for j in self.num_hours:
                if i == j:
                    col.append(A_small)
                elif i-1 == j or i+1 == j:
                    col.append(A_small)
                else:
                    col.append(torch.zeros_like(A_small))
            rows.append(torch.cat(col, dim=0))
        return torch.cat(rows, dim=1)
    

class Crius(NormalizedLaplacianMatrixGenerator):
    """
    Weighted onnection between all adjacent nodes in time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.weight = nn.Parameter(torch.FloatTensor(0.5))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in self.num_hours:
            col = []
            for j in self.num_hours:
                if i == j:
                    col.append(A_small)
                elif i-1 == j or i+1 == j:
                    col.append(A_small * self.weight)
                else:
                    col.append(torch.zeros_like(A_small))
            rows.append(torch.cat(col, dim=0))
        return torch.cat(rows, dim=1)
    
    
class Hyperion(NormalizedLaplacianMatrixGenerator):
    """
    No connection between nodes in adjacent time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in self.num_hours:
            col = []
            for j in self.num_hours:
                if i == j:
                    col.append(A_small)
                else:
                    col.append(torch.zeros_like(A_small))
            rows.append(torch.cat(col, dim=0))
        return torch.cat(rows, dim=1)
    
    
class Iapetus(NormalizedLaplacianMatrixGenerator):
    """
    Random connection between nodes all through the time intervals.
    """
    def __init__(self, num_hours):
        super().__init__(num_hours)
        self.p = nn.Parameter(torch.FloatTensor(0.2))
        
    def compute_big_matrix(self, A_small: torch.Tensor) -> torch.Tensor:
        rows = []
        for i in self.num_hours:
            col = []
            for j in self.num_hours:
                if i == j:
                    col.append(A_small)
                else:
                    col.append(torch.zeros_like(A_small))
            rows.append(torch.cat(col, dim=0))
        big_matrix = torch.cat(rows, dim=1)
        
        mask = (torch.rand_like(big_matrix) < self.p)
        alterations = (2*torch.rand_like(big_matrix) - 1)/2
        big_matrix = torch.clamp(big_matrix[mask]+alterations[mask], min=0.0, max=1.0)
        return big_matrix