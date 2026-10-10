import torch


def haversine_matrix(
    coords: torch.Tensor,       # Shape: (N, 2)
    radius: float = 6378.137    # By Default, it is Earth's radius
) -> torch.Tensor:
    """Computes haversine distance matrix for all pairs inside coords.

    Args:
        coords: The coordinates of each point. 
        radius: Defaults to 6378.137 (Earth's radius)

    Returns:
        torch.Tensor: Matrix of haversine distances
    """
    rads = torch.deg2rad(coords)
    lat1, lon1 = rads[:, 0:1], rads[:, 1:2]  # Shapes: (N, 1)
    lat2, lon2 = rads[:, 0], rads[:, 1]      # Shapes: (1*, N)
    
    dlat = lat2 - lat1  # (N, N)
    dlon = lon1 - lon2  # (N, N)
    
    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2.0) ** 2
    d = 2.0 * radius * torch.arcsin( torch.clamp(torch.sqrt(a), max = 1.0) )    # Prevent rounding errors by clamping
    
    return d


def compute_adj_matrix(
    D: torch.Tensor,
    sigma_sq: float = 1.0,
    epsilon: float|None = None
):
    """Calculates the weighted adjacency matrix.

    Args:
        D (torch.Tensor): Distance matrix between coordinates
        sigma_sq (float, optional): Defaults to 1.0.
        epsilon (float | None, optional): Defaults to None.
    """
    A = torch.exp( -D**2 / sigma_sq )
    
    if epsilon:
        mask = A <= epsilon
        A[mask] = 0.0
    
    return A


if __name__ == "__main__":
    coords = torch.tensor([
        [4.01, 2.51],
        [4.02, 2.52],
        [4.03, 2.53],
        [4.04, 2.54]
    ])
    
    D = haversine_matrix(coords)
    print(D)
    print(D**2)
    
    A = compute_adj_matrix(D, sigma_sq=float((D**2).mean()))
    print(A)