import numpy as np
import torch
from scipy.optimize import minimize

try:
    from train import prepare_hourly_data
    
except ModuleNotFoundError:
    from src.train import prepare_hourly_data
    

U_THRESHOLDS = {
    "PM2.5": 38.0,
    "O3": 63.0,
    "SO2": 20.2
}


def gpd_nll(params, y, eps=1e-6):
    """Negative log-likelihood of the GPD for excesses y >= 0.
    params = (log_sigma, xi); log_sigma keeps sigma > 0 automatically."""
    log_sigma, xi = params[0], params[1]
    sigma = torch.exp(log_sigma)
    z = y / sigma

    if torch.abs(xi) < eps:                      # exponential limit (xi -> 0)
        return y.numel() * log_sigma + z.sum()

    t = 1.0 + xi * z
    if torch.any(t <= 0):                        # outside the support
        return torch.tensor(1e10, dtype=y.dtype)
    return y.numel() * log_sigma + (1.0 + 1.0 / xi) * torch.log(t).sum()


def fit_gpd_bfgs(excesses, u=None):
    """
    excesses : torch tensor of (x - u). NaN where x did not exceed u.
               (If you pass the full series x, pass u and it computes x - u.)
    Returns dict with sigma, xi, standard errors, and the optimizer result.
    """
    x = excesses.detach().double()
    if u is not None:                            # full data given instead of excesses
        x = x - u
    y = x[~torch.isnan(x)]                       # drop non-exceedances
    y = y[y >= 0]                                # safety: excesses must be >= 0
    n = y.numel()
    if n < 10:
        raise ValueError(f"Only {n} excesses; too few to fit a GPD.")

    # Method-of-moments starting values
    m, v = y.mean(), y.var()
    xi0 = 0.5 * (1.0 - m**2 / v)
    xi0 = float(torch.clamp(xi0, -0.4, 0.8))
    sigma0 = float(0.5 * m * (m**2 / v + 1.0))
    x0 = np.array([np.log(max(sigma0, 1e-8)), xi0])

    def fun_and_grad(p):
        p_t = torch.tensor(p, dtype=torch.float64, requires_grad=True)
        loss = gpd_nll(p_t, y)
        if loss.requires_grad:
            loss.backward()
            g = p_t.grad.numpy().copy()
        else:                                    # penalty branch
            g = np.zeros_like(p)
        return loss.item(), g

    res = minimize(fun_and_grad, x0, jac=True, method="BFGS",
                   options={"gtol": 1e-6, "maxiter": 500})

    log_sigma_hat, xi_hat = res.x
    sigma_hat = float(np.exp(log_sigma_hat))

    # Standard errors: BFGS inverse-Hessian approx (res.hess_inv) on the (log_sigma, xi) scale.
    # Delta method for sigma = exp(log_sigma). For accurate SEs use the exact Hessian (below).
    cov = res.hess_inv
    se_xi = float(np.sqrt(cov[1, 1]))
    se_sigma = float(sigma_hat * np.sqrt(cov[0, 0]))

    return dict(sigma=sigma_hat, xi=float(xi_hat), se_sigma=se_sigma,
                se_xi=se_xi, n_exceed=n, success=res.success, result=res)


def exact_hessian_cov(y, sigma, xi):
    """Optional: exact observed-information covariance via autograd."""
    p = torch.tensor([np.log(sigma), xi], dtype=torch.float64)
    H = torch.autograd.functional.hessian(lambda q: gpd_nll(q, y), p)
    return torch.linalg.inv(H)


def quick_test():
    # Quick test: simulate GPD(sigma=2, xi=0.2), threshold u = 5, with NaNs for non-exceedances
    from scipy.stats import genpareto
    torch.manual_seed(0)
    y_true = torch.tensor(genpareto.rvs(0.2, scale=2.0, size=2000, random_state=1))
    ex = y_true.clone()
    ex[::3] = float("nan")                      # simulate non-exceedances
    out = fit_gpd_bfgs(ex)
    print({k: v for k, v in out.items() if k != "result"})
    
    
def main():
    pollutant = "O3"
    # LOAD DATA
    train_data, test_data, val_data = prepare_hourly_data(
        'data/datos_imputados.csv', pollutant, U_THRESHOLDS[pollutant])
    
    ex = train_data["y"].to('cpu')
    
    print(ex.shape)
    out = fit_gpd_bfgs(ex)
    print({k: v for k, v in out.items() if k != "result"})

if __name__ == "__main__":
    main()