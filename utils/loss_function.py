import numpy as np

def theo_max(xi, sigma):
    """
    Computes the upper bound of the GPD's support given shape and scale parameters
    """
    out = np.zeros_like(xi) + 9_999_999_999
    np.putmask(out, xi < 0, -sigma/xi)
    return out


def gp_penalized(raw_samples, xi, sigma):
    """ 
    Computes Negative Log-Likelihood assuming generalized pareto distribution
    NaNs are ignored
    Samples exceeding support of GPD are penalized
    """
    if "torch" in str(type(xi)):
        xi = xi.cpu().detach().numpy()
    if "torch" in str(type(sigma)):
        sigma = sigma.cpu().detach().numpy()
    if "torch" in str(type(raw_samples)):
        samples = raw_samples.cpu().detach().numpy()
    else:
        samples = raw_samples
        
    theo_maxes = theo_max(xi, sigma)
    nan_mask = ~np.isnan(samples)
    mask = (samples <= theo_maxes)
    cur_samples = samples
    log_thing = 1 + xi * cur_samples / sigma
    
    log_thing[(~mask) & (nan_mask)] = 1e-6
    out = np.log(sigma) + ( 1 + 1/xi) * np.log(log_thing)
    return np.nanmean(out[nan_mask]), np.sum( (~mask) & (nan_mask) ) / np.sum(nan_mask)