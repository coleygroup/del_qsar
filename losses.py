import torch
import torch.nn as nn
import numpy as np

def get_zstats(R, k1, k2, n1, n2, method='sqrt'):
    d = n2 / n1
    R_d = R / d
    
    if method == 'score':
        zstat = (k1 - k2 * R_d) / torch.sqrt((k1 + k2) * R_d)
    elif method == 'wald':
        zstat = (k1 - k2 * R_d) / torch.sqrt(k1 + k2 * R_d**2)
    elif method == 'sqrt':
        zstat =  2 *(torch.sqrt(k1 + 3/8.) - torch.sqrt((k2 + 3/8.) * R_d))
        zstat = zstat / torch.sqrt(1 + R_d)  
    else:
        raise NotImplementedError
    return zstat

def loss_fn_nlogprob(R, k1, k2, n1, n2, method='sqrt', zscale=1):
    """Return total negative log probability
    
    CDF approximation from https://pytorch.org/docs/stable/_modules/torch/distributions/normal.html#Normal.cdf"""
    zstat = get_zstats(R, k1, k2, n1, n2, method=method)
    zstat = torch.clamp(zstat / zscale, -5, +5)
    zstat = torch.abs(zstat)
##     cdf = 0.5 * (1 + torch.erf(zstat / np.sqrt(2)))
    sf = 1 - torch.erf(zstat / np.sqrt(2)) # 2*(1-cdf)
    loss = -torch.log(sf)
    return loss

mse_loss = nn.MSELoss(reduction='none')
def loss_fn_MSE(R, k1, k2, n1, n2, zscale=0):
    """Return mean squared error
    
    R_target calculated with same formula used in R_from_z in enrichments.py, with z=0"""
    a = np.power(0, 2) / 4 - (k2 + 3/8)
    b = 2 * torch.sqrt(k1 + 3/8) * torch.sqrt(k2 + 3/8)
    c = np.power(0, 2) / 4 - (k1 + 3/8)
    x = (-b) / (2*a)
    R_target = torch.pow(x, 2) * n2/n1
    return mse_loss(R, R_target)

bce_loss = nn.BCEWithLogitsLoss(reduction='none')
def loss_fn_BCE(preds, true_labels):
    """Return binary cross entropy loss"""
    return bce_loss(preds, true_labels)
    
if __name__ == '__main__': 
    # Test w/ expected values: z scores are 0, +2, -2
    assert ((np.array(get_zstats(
            torch.FloatTensor((2.8604651162790704, 1.128009901001709, 9.570765609202377)),
            torch.FloatTensor([15, 15, 15]),
            torch.FloatTensor([5, 5, 5]),
            1e6,
            1e6,
        )) - np.array([0, 2, -2]))**2).sum() < 1e-3

    # Expects tensor(-6.1801) (probability values are 1, 0.0455, 0.0455)
    assert np.abs(float(loss_fn_nlogprob(
            torch.FloatTensor((2.8604651162790704, 1.128009901001709, 9.570765609202377)),
            torch.FloatTensor([15, 15, 15]),
            torch.FloatTensor([5, 5, 5]),
            1e6,
            1e6,
        )) - 6.1801) < 1e-3
    