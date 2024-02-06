#%%
import torch
import dcor
from numba import njit
import numpy as np
from baselines.pecnet.utils import energy_score_scipy
from npsn.utils import energy_score_pytorch

debug = False

@njit('f8[:](f8[:,:,:,:], f8[:,:,:,:], f8, f8, f8)', cache = False, )
def energy_score_wrapped(y, x, K=1, d=2, b=1):
    """ This implementation only takes the temporal aspects into account but also
        does not separate the spatial dimensions (flatten out together).
    """
    # K=1
    K=int(K)
    d=int(d)
    b=int(b)
    n_samples = x.shape[0]
    n_scenarios = x.shape[1]
    n_horizon = x.shape[2]
    n_dims = x.shape[3]
    M = n_scenarios
    # K = np.abs(K)
    # K = M if K >= M else K
    K = M
    es = np.empty((n_samples,))
    for s in range(n_samples):
        #
        ed = 0
        for j in range(M):
            # print(y[s,0].shape)
            # print((x[s,j]-y[s,0]).shape)
            # Equivalent to the Frobenius distance
            ed += (np.sum(np.abs(x[s,j]-y[s,0])**d)**(1/d))**b
        ed/=M
        #
        ei=0
        for j in range(M):
            for k in range(K):
                ei += (np.sum(np.abs((x[s,j]-x[s,(j+k+1)%M]))**d)**(1/d))**b
        ei /= M*M
        es[s] = ed - 0.5 * ei
    return es

def energy_score(y, x, K=1, d=2, b=1):
    return energy_score_wrapped(y, x, K=K, d=d, b=b)


#%% generate random data
n_samples = 20
n_temporal = 12
n_spatial = 2
d = 2 # n_temporal * n_samples
y = torch.randn(n_temporal, n_spatial, dtype=torch.float64) # gt
x = torch.randn(n_samples, n_temporal, n_spatial, dtype=torch.float64) # pred
print(y.shape)
print(x.shape)

#%% Pytorch implementation (for sgcn and stgat)
# ed = torch.cdist(x.reshape(n_samples,-1), y.reshape(1,-1), p=d)
# ei = torch.pdist(x.reshape(20, -1), p=d)
# adjust_factor = (n_samples-1) / n_samples
# es_pytorch = ed.mean() - 0.5 * adjust_factor * ei.mean()
# print(ed.shape, ei.shape) if debug else None
# print(ed.mean(), ei.mean()) if debug else None
# print(es_pytorch)

energy_score_pytorch(y[:,np.newaxis], x[:,:,np.newaxis], d=d)

#%% Dcor implementation
es_dcor = 0.5*dcor.energy_distance(y.reshape(1,-1), x.reshape(n_samples,-1))
print(es_dcor)

# %% Numba (myown) implementation
es_numba = energy_score(y.numpy().astype('float64').reshape(1, 1, n_temporal, n_spatial), 
                        x.numpy().astype('float64').reshape(1, n_samples, n_temporal, n_spatial),
                        K=n_samples,
                        d=d,
                        b=1
                        )
print(es_numba)

#%% scipy implementation (for pecnet)
print(y.shape)
print(x.shape)
energy_score_scipy(y[np.newaxis], x[:,np.newaxis], d=d)
