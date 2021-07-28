"""
Demonstrate sparse recovery phenomena following
Donoho-Candes-Romberg-Tao "DCRT"

1. Construct a sparse N-dimensional signal X_sparse that has S-many nonzero entries
2. Based on N and S, choose a "sampling rate" K [here is where DCRT comes in]
3. Define K-indices_to_sample: random but fixed set of K-many indices
4. Attempt to "reconstruct" X_sparse via the following optimization objective for x in R^N:
    i) FFT(x) restricted to K-indices_to_sample  = FFT(X_sparse) restricted to K-indices_to_sample
    ii) L1 norm of x is minimal


Reference:
    Candes, ICM 2006,
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.419.4969&rep=rep1&type=pdf


What not to do:
    X_nonsparse = X_sparse[X_sparse > 0]
    fft_x_nonsparse = fft(X_nonsparse)
    initial_guess = set_signal(N=len(X_nonsparse))
    initial_guess = initial_guess.reshape(len(X_nonsparse),)
    loss_fn = lambda x: np.linalg.norm(x=(fft(a=x) - fft_x_nonsparse), ord=2) + np.linalg.norm(x=x, ord=1)

    min_result_object = minimize(fun=loss_fn, x0=initial_guess)
    min_result_soln = min_result_object['x']

"""

import numpy as np
from scipy.fft import dct
import cvxpy as cp
from matplotlib import pyplot as plt


def set_signal(low=-1, high=1, N=10):
    """Return a bounded uniform random vector in R^N."""
    s = np.random.uniform(low=low, high=high, size=(N, 1))
    return s

def sparsify_signal(s, n_sparse=1):
    """Set n_sparse elements of s to zero."""
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    indices_to_zero = np.random.choice(a=range(len(s)), size=n_sparse, replace=False)
    s[indices_to_zero] = 0  # we all love numpy
    return s

def get_nonzero_indices(s):
    if not isinstance(s, np.ndarray):
        s = np.array(s)
    return np.nonzero(s)


# PARAMS
N_dim = 512  # real-dim of where original signal lives
S_dim = 30  # support size: num non-zero entries in original signal
null_dim = N_dim - S_dim
K_samples = 60  # alternative: S_sparse * log(N_dim), for some log base...
delta = 1e-5  # lower bound nonzero solution terms; used in post-opt analysis

# INITIALIZE
X = set_signal(N=N_dim)  # don't reconstruct this, but sparse version of it
num_zeros = N_dim - S_dim
X_sparse = sparsify_signal(s=X, n_sparse=null_dim)  # signal to reconstruct
dft_matrix_N = dct(x=np.eye(N=N_dim), axis=0)
dft_matrix_K = dct(x=np.eye(N=K_samples), axis=0)


# PROJECT AND SAMPLE
Y = dft_matrix_N @ X_sparse  # projection of sparse signal to cosine fourier basis
sample_indices = np.random.choice(range(N_dim), K_samples, replace=False)
# Y_sample = Y[sample_indices]


# Optimize
x = cp.Variable(shape=N_dim)  # initialize; will sample K-many: given by sample_indices

prob = cp.Problem(cp.Minimize(cp.norm(x, p=1)),
                 [(dft_matrix_N @ x) [sample_indices.reshape(K_samples,1)] == Y[sample_indices]])
prob.solve()
print("status: {}".format(prob.status))
optimal_solution =  prob.value
optimal_x = x.value.reshape(N_dim, 1)

# Number of nonzero elements in the solution (its cardinality or diversity).
nnz_l1 = (np.absolute(x.value) > delta).sum()
E = np.isclose(optimal_x, X_sparse)
E_support = np.isclose(optimal_x[sample_indices], X_sparse[sample_indices])
error_rate = 1 - (E.sum() / N_dim)
S_error_rate = 1 - (E_support.sum() / K_samples)
print('Found a feasible x in R^{} that has {} nonzeros.'.format(N_dim, nnz_l1))
print('Reconstruction error rate is %0.2f' % error_rate)
print('Support Reconstruction error rate is %0.2f' % S_error_rate)



plt.plot(X_sparse, '.', color='b')
plt.plot(x.value, '.', color='g')





