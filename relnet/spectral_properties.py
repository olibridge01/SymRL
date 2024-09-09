import numpy as np
import scipy as sp

def get_laplacian(A):
    degs = A.sum(axis=1).reshape(-1)
    D = np.diagflat(degs)
    L = D - A
    return L

def compute_fiedler_vector(L, v0=None):
    _, eigvs = sp.sparse.linalg.eigsh(L, k=2, which='SM', return_eigenvectors=True, v0=v0)
    fiedler_vector = eigvs[:, 1]
    return fiedler_vector

def get_pseudoinverse(M):
    return np.linalg.pinv(M)