"""
This implementation is based on the work "improving Fractal Pre-training"
by Conner Anderson and Ryan Farrell (arXiv: 2110.03091)
Official repository: https://github.com/catalys1/fractal-pretraining
"""

from functools import partial

from cv2 import cvtColor, COLOR_HSV2RGB
import numba
import numpy as np


def sample_svs(n, a, rng=None):
    '''Sample singular values. 2*`n` singular values are sampled such that the following conditions
    are satisfied, for singular values sv_{i} and i = 0, ..., 2n-1:
    
    1. 0 <= sv_{i} <= 1
    2. sv_{2i} >= sv_{2i+1}
    3. w.T @ S = `a`, for S = [sv_{0}, ..., sv_{2n-1}] and w = [1, 2, ..., 1, 2]
    
    Args:
        n (int): number of pairs of singular values to sample.
        a (float): constraint on the weighted sum of all singular values. Note that a must be in the
            range (0, 3*n).
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
            
    Returns:
        Numpy array of shape (n, 2) containing the singular values.
    '''
    if rng is None: rng = np.random.default_rng()
    if a < 0: a == 0
    elif a > 3*n: a == 3*n
    s = np.empty((n, 2))
    p = a
    q = a - 3*n + 3
    # sample the first 2*(n-1) singular values (first n-1 pairs)
    for i in range(n - 1):
        s1 = rng.uniform(max(0, q/3), min(1, p))
        q -= s1
        p -= s1
        s2 = rng.uniform(max(0, q/2), min(s1, p/2))
        q = q - 2 * s2 + 3
        p -= 2 * s2
        s[i, :] = s1, s2
    # sample the last pair of singular values
    s2 = rng.uniform(max(0, (p-1)/2), p/3)
    s1 = p - 2*s2
    s[-1, :] = s1, s2
    
    return s


def sample_system(n=None, constrain=True, bval=1, rng=None, beta=None, sample_fn=None):
    '''Return n random affine transforms. If constrain=True, enforce the transforms
    to be strictly contractive (by forcing singular values to be less than 1).
    
    Args:
        n (Union[range,Tuple[int,int],List[int,int],None]): range of values to sample from for the number of
            transforms to sample. If None (default), then sample from range(2, 8).
        constrain (bool): if True, enforce contractivity of transformations. Technically, an IFS must be
            contractive; however, FractalDB does not enforce it during search, so it is left as an option here.
            Default: True.
        bval (Union[int,float]): maximum magnitude of the translation parameters sampled for each transform.
            The translation parameters don't effect contractivity, and so can be chosen arbitrarily. Ignored and set
            to 1 when constrain is False. Default: 1.
        rng (Optional[numpy.random._generator.Generator]): random number generator. If None (default), it defaults
            to np.random.default_rng().
        beta (float or Tuple[float, float]): range for weighted sum of singular values when constrain==True. Let 
            q ~ U(beta[0], beta[1]), then we enforce $\sum_{i=0}^{n-1} (s^i_1 + 2*s^i_2) = q$.
        sample_fn (callable): function used for sampling singular values. Should accept three arguments: n, for
            the size of the system; a, for the sigma-factor; and rng, the random generator. When None (default),
            uses sample_svs.
    
    Returns:
        Numpy array of shape (n, 2, 3), containing n sets of 2x3 affine transformation matrices.
        '''
    if rng is None:
        rng = np.random.default_rng()
    if n is None:
        n = rng.integers(2, 8)
    elif isinstance(n, range):
        n = rng.integers(n.start, n.stop)
    elif isinstance(n, (tuple, list)):
        n = rng.integers(*n)
        
    if beta is None:
        beta = ((5 + n) / 2, (6 + n) / 2)

    if sample_fn is None:
        sample_fn = sample_svs
        
    if constrain:
        # sample a matrix with singular values < 1 (a contraction)
        # 1. sample the singular vectors--random orthonormal matrices--by randomly rotating the standard basis
        base = np.sign(rng.random((2*n, 2, 1)) - 0.5) * np.eye(2)
        angle = rng.uniform(-np.pi, np.pi, 2*n)
        ss = np.sin(angle)
        cc = np.cos(angle)
        rmat = np.empty((2 * n, 2, 2))
        rmat[:, 0, 0] = cc
        rmat[:, 0, 1] = -ss
        rmat[:, 1, 0] = ss
        rmat[:, 1, 1] = cc
        uv = rmat @ base
        u, v = uv[:n], uv[n:]
        # 2. sample the singular values
        a = rng.uniform(*beta)
        print("Sampled sigma-factor :", a)
        s = sample_fn(n, a, rng)
        # 3. sample the translation parameters from Uniform(-bval, bval) and create the transformation matrix
        m = np.empty((n, 2, 3))
        m[:, :, :2] = u * s[:, None, :] @ v
        m[:, :, 2] = rng.uniform(-bval, bval, (n, 2))
    else:
        m = rng.uniform(-1, 1, (n, 2, 3))

    return m, a
