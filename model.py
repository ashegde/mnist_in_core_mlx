import math
from typing import Callable
import mlx.core as mx

def newton_schulz(A: mx.array, n_steps: int = 15, eps: float = 1e-9) -> mx.array:
  """
  Rectangular Newton-Schulz Iteration

  iterate: X = 1.5 X - 0.5 XX'X
  """
  # normalize to ensure singular values of A are
  # contained in between 0 and \sqrt{3}.
  X = A / (eps + mx.linalg.norm(A, ord = 'fro', axis=(-2,-1)))

  for _ in range(n_steps):
      #XXt = mx.einsum('...ij,...kj->...ik', X, X)
      #X = 1.5 * X - 0.5 * mx.einsum('...ij,...jk->...ik', XXt, X)
      XXt = mx.matmul(X, X.T)
      X = 1.5 * X - 0.5 * mx.matmul(XXt, X)
  return X

def stiefel(gain: float = 1.0, n_steps: int = 15,  dtype: mx.Dtype = mx.float32) -> Callable[[mx.array], mx.array]:
    # syntax based on https://github.com/ml-explore/mlx/blob/main/python/mlx/nn/init.py
    '''
    Generates a uniform random m-by-k matrix X from the Stiefel manifold V_{k,m} (X'X = I_{k-by-k}).
    Note, this implies that (a) the k m-dimensional columns of X are orthonormal and (b) m >= k -- i.e.,
    nrows >= ncols.

    This function implements the following method based on Theorem 2.2.1 of

    Chikuse, Y. (2003). Statistics on special manifolds (Vol. 174). Springer Science & Business Media.
    
    Part (iii) of this theorem states that if X is uniformly distributed on V_{k,m}, it can be expressed as X = Z(Z'Z)^{-0.5} where Z
    is an m-by-k matrix with entries independent and identically distributed as N(0,1). 

    If we take the SVD of Z: Z = USV', we can rewrite X as X = UV'. Thus, we can sample X
    by generating Z and then use Newton-Schulz iteration to drive the singular values to unity. 
    Specifically, we use the Newton-Schultz iteration suggested in,

    Bernstein, J., & Newhouse, L. (2024). Modular Duality in Deep Learning. arXiv preprint arXiv:2410.21265.
    '''

    def initializer(a: mx.array) -> mx.array:
        if a.ndim < 2:
            raise ValueError("Only arrays with 2 or more dimensions are supported")

        if a.size == 0:
            # do nothing
            return a

        # flattened dims
        nrows = a.shape[0]
        ncols = a.size // nrows
        flattened = mx.random.normal(shape=(nrows, ncols), scale=1.0, loc=0.0, dtype=dtype)
        if nrows < ncols: # ensures m >= k 
            flattened = flattened.T
        
        # it seems that linalg.qr only works on cpu
        x = newton_schulz(flattened, n_steps)

        if nrows < ncols:
            x = x.T
            
        return gain * x

    return initializer

def build_params(layer_sizes: list[int] = [784, 512, 512, 10]):
    def random_layer_params(in_dim: int, out_dim: int, initializer) -> tuple[mx.array, mx.array]:
        return initializer(mx.zeros((in_dim, out_dim))), mx.zeros((out_dim,))

    init_fn = stiefel()
    
    params = [
        random_layer_params(d_in, d_out, init_fn)
        for d_in, d_out in zip(layer_sizes[:-1], layer_sizes[1:])
    ]
    return params

def predict_single(params: list[mx.array], flattened_image: mx.array):
    # per-example predictions

    def relu(z: mx.array) -> mx.array:
        return mx.maximum(z, 0.0)
    
    x = flattened_image
    # forward
    for w, b in params[:-1]:
        outs = mx.matmul(x, w) + b
        x = relu(outs)

    final_w, final_b = params[-1]
    logits = mx.matmul(x, final_w) + final_b
    return logits - mx.logsumexp(logits) # log(softmax probs)