import numpy as np

from typing import Tuple


def m_step(x: np.ndarray, es: np.ndarray, ess: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    mu, sigma, pie = m_step(x,es,ess)

    Inputs:
    -----------------
           x: shape (n, d) data matrix
          es: shape (n, k) E_q[s]
         ess: shape (k, k) sum over data points of E_q[ss'] (n, k, k)
                           if E_q[ss'] is provided, the sum over n is done for you.

    Outputs:
    --------
          mu: shape (d, k) matrix of means in p(y|{s_i},mu,sigma)
       sigma: shape (,)    standard deviation in same
         pie: shape (1, k) vector of parameters specifying generative distribution for s
    """
    n, d = x.shape
    if es.shape[0] != n:
        raise TypeError('es must have the same number of rows as x')
    k = es.shape[1]
    if ess.shape == (n, k, k):
        ess = np.sum(ess, axis=0)
    if ess.shape != (k, k):
        raise TypeError('ess must be square and have the same number of columns as es')

    mu = np.dot(np.dot(np.linalg.inv(ess), es.T), x).T
    sigma = np.sqrt((np.trace(np.dot(x.T, x)) + np.trace(np.dot(np.dot(mu.T, mu), ess))
                     - 2 * np.trace(np.dot(np.dot(es.T, x), mu))) / (n * d))
    pie = np.mean(es, axis=0, keepdims=True)
    
    return mu, sigma, pie
