"""
surrogate.py
------------
From-scratch kernel-density Bayesian surrogate, following the
mathematical formulation in:

    Hase, Roch, Kreisbeck, Aspuru-Guzik, "Phoenics: A Universal Deep
    Bayesian Optimizer", ACS Cent. Sci. 2018, 4, 1134.
    (precision schedule tau_n ~ n^2, Sec. VI.C of the SI)

    Hase, Aldeghi, Hickman, Roch, Aspuru-Guzik, "Gryffin: An algorithm
    for Bayesian optimization of categorical variables informed by
    expert knowledge", Appl. Phys. Rev. 2021, 8, 031406.

WHAT THIS IS AND WHAT IT DEVIATES FROM
---------------------------------------
In the original Phoenics/Gryffin code, a small Bayesian Neural Network
(trained with NUTS/variational inference, in PyMC3/PyTorch) is used to
place a Gaussian kernel *location* near each observation and to sample
a *precision* from a Gamma(alpha=12 n^2, beta=1) prior, every time
recommend() is called. That BNN is an autoencoder-like regularizer: for
purely continuous/numeric-discrete parameters, its kernel locations
converge essentially to the observations themselves, and its role
reduces to (a) placing one Gaussian kernel per observation and (b)
picking the precision from that schedule.

This class is dimension-agnostic: it just computes squared Euclidean
distance on whatever `X` it is given. That is exactly what makes it
work correctly for categorical parameters too, PROVIDED the caller
(`data_processing.ParameterSpace`) one-hot encodes them first: the
squared Euclidean distance between any two distinct one-hot rows is
always exactly 2 regardless of which two categories they are, and 0
for the same category -- i.e. plain Euclidean distance on a one-hot
block already gives every pair of categories equal, unordered
distance, with no special-casing needed here. This mirrors Gryffin's
own "naive" (descriptor-free) categorical kernel.

This module reproduces that *outcome* directly and deterministically
with plain NumPy, without needing PyMC3/PyTorch or an MCMC sampler:
  - one isotropic Gaussian kernel centered at each observation
  - precision tau_n = 12 * n^2 (the posterior *mean* of the paper's
    Gamma(12 n^2, 1) prior), which the Phoenics SI (Sec. VI.C) reports
    as the best-performing schedule among those tested
  - the kernel-regression surrogate of Eq. 11 (Phoenics) / Eq. 1
    (Hickman et al. known-constraints paper)

This is an exact, faithful, pure-NumPy stand-in for the "naive"
(descriptor-free) Gryffin surrogate on purely continuous spaces, and
it has the major practical advantage of being fully analytically
differentiable, which lets the acquisition optimizer in acquisition.py
use exact gradients (Adam) instead of finite differences.
"""

import numpy as np


class KernelDensitySurrogate:
    """
    Isotropic Gaussian kernel-density regression surrogate on the unit
    hypercube [0, 1]^d.

    Usage
    -----
        surrogate = KernelDensitySurrogate()
        surrogate.fit(U, merit)     # U: (n, d) in [0,1]^d, merit: (n,)
        mu = surrogate.predict(U_query)
    """

    def __init__(self, alpha_coeff=12.0, min_precision=1.0):
        self.alpha_coeff = alpha_coeff
        self.min_precision = min_precision
        self.X = None          # (n, d) observed points, unit cube
        self.y = None          # (n,)   scalarized merit, lower=better
        self.tau = None        # scalar precision
        self.d = None

    # ------------------------------------------------------------------
    def fit(self, X, y):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        self.X = X
        self.y = y
        self.d = X.shape[1]
        n = X.shape[0]
        # precision schedule: tau_n ~ 12 n^2 (Phoenics SI, Sec. VI.C)
        self.tau = max(self.alpha_coeff * n ** 2, self.min_precision)
        return self

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if self.X is None:
            raise RuntimeError("call .fit(X, y) before using the surrogate")

    # ------------------------------------------------------------------
    def _kernels(self, Xq):
        """
        Returns
        -------
        p : ndarray, shape (m, n)   kernel density of each query point
                                     with respect to each observation
        diff : ndarray, shape (m, n, d)   (x_query - x_observation)
        """
        self._check_fitted()
        Xq = np.atleast_2d(Xq)
        diff = Xq[:, None, :] - self.X[None, :, :]          # (m, n, d)
        sqdist = np.sum(diff ** 2, axis=2)                  # (m, n)
        norm_const = (self.tau / (2.0 * np.pi)) ** (self.d / 2.0)
        p = norm_const * np.exp(-0.5 * self.tau * sqdist)   # (m, n)
        return p, diff

    # ------------------------------------------------------------------
    def predict(self, Xq):
        """
        Kernel-regression prediction of the scalarized merit at query
        points Xq (Eq. 11 of the Phoenics paper, lambda=0 case).
        """
        p, _ = self._kernels(Xq)
        num = p @ self.y
        den = np.clip(p.sum(axis=1), 1e-300, None)
        return num / den

    # ------------------------------------------------------------------
    def alpha_batch(self, Xq, lam, p_uniform=1.0):
        """
        Vectorized acquisition function value (Eq. 12, Phoenics /
        Eq. 1, Hickman et al.) at a batch of query points.

            alpha(x) = (sum_k f_k p_k(x) + lam * p_uniform) /
                       (sum_k p_k(x) + p_uniform)

        lam > 0 biases toward exploitation, lam < 0 toward exploration,
        lam = 0 recovers the plain surrogate prediction.
        """
        p, _ = self._kernels(Xq)
        num = p @ self.y + lam * p_uniform
        den = p.sum(axis=1) + p_uniform
        return num / den

    # ------------------------------------------------------------------
    def value_and_grad_acquisition(self, x, lam, p_uniform=1.0):
        """
        Acquisition value AND its exact analytic gradient at a single
        point x (shape (d,)), used by the Adam optimizer in
        acquisition.py.
        """
        Xq = np.asarray(x, dtype=float).reshape(1, -1)
        p, diff = self._kernels(Xq)
        p = p[0]            # (n,)
        diff = diff[0]      # (n, d)  = x - X_k

        y = self.y
        N = np.sum(y * p) + lam * p_uniform
        D = np.sum(p) + p_uniform
        alpha = N / D

        # d p_k/dx = p_k * (-tau) * (x - X_k)
        dp = -self.tau * diff * p[:, None]        # (n, d)
        dN = np.sum(y[:, None] * dp, axis=0)      # lam*p_uniform term has zero gradient
        dD = np.sum(dp, axis=0)
        grad = (dN * D - N * dD) / (D ** 2)
        return float(alpha), grad
