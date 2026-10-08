"""
chimera.py
----------
From-scratch implementation of Chimera, the hierarchy-based achievement
scalarizing function (ASF) from:

    Hase, Roch, Aspuru-Guzik, "Chimera: enabling hierarchy based
    multi-objective optimization for self-driving laboratories",
    Chem. Sci., 2018, 9, 7642.

Chimera turns n objectives, ranked in a user-defined priority order,
into a single scalar merit to be *minimized*. Objective 0 is the most
important; lower-priority objectives are only allowed to improve as
long as they do not degrade a higher-priority objective beyond its
tolerance.

Implements Eqs. 1-5 of the paper (discrete/Heaviside variant, which is
exact and needs no smoothing parameter since it is only ever evaluated
on the finite set of *observed* points, not inside a gradient).
"""

import numpy as np


class Chimera:
    """
    Parameters
    ----------
    goals : list of {'min', 'max'}, length n_objectives
        Ordered by priority: goals[0] is the most important objective.
    tolerances : list of float, length n_objectives
        Meaning depends on `absolutes[k]`:
          - absolute=False (default): relative tolerance, i.e. the
            fraction of the *observed range* of objective k (computed
            only within the region where higher-priority objectives
            already satisfy their own tolerance) that objective k is
            allowed to give up. Typical values: 0.1 - 0.3.
          - absolute=True: `tolerances[k]` is treated as an absolute
            threshold value on objective k itself, given in the
            objective's *original* units and *original* min/max sense
            (e.g. "yield must stay above 90%" -> goal='max',
            tolerance=0.9, absolute=True).
    absolutes : list of bool, optional
        See above. Defaults to all False.
    """

    def __init__(self, goals, tolerances, absolutes=None):
        self.n_obj = len(goals)
        if len(tolerances) != self.n_obj:
            raise ValueError("tolerances must have the same length as goals")
        self.goals = list(goals)
        self.tolerances = np.array(tolerances, dtype=float)
        self.absolutes = list(absolutes) if absolutes is not None else [False] * self.n_obj
        if len(self.absolutes) != self.n_obj:
            raise ValueError("absolutes must have the same length as goals")
        for g in self.goals:
            if g not in ("min", "max"):
                raise ValueError("goal must be 'min' or 'max'")

    # ------------------------------------------------------------------
    def _to_minimization(self, objs):
        """Flip the sign of every 'max' objective so everything is a
        minimization problem internally."""
        signs = np.array([1.0 if g == "min" else -1.0 for g in self.goals])
        return objs * signs[None, :], signs

    # ------------------------------------------------------------------
    def scalarize(self, objs):
        """
        Compute the raw (un-normalized) Chimera merit for every row of
        `objs`.

        Parameters
        ----------
        objs : array-like, shape (n_samples, n_objectives)
            Objective values in their *original* units and order
            (column k corresponds to goals[k] / tolerances[k]).

        Returns
        -------
        chi : ndarray, shape (n_samples,)
            Lower is always better (this is a minimization merit),
            regardless of the individual objectives' goals.
        """
        objs = np.asarray(objs, dtype=float)
        n, m = objs.shape
        if m != self.n_obj:
            raise ValueError(f"expected {self.n_obj} objective columns, got {m}")

        F, signs = self._to_minimization(objs)

        f_tol = np.zeros(m)
        f_min = np.zeros(m)
        mask = np.ones(n, dtype=bool)          # Y_{-1} = entire dataset
        region_masks = []

        for k in range(m):
            region_masks.append(mask.copy())
            fk_region = F[mask, k]

            if fk_region.size == 0:
                # nothing satisfies the higher-priority tolerances;
                # objective k becomes irrelevant from here on
                f_tol[k] = np.inf
                continue

            if self.absolutes[k]:
                # tolerance given directly, in original units/sense ->
                # convert to the internal (minimization) sign
                f_tol[k] = self.tolerances[k] * signs[k]
            else:
                lo, hi = fk_region.min(), fk_region.max()
                f_tol[k] = lo + self.tolerances[k] * (hi - lo)

            # update the region for the *next* objective: points that
            # still satisfy objective k's tolerance
            mask = mask & (F[:, k] <= f_tol[k])

        # shift parameters f_min_k = min of objective k within the
        # region where objective k itself was evaluated (Y_k)
        for k in range(m):
            reg = region_masks[k] if region_masks[k].any() else np.ones(n, dtype=bool)
            f_min[k] = F[reg, k].min()

        # Heaviside indicators: Theta_plus[i, k] = 1 if point i
        # satisfies objective k's tolerance
        Theta_plus = (F <= f_tol[None, :]).astype(float)
        Theta_minus = 1.0 - Theta_plus

        # Eq. 5 of the Chimera paper
        chi = F[:, 0] * Theta_plus[:, 0]
        prod_all_minus = np.prod(Theta_minus, axis=1)
        chi = chi + (F[:, 0] - f_min[m - 1]) * prod_all_minus
        for k in range(1, m):
            prefix_minus = np.prod(Theta_minus[:, :k], axis=1)
            chi = chi + (F[:, k] - f_min[k - 1]) * Theta_plus[:, k] * prefix_minus

        return chi

    # ------------------------------------------------------------------
    def scalarize_and_transform(self, objs, transform="sqrt"):
        """
        Scalarize, min-max normalize to [0, 1], and apply a monotonic
        transform that sharpens the region around the optimum (this
        mirrors Gryffin's default `obj_transform`).

        Returns
        -------
        merit : ndarray, shape (n_samples,)
            Values in [0, 1]; lower is better. Feed this directly to
            the surrogate as the scalar target.
        """
        chi = self.scalarize(objs)
        lo, hi = chi.min(), chi.max()
        if hi > lo:
            chi_norm = (chi - lo) / (hi - lo)
        else:
            chi_norm = np.zeros_like(chi)

        if transform == "sqrt":
            return np.sqrt(chi_norm)
        elif transform == "cbrt":
            return np.cbrt(chi_norm)
        elif transform == "square":
            return chi_norm ** 2
        elif transform is None or transform == "identity":
            return chi_norm
        else:
            raise ValueError(f"unknown transform '{transform}'")
