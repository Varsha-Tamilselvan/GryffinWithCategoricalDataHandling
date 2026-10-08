"""
optimizer.py
------------
Top-level `SimpleGryffin` class tying together:

    data_processing.ParameterSpace   (scaling 7 continuous inputs)
    chimera.Chimera                  (5 targets -> 1 scalar merit)
    surrogate.KernelDensitySurrogate (Bayesian kernel-density model)
    acquisition.optimize_acquisition (Adam-based acquisition search)

Usage pattern (matches the recommend/observe loop of real Gryffin):

    opt = SimpleGryffin(parameters, objectives, general)

    observations = []                       # or your historical data
    for round in range(budget):
        proposals = opt.recommend(observations, num_proposals=2)
        for p in proposals:
            result = run_your_experiment(p)      # -> dict of 5 targets
            p.update(result)
            observations.append(p)

`observations` is always the FULL accumulated history; the optimizer
itself is stateless between calls, exactly like the real Gryffin API.
"""

import numpy as np

from data_processing import ParameterSpace, clean_observations, objectives_to_array
from chimera import Chimera
from surrogate import KernelDensitySurrogate
from acquisition import optimize_acquisition, select_diverse_batch


DEFAULT_GENERAL = {
    "sampling_strategies": 2,   # number of lambda values / proposals per round
    "num_random_samples": 2000,  # global random search pool size
    "num_restarts": 10,          # how many of those get Adam-refined
    "adam_lr": 0.05,
    "adam_steps": 60,
    "obj_transform": "sqrt",
    "min_rel_dist": 0.05,        # diversity floor between proposals in one batch
    "random_seed": None,
    "verbose": True,
}


class SimpleGryffin:
    def __init__(self, parameters, objectives, general=None):
        """
        Parameters
        ----------
        parameters : list of dict
            [{'name': 'x1', 'low': 0.0, 'high': 1.0}, ... ]  (7 entries)
        objectives : list of dict, IN PRIORITY ORDER (index 0 = most
            important for Chimera's hierarchy)
            [{'name': 'y1', 'goal': 'max', 'tolerance': 0.2, 'absolute': False},
             ...]                                             (5 entries)
        general : dict, optional
            Overrides for DEFAULT_GENERAL.
        """
        self.space = ParameterSpace(parameters)
        self.param_names = self.space.names

        self.obj_names = [o["name"] for o in objectives]
        goals = [o["goal"] for o in objectives]
        tolerances = [o.get("tolerance", 0.2) for o in objectives]
        absolutes = [o.get("absolute", False) for o in objectives]
        self.chimera = Chimera(goals, tolerances, absolutes)

        self.general = dict(DEFAULT_GENERAL)
        if general:
            self.general.update(general)

        self._rng = np.random.default_rng(self.general["random_seed"])

        self.last_merit_ = None     # scalarized merit of the last recommend() call's training data
        self.last_surrogate_ = None

    # ------------------------------------------------------------------
    def _bootstrap(self, num_proposals):
        """Pure random sampling, used when there is no data yet.
        Every discrete/discrete_set/categorical block is drawn from
        its legal rows only -- never an arbitrary float, and never an
        invalid partial one-hot code for a categorical parameter."""
        d = self.space.encoded_dim
        U = self._rng.uniform(size=(num_proposals, d))
        for blk in self.space.blocks():
            idx, cands = blk["indices"], blk["candidates"]
            rows = cands[self._rng.integers(len(cands), size=num_proposals)]
            U[:, idx] = rows
        return [self.space.decode_vector(u) for u in U]

    # ------------------------------------------------------------------
    def recommend(self, observations, num_proposals=None, clean=True):
        """
        Parameters
        ----------
        observations : list of dict
            Full accumulated history. Each dict must contain every
            parameter name and every objective name as numeric keys.
            Pass an empty list on the very first call (cold start).
        num_proposals : int, optional
            How many new parameter sets to propose this round. Defaults
            to general['sampling_strategies'].
        clean : bool
            Run `data_processing.clean_observations` first (drops rows
            with missing/non-numeric/NaN values and exact duplicate
            parameter vectors).

        Returns
        -------
        proposals : list of dict
            Each dict has the 7 parameter names as keys, in REAL units
            (already un-scaled from the internal unit hypercube).
        """
        n_strat = num_proposals or self.general["sampling_strategies"]

        if not observations:
            if self.general["verbose"]:
                print(f"[recommend] no observations yet -> {n_strat} random bootstrap point(s)")
            return self._bootstrap(n_strat)

        if clean:
            observations = clean_observations(
                observations, self.space, self.obj_names,
                verbose=self.general["verbose"],
            )
            if not observations:
                if self.general["verbose"]:
                    print("[recommend] nothing left after cleaning -> bootstrap")
                return self._bootstrap(n_strat)

        # encode_dict already scales continuous/discrete coords to
        # [0,1] AND one-hot-encodes categorical ones in the same step
        U = self.space.encode_observations(observations)
        Y_raw = objectives_to_array(observations, self.obj_names)

        merit = self.chimera.scalarize_and_transform(Y_raw, transform=self.general["obj_transform"])
        self.last_merit_ = merit

        surrogate = KernelDensitySurrogate().fit(U, merit)
        self.last_surrogate_ = surrogate

        blocks = self.space.blocks()  # one entry per discrete_range/discrete_set/categorical parameter

        lambdas = np.linspace(1.0, -1.0, n_strat)  # +1 exploit ... -1 explore
        cand_x, cand_val = [], []
        for lam in lambdas:
            x, val = optimize_acquisition(
                surrogate, lam, self.space.encoded_dim,
                num_random_samples=self.general["num_random_samples"],
                num_restarts=self.general["num_restarts"],
                adam_lr=self.general["adam_lr"],
                adam_steps=self.general["adam_steps"],
                blocks=blocks,
                rng=self._rng,
            )
            cand_x.append(x)
            cand_val.append(val)

        kept_idx = select_diverse_batch(cand_x, cand_val, min_rel_dist=self.general["min_rel_dist"])
        U_new = [cand_x[i] for i in kept_idx]

        if self.general["verbose"]:
            best_seen = observations[int(np.argmin(merit))]
            print(f"[recommend] {len(observations)} obs -> proposing {len(U_new)} point(s); "
                  f"best merit so far = {merit.min():.4f} at { {k: round(best_seen[k], 4) for k in self.obj_names} }")

        return [self.space.decode_vector(u) for u in U_new]

    # ------------------------------------------------------------------
    def predict_merit(self, param_dicts):
        """
        Convenience helper: evaluate the current surrogate's predicted
        merit (not the acquisition function) for a list of parameter
        dicts, using whatever surrogate was built on the last
        `recommend()` call. Useful for `gryffin.get_regression_surrogate`
        style inspection.
        """
        if self.last_surrogate_ is None:
            raise RuntimeError("call recommend() at least once before predict_merit()")
        U = self.space.encode_observations(param_dicts)
        return self.last_surrogate_.predict(U)
