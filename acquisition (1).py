"""
acquisition.py
--------------
Optimizes the Phoenics/Gryffin acquisition function

    alpha(x) = (sum_k f_k p_k(x) + lam * p_uniform) /
               (sum_k p_k(x) + p_uniform)

(to be MINIMIZED, per Eq. 1 of Hickman et al. 2022 and Eq. 12 of
Hase et al. 2018) using the same two-stage strategy described in those
papers' Methods sections:

    1. Global search: draw many points uniformly at random over the
       unit hypercube and evaluate alpha on all of them cheaply
       (vectorized).
    2. Local refinement: take the best few candidates and refine them
       with a first-order gradient method. The Gryffin papers use Adam
       by default (Kingma & Ba, 2017) for continuous parameters, so we
       implement Adam from scratch here using the exact analytic
       gradient supplied by `surrogate.value_and_grad_acquisition`.

A simple distance-based diversity penalty (Sec. "sample selector" of
the Gryffin source, and the self-avoidance behaviour discussed in the
known-constraints paper, Fig. S3) is applied when multiple proposals
are requested, so that a batch of recommendations does not collapse
onto nearly the same point.
"""

import numpy as np


def adam_minimize(value_and_grad_fn, x0, lr=0.05, steps=60,
                   beta1=0.9, beta2=0.999, eps=1e-8, bounds=(0.0, 1.0),
                   blocks=None, batch_value_fn=None, max_block_eval=25, rng=None):
    """
    Minimal from-scratch Adam optimizer (Kingma & Ba, 2017) operating
    on a single point x in R^d, with simple clipping back into
    `bounds` after every step (the internal space is always the unit
    hypercube / one-hot-extended unit hypercube).

    `value_and_grad_fn(x) -> (value, grad)` must return both the
    scalar objective value and its gradient at x.

    blocks
    ------
    Output of `ParameterSpace.blocks()`: one entry per non-continuous
    parameter, `{"indices": [...], "candidates": ndarray(n_options, width)}`.
      - a numeric discrete parameter (discrete_range/discrete_set)
        is a block of width 1 -- `indices` is a single coordinate,
        `candidates` is a column of its legal scaled values.
      - a categorical parameter is a block of width = n_categories --
        `indices` are ALL of its one-hot coordinates, `candidates` is
        the identity matrix (each row = one category's one-hot code).
    Gradient descent has no meaning on these coordinates: there is
    nothing "between" category A and category C to move to, and
    nothing between a one-hot code of [1,0,0] and [0,1,0] either. So,
    following Gryffin's own `NaiveDiscreteOptimizer`/
    `NaiveCategoricalOptimizer` (`GradientOptimizer._single_opt_iteration`)
    in spirit, every block's coordinates are excluded from the Adam
    update entirely (their gradient components are zeroed before the
    step).

    How a block's own value gets picked each iteration, though, is an
    improvement over Gryffin's original single-random-guess-per-step:
    `batch_value_fn`, when given, lets us score EVERY legal candidate
    row for a block in one vectorized call (cheap -- these candidate
    lists are small) and just keep the best one outright, rather than
    trying one random row and hoping it's an improvement. With
    `max_block_eval` candidates or fewer this is a full exhaustive
    search of the block every iteration; for a larger discrete_range
    it evaluates a random subset of that size (plus the current value)
    instead of the whole range, to keep the cost bounded. If
    `batch_value_fn` is not given, this falls back to the original
    single-random-try-and-keep-if-better behaviour.

    Returns
    -------
    best_x : ndarray
    best_val : float
        The best (lowest) value and position encountered along the
        trajectory (not necessarily the final iterate), matching the
        "keep the best observed sample" behaviour used for acquisition
        optimization in the Gryffin/Phoenics source.
    """
    rng = rng or np.random.default_rng()
    blocks = blocks or []
    block_dims = [i for blk in blocks for i in blk["indices"]]  # every non-continuous coordinate

    x = np.array(x0, dtype=float)
    # make sure the starting point is already legal on every block
    for blk in blocks:
        idx, cands = blk["indices"], blk["candidates"]
        x[idx] = cands[rng.integers(len(cands))]

    m = np.zeros_like(x)
    v = np.zeros_like(x)
    lo, hi = bounds

    best_x = x.copy()
    best_val, _ = value_and_grad_fn(x)

    def refine_block(x, val, blk):
        """Set this block's coordinates to whichever legal row scores
        best, given every other coordinate held fixed at its current
        value. Returns the (possibly unchanged) x and its new value."""
        idx, cands = blk["indices"], blk["candidates"]
        if len(cands) <= 1:
            return x, val

        if batch_value_fn is None:
            # fallback: one random guess, keep only if it improves
            current = x[idx].copy()
            x[idx] = cands[rng.integers(len(cands))]
            cand_val, _ = value_and_grad_fn(x)
            if cand_val < val:
                return x, cand_val
            x[idx] = current
            return x, val

        # exhaustive (or capped random-subset) vectorized search
        if len(cands) > max_block_eval:
            sub_idx = rng.choice(len(cands), size=max_block_eval, replace=False)
            rows = cands[sub_idx]
        else:
            rows = cands

        trials = np.tile(x, (len(rows), 1))
        trials[:, idx] = rows
        vals = batch_value_fn(trials)
        best_i = int(np.argmin(vals))
        if vals[best_i] < val:
            x[idx] = rows[best_i]
            return x, float(vals[best_i])
        return x, val

    for t in range(1, steps + 1):
        val, grad = value_and_grad_fn(x)

        # (a) continuous Adam step -- non-continuous coordinates excluded from the gradient
        grad = grad.copy()
        for i in block_dims:
            grad[i] = 0.0

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x = x - lr * m_hat / (np.sqrt(v_hat) + eps)
        x = np.clip(x, lo, hi)
        val, _ = value_and_grad_fn(x)

        # (b) best-of-candidates search per block (discrete or categorical)
        for blk in blocks:
            x, val = refine_block(x, val, blk)

        if val < best_val:
            best_val = val
            best_x = x.copy()

    return best_x, best_val


def optimize_acquisition(surrogate, lam, dim, num_random_samples=2000,
                          num_restarts=10, adam_lr=0.05, adam_steps=60,
                          blocks=None, rng=None):
    """
    Full two-stage acquisition optimization for one sampling strategy
    (one value of lambda): random global search -> Adam local refine.

    `dim` is the full ENCODED width (continuous/discrete coords count
    1 each, a categorical parameter counts once per category -- see
    `ParameterSpace.encoded_dim`). `blocks` (see `adam_minimize`) is
    `ParameterSpace.blocks()`: when given, the random global search
    also only samples legal rows on those block coordinates (never an
    arbitrary float, and never an invalid one-hot code), and the
    refinement stage scores every legal row of each block with
    `surrogate.alpha_batch` (vectorized, so it's cheap) and keeps the
    best one outright -- instead of a gradient step, and instead of a
    single random guess.

    Returns
    -------
    best_x : ndarray, shape (dim,)   in the encoded unit space
    best_val : float                 acquisition value at best_x
    """
    rng = rng or np.random.default_rng()
    blocks = blocks or []

    # Stage 1: global random search, vectorized -- but snap every
    # block onto a legal row so we never propose e.g. media=13.7 or
    # an invalid partial one-hot code
    Xr = rng.uniform(size=(num_random_samples, dim))
    for blk in blocks:
        idx, cands = blk["indices"], blk["candidates"]
        rows = cands[rng.integers(len(cands), size=num_random_samples)]
        Xr[:, idx] = rows

    alphas = surrogate.alpha_batch(Xr, lam)
    order = np.argsort(alphas)
    top_idx = order[: min(num_restarts, num_random_samples)]

    # Stage 2: local refinement from each of the top candidates --
    # continuous dims via Adam, discrete/categorical blocks via greedy swap
    best_x, best_val = None, np.inf
    for i in top_idx:
        x0 = Xr[i]
        xr, val = adam_minimize(
            lambda x: surrogate.value_and_grad_acquisition(x, lam),
            x0, lr=adam_lr, steps=adam_steps, bounds=(0.0, 1.0),
            blocks=blocks, batch_value_fn=lambda Xb: surrogate.alpha_batch(Xb, lam), rng=rng,
        )
        if val < best_val:
            best_val = val
            best_x = xr

    return best_x, best_val


def select_diverse_batch(candidates, candidate_vals, min_rel_dist=0.05):
    """
    Greedy diversity filter for a batch of proposed points (one per
    sampling strategy). If two candidates are closer than
    `min_rel_dist` (as a fraction of the unit-hypercube diagonal),
    the one with the worse (higher) acquisition value is nudged away
    by re-drawing it is NOT done here to keep things simple and
    deterministic; instead we just flag the collision so the caller
    can decide what to do (e.g. increase sampling_strategies).

    This mirrors the self-avoidance behaviour Gryffin exhibits natively
    through its sample selector (see known-constraints paper, Fig. S3)
    -- the kernel density already pushes new points away from existing
    observations, so this helper's job is only to avoid two *proposals
    in the same batch* landing on top of each other.

    Returns
    -------
    kept_idx : list of int
        Indices (into `candidates`) that are mutually far enough apart.
    """
    candidates = np.asarray(candidates, dtype=float)
    n = len(candidates)
    if n <= 1:
        return list(range(n))

    order = np.argsort(candidate_vals)  # best (lowest alpha) first
    kept = []
    for idx in order:
        x = candidates[idx]
        too_close = False
        for kidx in kept:
            if np.linalg.norm(x - candidates[kidx]) < min_rel_dist:
                too_close = True
                break
        if not too_close:
            kept.append(idx)

    if not kept:
        kept = [order[0]]
    return kept
