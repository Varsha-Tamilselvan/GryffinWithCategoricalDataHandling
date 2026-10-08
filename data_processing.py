"""
data_processing.py
-------------------
Cleaning of raw experimental records and scaling/encoding of input
parameters into the internal representation the surrogate and
acquisition function operate on.

Each parameter has a 'kind' (default 'continuous'):

  - 'continuous'     : {'name', 'low', 'high'}
                       any real value in [low, high].
  - 'discrete_range' : {'name', 'low', 'high'}
                       any INTEGER in [low, high] (consecutive, step 1)
                       -- e.g. "number of plates: 5 to 20". Mirrors
                       Gryffin's own 'discrete' type, which internally
                       expands to options = arange(low, high + 1).
  - 'discrete_set'   : {'name', 'options'}
                       only the specific NUMERIC values listed are
                       legal, not necessarily consecutive -- e.g.
                       "concentration: one of 0.5, 1.0, 2.5". Distance
                       between two values is their numeric difference
                       (12 is closer to 10 than to 15).
  - 'categorical'    : {'name', 'options'}
                       only the specific LABELS listed are legal, and
                       they are UNORDERED -- e.g. "production media:
                       one of 'A', 'B', 'C'" (labels need not be
                       numbers; even if they are numbers, e.g. media
                       IDs 10/12/15, no numeric closeness is assumed
                       between them). Encoded as one-hot, so every
                       pair of distinct categories is equally distant
                       -- this is the same "naive"/descriptor-free
                       categorical kernel Gryffin itself falls back to
                       when no expert descriptors are supplied for a
                       categorical variable.

No external dependencies beyond numpy.
"""

import numpy as np


class ParameterSpace:
    """
    Holds every input parameter's definition and converts between
    real/labeled values (what your experiment understands) and the
    internal ENCODED representation (what the surrogate/acquisition
    operate on): continuous and numeric-discrete parameters contribute
    one encoded coordinate each, scaled to [0, 1]; categorical
    parameters contribute one encoded coordinate PER OPTION (a one-hot
    block), so the encoded space can be wider than the number of
    user-facing parameters.

    Example
    -------
        [{'name': 'temperature', 'kind': 'continuous',    'low': 20.0, 'high': 120.0},
         {'name': 'num_plates',  'kind': 'discrete_range', 'low': 5,    'high': 20},
         {'name': 'conc',        'kind': 'discrete_set',   'options': [0.5, 1.0, 2.5]},
         {'name': 'media',       'kind': 'categorical',    'options': ['A', 'B', 'C']}]
    """

    def __init__(self, param_specs):
        if len(param_specs) == 0:
            raise ValueError("param_specs must contain at least one parameter")

        self.names = [p["name"] for p in param_specs]
        self.kinds = [p.get("kind", "continuous") for p in param_specs]
        self.n_params = len(self.names)

        self.low = [None] * self.n_params          # continuous / discrete_range / discrete_set only
        self.high = [None] * self.n_params
        self.allowed_real = [None] * self.n_params  # sorted ndarray of legal values (discrete_range/discrete_set)
        self.categories = [None] * self.n_params     # list of labels (categorical only)

        offsets = []
        widths = []
        offset = 0
        for i, (p, kind) in enumerate(zip(param_specs, self.kinds)):
            if kind == "continuous":
                lo, hi = float(p["low"]), float(p["high"])
                if hi <= lo:
                    raise ValueError(f"{p['name']}: continuous needs high > low")
                self.low[i], self.high[i] = lo, hi
                width = 1

            elif kind == "discrete_range":
                lo, hi = int(p["low"]), int(p["high"])
                if hi <= lo:
                    raise ValueError(f"{p['name']}: discrete_range needs high > low")
                self.low[i], self.high[i] = float(lo), float(hi)
                self.allowed_real[i] = np.arange(lo, hi + 1, dtype=float)
                width = 1

            elif kind == "discrete_set":
                opts = np.array(sorted(float(v) for v in p["options"]))
                if len(opts) < 2:
                    raise ValueError(f"{p['name']}: discrete_set needs at least 2 options")
                self.low[i], self.high[i] = float(opts.min()), float(opts.max())
                self.allowed_real[i] = opts
                width = 1

            elif kind == "categorical":
                labels = list(p["options"])
                if len(labels) < 2:
                    raise ValueError(f"{p['name']}: categorical needs at least 2 options")
                if len(set(labels)) != len(labels):
                    raise ValueError(f"{p['name']}: categorical options must be unique")
                self.categories[i] = labels
                width = len(labels)

            else:
                raise ValueError(f"unknown parameter kind '{kind}' for '{p['name']}'")

            offsets.append(offset)
            widths.append(width)
            offset += width

        self.offsets = offsets
        self.widths = widths
        self.encoded_dim = offset  # total width of the internal representation
        self.discrete_dims = [i for i, k in enumerate(self.kinds) if k != "continuous"]
        self.categorical_dims = [i for i, k in enumerate(self.kinds) if k == "categorical"]

    # ------------------------------------------------------------------
    # encode: real/labeled dict  ->  internal vector (length encoded_dim)
    # ------------------------------------------------------------------
    def encode_dict(self, d):
        u = np.zeros(self.encoded_dim)
        for i, name in enumerate(self.names):
            if name not in d:
                raise KeyError(f"observation missing required parameter '{name}'")
            start = self.offsets[i]
            kind = self.kinds[i]

            if kind == "categorical":
                label = d[name]
                try:
                    cat_idx = self.categories[i].index(label)
                except ValueError:
                    raise ValueError(
                        f"'{label}' is not a legal option for categorical parameter "
                        f"'{name}' (allowed: {self.categories[i]})"
                    )
                u[start + cat_idx] = 1.0
            else:
                val = float(d[name])
                u[start] = (val - self.low[i]) / (self.high[i] - self.low[i])
        return u

    def encode_observations(self, observations):
        """observations: list of dicts -> ndarray (n, encoded_dim)"""
        return np.array([self.encode_dict(o) for o in observations])

    # ------------------------------------------------------------------
    # decode: internal vector  ->  real/labeled dict
    # ------------------------------------------------------------------
    def decode_vector(self, u):
        u = np.asarray(u, dtype=float)
        out = {}
        for i, name in enumerate(self.names):
            start = self.offsets[i]
            kind = self.kinds[i]

            if kind == "categorical":
                block = u[start:start + self.widths[i]]
                cat_idx = int(np.argmax(block))  # snap: nearest one-hot corner
                out[name] = self.categories[i][cat_idx]
            elif kind == "continuous":
                val = self.low[i] + np.clip(u[start], 0.0, 1.0) * (self.high[i] - self.low[i])
                out[name] = float(val)
            else:  # discrete_range / discrete_set -- snap to nearest legal value
                val = self.low[i] + np.clip(u[start], 0.0, 1.0) * (self.high[i] - self.low[i])
                opts = self.allowed_real[i]
                snapped = opts[np.argmin(np.abs(opts - val))]
                out[name] = int(snapped) if float(snapped).is_integer() else float(snapped)
        return out

    # ------------------------------------------------------------------
    # blocks: the finite-choice coordinate groups the acquisition
    # optimizer must treat as one atomic swap (never a gradient step)
    # ------------------------------------------------------------------
    def blocks(self):
        """
        Returns a list of (indices, candidates) for every NON-continuous
        parameter:
          - indices    : list of encoded-vector positions this parameter
                         occupies (length 1 for discrete_range/discrete_set,
                         length n_categories for categorical)
          - candidates : ndarray (n_options, len(indices)) -- each row is
                         one legal full assignment for those positions
                         (a single scaled value, or a one-hot row)

        Used by the acquisition optimizer exactly like Gryffin's
        NaiveDiscreteOptimizer/NaiveCategoricalOptimizer: try one
        random candidate row per block per iteration, keep only if it
        improves the acquisition value; never move these coordinates
        with a gradient step.
        """
        out = []
        for i in self.discrete_dims:
            start = self.offsets[i]
            width = self.widths[i]
            indices = list(range(start, start + width))
            if self.kinds[i] == "categorical":
                candidates = np.eye(width)  # one-hot rows
            else:
                opts = self.allowed_real[i]
                unit_vals = (opts - self.low[i]) / (self.high[i] - self.low[i])
                candidates = unit_vals.reshape(-1, 1)
            out.append({"indices": indices, "candidates": candidates})
        return out


def clean_observations(observations, space, objective_names,
                        drop_duplicates=True, verbose=True):
    """
    Validate and clean a list of observation dicts before they are fed
    to the optimizer.

    Parameters
    ----------
    observations : list of dict
    space : ParameterSpace
        Used to know each parameter's kind: numeric params are cast to
        float and checked for finiteness; categorical params are
        checked against their declared legal labels (no numeric cast).
    objective_names : list of str
    drop_duplicates : bool
        Remove rows whose full parameter assignment exactly repeats an
        earlier row.
    verbose : bool

    Returns
    -------
    cleaned : list of dict
    """
    all_keys = space.names + objective_names
    cleaned = []
    seen = []
    n_dropped_missing = 0
    n_dropped_bad_value = 0
    n_dropped_dup = 0

    for row in observations:
        if not all(k in row for k in all_keys):
            n_dropped_missing += 1
            continue

        ok = True
        for i, name in enumerate(space.names):
            if space.kinds[i] == "categorical":
                if row[name] not in space.categories[i]:
                    ok = False
                    break
            else:
                try:
                    v = float(row[name])
                except (TypeError, ValueError):
                    ok = False
                    break
                if not np.isfinite(v):
                    ok = False
                    break
        if not ok:
            n_dropped_bad_value += 1
            continue

        for name in objective_names:
            try:
                v = float(row[name])
            except (TypeError, ValueError):
                ok = False
                break
            if not np.isfinite(v):
                ok = False
                break
        if not ok:
            n_dropped_bad_value += 1
            continue

        if drop_duplicates:
            key = tuple(row[name] for name in space.names)
            if key in seen:
                n_dropped_dup += 1
                continue
            seen.append(key)

        cleaned.append(row)

    if verbose:
        print(
            f"[clean_observations] kept {len(cleaned)}/{len(observations)} rows "
            f"(missing_keys={n_dropped_missing}, invalid_value={n_dropped_bad_value}, "
            f"duplicates={n_dropped_dup})"
        )

    return cleaned


def objectives_to_array(observations, objective_names):
    """list of dicts -> ndarray (n, len(objective_names))"""
    return np.array([[float(o[name]) for name in objective_names] for o in observations])
