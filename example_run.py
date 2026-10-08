"""
example_run.py
---------------
Minimal end-to-end demonstration of the full pipeline on a synthetic
problem with 7 continuous inputs and 5 targets, mirroring the
structure you'd use with your own experiment.

Run with:  python3 example_run.py
"""

import numpy as np
from optimizer import SimpleGryffin


# ----------------------------------------------------------------------
# 1) Define your 7 continuous parameters (real physical bounds)
# ----------------------------------------------------------------------
parameters = [
    {"name": f"x{i+1}", "low": 0.0, "high": 10.0} for i in range(7)
]

# ----------------------------------------------------------------------
# 2) Define your 5 objectives, IN PRIORITY ORDER for Chimera.
#    (edit goal / tolerance / absolute to match your real priorities)
# ----------------------------------------------------------------------
objectives = [
    {"name": "y1", "goal": "max", "tolerance": 0.2, "absolute": False},
    {"name": "y2", "goal": "min", "tolerance": 0.2, "absolute": False},
    {"name": "y3", "goal": "max", "tolerance": 0.2, "absolute": False},
    {"name": "y4", "goal": "min", "tolerance": 0.2, "absolute": False},
    {"name": "y5", "goal": "max", "tolerance": 0.2, "absolute": False},
]

general = {
    "sampling_strategies": 2,   # 1 exploit-biased + 1 explore-biased proposal/round
    "random_seed": 42,
    "verbose": True,
}


# ----------------------------------------------------------------------
# 3) Stand-in for your real experiment: a synthetic 7D -> 5D function.
#    Replace `run_experiment` with whatever calls your lab equipment /
#    simulation and returns a dict with keys y1..y5.
# ----------------------------------------------------------------------
def run_experiment(params):
    x = np.array([params[f"x{i+1}"] for i in range(7)]) / 10.0  # -> [0,1]^7
    center = np.array([0.3, 0.7, 0.5, 0.2, 0.8, 0.4, 0.6])
    dist2 = np.sum((x - center) ** 2)

    y1 = float(np.exp(-dist2) * 100)                            # maximize
    y2 = float(10 * dist2 + 0.5 * x[0])                         # minimize
    y3 = float(np.sin(3 * x[1]) * np.cos(2 * x[2]) * 50 + 50)   # maximize
    y4 = float(np.sum(x ** 2))                                  # minimize
    y5 = float(100 - 80 * dist2)                                # maximize

    return {"y1": y1, "y2": y2, "y3": y3, "y4": y4, "y5": y5}


def merit_of(opt, obs):
    return opt.chimera.scalarize([[obs[n] for n in opt.obj_names]])[0]


# ----------------------------------------------------------------------
# 4) The actual recommend -> run -> update loop
# ----------------------------------------------------------------------
def main():
    opt = SimpleGryffin(parameters, objectives, general)

    observations = []  # cold start; swap in your historical data here if you have any

    n_rounds = 12
    for round_idx in range(n_rounds):
        proposals = opt.recommend(observations, num_proposals=2)

        for p in proposals:
            result = run_experiment(p)
            p.update(result)
            observations.append(p)

        best = min(observations, key=lambda o: merit_of(opt, o))
        print(f"  round {round_idx + 1}: best so far -> "
              f"{ {k: round(best[k], 3) for k in opt.obj_names} }\n")

    print("Final best observation:")
    best = min(observations, key=lambda o: merit_of(opt, o))
    for k, v in best.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
