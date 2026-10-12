"""The policy networks as plain numpy, from the files written by
scripts/export_policy_weights.py.

Not used to drive the robot: both policies observe quantities only the
simulator provides (see INTERFACE.md section 3). It exists so the weights kept
in this repository can be loaded and checked without PyTorch.
"""

import numpy as np


class PolicyNet:
    def __init__(self, path):
        d = np.load(path, allow_pickle=False)
        self.layers = [(d["w%d" % i], d["b%d" % i]) for i in range(int(d["n_layers"]))]
        self.w_out, self.b_out = d["w_out"], d["b_out"]
        self.mean, self.std = d["obs_mean"], np.sqrt(d["obs_var"]) + 1e-8
        self.obs_dim, self.act_dim = int(self.mean.shape[0]), int(self.b_out.shape[0])

    def act(self, obs):
        """Mean action for an observation (the environment then clips it to [-1, 1])."""
        x = np.clip((np.asarray(obs, dtype=np.float32) - self.mean) / self.std, -5.0, 5.0)
        for w, b in self.layers:
            x = x @ w.T + b
            x = np.where(x > 0, x, np.expm1(np.minimum(x, 0.0)))     # ELU
        return x @ self.w_out.T + self.b_out
