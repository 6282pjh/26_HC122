"""Extract the policy from a skrl PPO checkpoint into a small .npz.

A training checkpoint holds the policy, the value function and the optimiser
state. Only the policy network and the observation normaliser are needed to
reproduce the policy's actions, and they are a fraction of the file. The result
is checked against the PyTorch network on random observations before it is kept.

Needs torch and numpy; does not need Isaac Lab.

Usage:
    python scripts/export_policy_weights.py <agent.pt> --out weights/incision_policy.npz
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from rl_policy.deploy.policy_np import PolicyNet  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("checkpoint")
parser.add_argument("--out", required=True)
args = parser.parse_args()

ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
pol = ck["policy"]
pre = ck.get("observation_preprocessor") or ck.get("state_preprocessor")
keys = sorted((k for k in pol if k.startswith("net_container") and k.endswith("weight")),
              key=lambda k: int(k.split(".")[1]))
pack = {"n_layers": len(keys)}
for i, k in enumerate(keys):
    pack["w%d" % i] = pol[k].numpy()
    pack["b%d" % i] = pol[k.replace("weight", "bias")].numpy()
pack.update(w_out=pol["policy_layer.weight"].numpy(), b_out=pol["policy_layer.bias"].numpy(),
            obs_mean=pre["running_mean"].numpy().astype(np.float32),
            obs_var=pre["running_variance"].numpy().astype(np.float32))
np.savez_compressed(args.out, **pack)

# Reference forward pass in torch: skrl's RunningStandardScaler, then the ELU MLP.
net = PolicyNet(args.out)
obs = torch.randn(256, net.obs_dim) * 2.0
x = torch.clamp((obs - pre["running_mean"].float()) / (torch.sqrt(pre["running_variance"].float()) + 1e-8), -5.0, 5.0)
for k in keys:
    x = torch.nn.functional.elu(x @ pol[k].T + pol[k.replace("weight", "bias")])
ref = (x @ pol["policy_layer.weight"].T + pol["policy_layer.bias"]).numpy()
err = float(np.abs(net.act(obs.numpy()) - ref).max())
size = os.path.getsize(args.out)
sha = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
print("observations %d, actions %d, hidden %s" % (net.obs_dim, net.act_dim, [w.shape[0] for w, _ in net.layers]))
print("numpy vs torch, largest action difference: %.2e" % err)
print("%s  %.2f MB  sha256 %s" % (args.out, size / 1e6, sha))
if err > 1e-4:
    os.remove(args.out)
    sys.exit("MISMATCH: the file was removed.")
