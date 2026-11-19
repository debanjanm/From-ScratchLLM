"""Checkpointing — save and resume training EXACTLY.

A checkpoint that only stores weights lets you run inference but not resume
training: Adam's momentum (m), variance (v) and step count t are state too.
Restart with fresh optimizer state and the next steps differ from what an
uninterrupted run would have done (and often spike the loss). A resumable
checkpoint therefore stores: model params, optimizer m/v/t, the step number,
and the config needed to rebuild the model. np.savez writes it as one .npz
file; flat key prefixes ("p/", "m/", "v/") keep the dict structure.

The test below is the real proof: train 6 steps straight, versus train 3,
save, load into a brand-new model and optimizer, train 3 more. The final
weights must be identical.
"""

import importlib
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
_bp = importlib.import_module("01_backprop_gpt")
_opt = importlib.import_module("02_optimizers")


def save_checkpoint(path, params: dict, opt: "_opt.AdamW", step: int, config: dict):
    flat = {f"p/{k}": v for k, v in params.items()}
    flat.update({f"m/{k}": v for k, v in opt.m.items()})
    flat.update({f"v/{k}": v for k, v in opt.v.items()})
    np.savez(path, step=step, t=opt.t, config=np.array(list(config.values())),
             config_keys=np.array(list(config.keys())), **flat)


def load_checkpoint(path):
    z = np.load(path)
    pick = lambda prefix: {k[2:]: z[k] for k in z.files if k.startswith(prefix)}
    config = {k: int(v) for k, v in zip(z["config_keys"], z["config"])}
    return pick("p/"), {"t": int(z["t"]), "m": pick("m/"), "v": pick("v/")}, int(z["step"]), config


def build(config):
    return _bp.TrainableGPT(config["vocab"], config["d_model"], config["heads"],
                            config["layers"], config["max_len"], seed=config["seed"])


if __name__ == "__main__":
    config = dict(vocab=11, d_model=8, heads=2, layers=1, max_len=8, seed=0)
    batches = [np.random.default_rng(i).integers(0, 11, size=(2, 6)) for i in range(6)]

    def train(model, opt, steps):
        for ids in steps:
            _, grads = model.loss_and_grads(ids)
            opt.step(model.params, grads)

    # A: uninterrupted 6 steps
    a = build(config)
    opt_a = _opt.AdamW(a.params, lr=1e-2)
    train(a, opt_a, batches)

    # B: 3 steps, checkpoint, rebuild everything from disk, 3 more steps
    b = build(config)
    opt_b = _opt.AdamW(b.params, lr=1e-2)
    train(b, opt_b, batches[:3])
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "ckpt.npz"
        save_checkpoint(path, b.params, opt_b, step=3, config=config)
        params, state, step, cfg = load_checkpoint(path)
        snapshot = {k: v.copy() for k, v in params.items()}   # untouched copy of the saved weights
    assert step == 3 and cfg == config

    b2 = build(cfg)
    b2.params = params
    opt_b2 = _opt.AdamW(b2.params, lr=1e-2)
    opt_b2.load_state_dict(state)
    train(b2, opt_b2, batches[3:])

    same = all(np.array_equal(a.params[k], b2.params[k]) for k in a.params)
    print("resumed run identical to uninterrupted run:", same)
    assert same

    # failure mode this guards against: restore weights only, restart Adam from scratch
    c = build(cfg)
    c.params = {k: v.copy() for k, v in snapshot.items()}
    train(c, _opt.AdamW(c.params, lr=1e-2), batches[3:])
    differs = any(not np.array_equal(a.params[k], c.params[k]) for k in a.params)
    print("weights-only resume (fresh optimizer state) diverges from the true run:", differs)
    assert differs
