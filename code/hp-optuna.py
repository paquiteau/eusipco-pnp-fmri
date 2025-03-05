""" Grid search of hyper parameters for pnp reconstruction with optuna."""

import optuna
import torch
from snake.mrd_utils import NonCartesianFrameDataLoader
from seqpnp.sequentialpnp import get_custom_init, get_DPIR_params, OPTIMIZERS
from seqpnp.physic import Nufft
from seqpnp.drunet import load_drunet_mri
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optimizers import (
    BaseOptim,
    optim_builder,
)
from deepinv.optim.prior import PnP
from functools import partial
# +
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim




def compute_psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)."""
    return compare_psnr(abs(gt), abs(pred), data_range=abs(gt).max() - abs(gt).min())


def compute_ssim(gt, pred):
    """Compute Structural Similarity Index Metric (SSIM)."""
    return compare_ssim(
        abs(gt),
        abs(pred),
        # gt.transpose(1, 2, 0),
        # pred.transpose(1, 2, 0),
        # multichannel=True,
        data_range=abs(gt).max() - abs(gt).min(),
    )
# %%

device="cuda"
density_compensation=None #"pipe"
nufft_backend="stacked-cufinufft"
x_init="pinv"
model_weights = "weights/drunet_3d_0303.pth"
max_iter_per_frame = 20
optimizer="hqs-f1"

with NonCartesianFrameDataLoader("example_spiral_optim.mrd") as data_loader:
    phantom = data_loader.get_phantom()
    sim_conf = data_loader.get_sim_conf()
    ground_truth = phantom.contrast(sim_conf=sim_conf)
    traj, kspace = data_loader.get_kspace_frame(0)

    physics = Nufft(
        data_loader.shape,
        data_loader.get_kspace_frame(0)[0],
        density=density_compensation,
        smaps=data_loader.get_smaps(),
        backend=nufft_backend,
    ).to(device)
    lipchitz_cst = physics.nufft.get_lipschitz_cst(max_iter=100)

denoiser = load_drunet_mri(model_weights, device=device, dim=3, norm_factor=1)
try:
    IteratorKlass = OPTIMIZERS[optimizer]
except KeyError as e:
    raise ValueError("Unknown optimizer config.") from e

kwargs_optim = {
    "early_stop": False,
    "verbose": True,
    "max_iter": max_iter_per_frame,
}


def objective(trial):
    xi = trial.suggest_float("xi", 0.5, 1)
    stepsize = trial.suggest_float("stepsize", 1e1, 1e3)
    sigma = trial.suggest_float("sigma", 1e-5, 1e-1)
    data_torch = torch.from_numpy(kspace).to(device)
    extra_kwargs_optim = {
        "custom_init": get_custom_init(
        x_init, physics, data_torch, device
        ),
        "params_algo": get_DPIR_params(
        xi=xi,
        stepsize=stepsize,
        sigma=sigma,
        lipschitz_cst=lipchitz_cst,
        n_iter=max_iter_per_frame,
        ),
    }
    init_optim = partial(
        optim_builder,
        iteration=IteratorKlass(),
        prior=PnP(denoiser),
        data_fidelity=L2(),
        **kwargs_optim,
    )

    optim: BaseOptim = init_optim(**extra_kwargs_optim).to(device)
    optim.fixed_point.show_progress_bar = True
    x_est = optim(data_torch, physics=physics, compute_metrics=False)
    psnr = compute_psnr(ground_truth, abs(x_est.to("cpu").numpy().squeeze()))
    return psnr

study = optuna.create_study(direction="maximize", storage="sqlite:///db.sqlite3")
study.optimize(objective, n_trials=100)
print(f"Best value: {study.best_value} (params: {study.best_params})")
