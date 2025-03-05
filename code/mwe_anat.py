# %%
"""A minimal working example of a PnP reconstruction using DeepInv and SNAKE."""

# %%
from pathlib import Path
import matplotlib.pyplot as plt

from snake.core import GreConfig, Phantom, SimConfig
from snake.core.engine import NufftAcquisitionEngine
from snake.core.sampling import StackOfSpiralSampler
from snake.core.simulation import default_hardware
from snake.core.smaps import get_smaps
from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.plotting import axis3dcut



from seqpnp.sequentialpnp import SequentialPnPReconstructor

# +
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def compute_psnr(gt, pred):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
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


# -

# ## Setup the simulation configuration

# +
sim_conf = SimConfig(
    max_sim_time=1,
    seq=GreConfig(TR=50, TE=22, FA=5),
    hardware=default_hardware,
)
sim_conf.hardware.n_coils = 8  # Update to get multi coil results.
sim_conf.hardware.field = 7
sim_conf.fov.size =(192, 192, 144)
sim_conf.fov.angles=(-5, 0, 0) 
sim_conf.fov.res_mm=(3,3,3)
sim_conf.fov.offset=(-90,-110, -35)


sim_conf

# +

phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T", output_res=1)
print(phantom.affine)
print(sim_conf.fov.affine)
#phantom = phantom.resample(new_affine=sim_conf.fov.affine, new_shape=sim_conf.shape,use_gpu=True) # Cleanup here 
# -
ground_truth = phantom.contrast(sim_conf=sim_conf)
axis3dcut(abs(ground_truth.T), None, None, cuts=(0.5, 0.5, 0.5), width_inches=5)



sampler = StackOfSpiralSampler(
    accelz=4,
    acsz=0.1,
    orderz="top-down",
    nb_revolutions=12,
    obs_time_ms=25,
    in_out=True,
    constant=True,
    spiral_name="archimedes"
)


engine = NufftAcquisitionEngine(model="T2s", snr=500)

engine(
    "example_spiral.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[],
    worker_chunk_size=60,
    n_workers=4,
    nufft_backend="cufinufft",
)


# %%
from snake.toolkit.reconstructors import ZeroFilledReconstructor

# +
zer_rec = ZeroFilledReconstructor(
    nufft_backend="gpunufft", density_compensation="pipe",
    
)
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    adj_data = zer_rec.reconstruct(data_loader)
    traj, _ = data_loader.get_kspace_frame(0)
    phantom = data_loader.get_phantom()
    sim_conf = data_loader.get_sim_conf()
# -


# %%
len(adj_data)

ground_truth = phantom.contrast(sim_conf=sim_conf)


fig, axs = plt.subplots(2,1, figsize=(5,8))
fig, ax, _ = axis3dcut(abs(adj_data[0].T), None, None, cuts=(0.5, 0.5, 0.5), ax=axs[0], width_inches=5)
fig, ax1, _ = axis3dcut(abs(ground_truth.T), None, None, cuts=(0.5, 0.5, 0.5), ax=axs[1], width_inches=5)
psnr = compute_psnr(ground_truth, abs(adj_data[0]))
ssim = compute_ssim(ground_truth, abs(adj_data[0]))
ax.set_title(f"PSNR={psnr:.3f}, SSIM={ssim:.3f}")

from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Callable, ClassVar, Literal

import numpy as np
from seqpnp import physic
import torch
from deepinv.optim.data_fidelity import L2
from deepinv.optim.optim_iterators import OptimIterator
from deepinv.optim.optimizers import (
    BaseOptim,
    optim_builder,
)
from deepinv.optim.prior import PnP
from numpy.typing import NDArray
from snake.core import SimConfig
from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.reconstructors import BaseReconstructor

from seqpnp.drunet import load_drunet_mri
from seqpnp.physic import Nufft
from seqpnp.utils import (
    PreconditionedHQSIteration,
    PreconditionedPnPIteration,
    get_DPIR_params,
) 
device="cuda"

# %%
import matplotlib.pyplot as plt
from time import perf_counter
from seqpnp.sequentialpnp import SequentialPnPReconstructor

# %%
model_weights = "weights/drunet_3d_2602.pth"


# %%
device="cuda"
density_compensation=None #"pipe"
nufft_backend="gpunufft"
x_init="adjoint-pipe"
dpir_params = {"xi":0.75, "stepsize":"1", "lamb":100, "sigma":1e-3}
max_iter_per_frame=8
optimizer="hqs"
model_weights = "weights/drunet_3d_0602.pth"


with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    ksp, data =   data_loader.get_kspace_frame(0)
    physics = Nufft(
    data_loader.shape,
    ksp,
    density=density_compensation,
    smaps=data_loader.get_smaps(),
    backend=nufft_backend,
    ).to(device)
    pinv = physics.A_dagger(torch.from_numpy(data).to(device))
    pinv = pinv.to("cpu").numpy()[0]
fig, ax= plt.subplots()
axis3dcut(abs(pinv.T), None, None, cuts=(0.5, 0.5, 0.5), ax=ax)
psnr = compute_psnr(ground_truth, abs(pinv[0]))
ssim = compute_ssim(ground_truth, abs(pinv[0]))
ax.set_title(f"pinv PSNR={psnr:.3f}, SSIM={ssim:.3f}")

# +
# %%
device="cuda"
density_compensation=None #"pipe"
nufft_backend="stacked-cufinufft"
x_init="pinv"
model_weights = "weights/drunet_3d_0303.pth"

with NonCartesianFrameDataLoader("example_spiral_optim.mrd") as data_loader:
    rec = SequentialPnPReconstructor(
        model_weights=model_weights,
        nufft_backend=nufft_backend,
        x_init=x_init,
        dpir_params = {'xi': 0.9735980562499394, 'stepsize': 12.797325331814873, 'sigma': 0.0038954273339963873},
        density_compensation=density_compensation,
        max_iter_per_frame=20,
        optimizer="hqs-f1",
    )
    data_rec=rec.reconstruct(data_loader)
fig, ax, _ = axis3dcut(abs(data_rec[0].T), None, None, cuts=(0.5, 0.5, 0.5))
psnr = compute_psnr(ground_truth, abs(data_rec[0]))
ssim = compute_ssim(ground_truth, abs(data_rec[0]))
ax.set_title(f"hqs PSNR={psnr:.3f}, SSIM={ssim:.3f}")

# + active=""
#
# -





from snake.toolkit.reconstructors import SequentialReconstructor

# %%
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    rec = SequentialReconstructor(
        nufft_backend=nufft_backend,
        density_compensation=density_compensation,
        max_iter_per_frame=50,
        wavelet="db4",
    )
    datawaavelet=rec.reconstruct(data_loader)

fig, ax, _ = axis3dcut(abs(datawaavelet[0].T), None, None, cuts=(0.5, 0.5, 0.5))
psnr = compute_psnr(ground_truth, abs(datawaavelet[0]))
ssim = compute_ssim(ground_truth, abs(datawaavelet[0]))
ax.set_title(f"wavelet PSNR={psnr:.3f}, SSIM={ssim:.3f}")








