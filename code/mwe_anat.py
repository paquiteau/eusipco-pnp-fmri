# %%
"""A minimal working example of a PnP reconstruction using DeepInv and SNAKE."""

# %%
from pathlib import Path

from snake.core import GreConfig, Phantom, SimConfig
from snake.core.engine import NufftAcquisitionEngine
from snake.core.handlers.fov import FOVHandler
from snake.core.sampling import StackOfSpiralSampler
from snake.core.simulation import default_hardware
from snake.core.smaps import get_smaps
from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.plotting import axis3dcut


from seqpnp.sequentialpnp import SequentialPnPReconstructor

# %%
# ## Setup the simulation configuration
# %%

sim_conf = SimConfig(
    max_sim_time=2,
    seq=GreConfig(TR=50, TE=22, FA=12),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(181, 217, 181),
)
sim_conf.hardware.n_coils = 8  # Update to get multi coil results.
sim_conf.hardware.field = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


sampler = StackOfSpiralSampler(
    accelz=2,
    acsz=0.1,
    orderz="top-down",
    nb_revolutions=12,
    obs_time_ms=20,
    constant=True,
    spiral_name="archimedes"
)


fov_handler = FOVHandler(
    center=(90, 110, 110),
    size=(192, 192, 144),
    angles=(5, 0, 0),
    target_res=(3.0, 3.0, 3.0),
)

# %%
#

engine = NufftAcquisitionEngine(model="T2s", snr=1000)

engine(
    "example_spiral.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[fov_handler],
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend="stacked-cufinufft",
)


# %%
from snake.toolkit.reconstructors import ZeroFilledReconstructor

zer_rec = ZeroFilledReconstructor(
    nufft_backend="gpunufft", density_compensation="pipe"
)
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    adj_data = zer_rec.reconstruct(data_loader)
    traj, _ = data_loader.get_kspace_frame(0)



# %%
from mrinufft.trajectories import display_3D_trajectory, display_2D_trajectory
display_3D_trajectory(traj.reshape(35,-1, 3))

# %%
display_2D_trajectory(traj.reshape(35, -1, 3)[0:1,:,:2])

# %%
len(adj_data)

# %%
axis3dcut(abs(adj_data[0].T), None, None, cuts=(0.5, 0.5, 0.5))

# %%

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

def get_custom_init(
    x_init: str | NDArray | torch.Tensor,
    physics: Nufft,
    data: torch.Tensor,
    device: str,
):
    if isinstance(x_init, torch.Tensor):
        pass
    elif isinstance(x_init, np.ndarray):
        x_init = torch.from_numpy(x_init).to(device)
    elif "adjoint-" in x_init:
        _, density = x_init.split("-")
        physics.nufft.compute_density(density)
        x_init = physics.nufft.adj_op(data)
        physics.nufft.compute_density(None)  # remove the density.
    elif x_init == "adjoint":
        x_init = physics.A_adjoint(data)
    elif x_init == "pinv":
        x_init = physics.A_dagger(data)
    else:
        x_init = physics.A_adjoint(data)
    if isinstance(x_init, torch.Tensor):
        return lambda y, p: {"est": (x_init, x_init.detach().clone())}
    else:
        raise ValueError("Unknown custom init.")




OPTIMIZERS: dict[str, Callable[..., OptimIterator]] = {
    "hqs": partial(PreconditionedHQSIteration, precond=None),
    "hqs-cheb": partial(PreconditionedHQSIteration, precond="cheby"),
    "hqs-f1": partial(PreconditionedHQSIteration, precond="static"),
    "pnp": partial(PreconditionedPnPIteration, precond=None),
    "pnp-f1": partial(PreconditionedPnPIteration, precond="static"),
    "pnp-cheb": partial(PreconditionedPnPIteration, precond="cheby"),
}



# %%
model_weights = "weights/drunet_3d_0602.pth"

denoiser = load_drunet_mri(
     model_weights, device=device, dim=3, norm_factor= 1
)

# %%
device="cuda"
density_compensation=None #"pipe"
nufft_backend="stacked-cufinufft"
x_init="pinv"
dpir_params = {"xi":0.75, "stepsize":"1", "lamb":100, "sigma":1e-3}
max_iter_per_frame=8
optimizer="hqs"

# %%
import matplotlib.pyplot as plt
from time import perf_counter

# %%
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    traj, data = data_loader.get_kspace_frame(0)
    shape=data_loader.shape
    smaps = data_loader.get_smaps()
    data_torch = torch.from_numpy(data).to(device)


# %%
physics = Nufft(
    shape,
    traj,
    density=density_compensation,
    smaps=smaps,
    backend=nufft_backend,
).to(device)

# %%
physics.nufft.get_lipschitz_cst()

# %%
pinv = physics.A_dagger(data_torch)

# %%
axis3dcut(abs(pinv.to("cpu").numpy().T),  None, None, cuts=(0.5, 0.3, 0.5))

# %%
with torch.no_grad():
    out = denoiser(pinv, 0.001)


# %%
axis3dcut(abs(out.to("cpu").numpy().T),  None, None, cuts=(0.5, 0.3, 0.5))

# %%
# with torch.no_grad():
#     out2 = L2().prox(out, data_torch, physics, gamma=1) # cur_params["stepsize"])
# axis3dcut(abs(out2.to("cpu").numpy().T),  None, None, cuts=(0.5, 0.3, 0.5))

# %%
x_init = pinv

try:
    IteratorKlass = OPTIMIZERS[ optimizer]
except KeyError as e:
    raise ValueError("Unknown optimizer config.") from e

kwargs_optim = {
    "early_stop": False,
    "verbose": True,
    "max_iter":  max_iter_per_frame,
}

init_optim = partial(
    optim_builder,
    iteration=IteratorKlass(),
    prior=PnP( denoiser),
    data_fidelity=L2(),
    **kwargs_optim,
)

with torch.no_grad():
       # physics.nufft.samples = traj
    extra_kwargs_optim = {
        "custom_init": get_custom_init(
            x_init, physics, data_torch, device
        ),
        "params_algo": get_DPIR_params(
            **dpir_params,
            lipschitz_cst=physics.nufft.get_lipschitz_cst(),
            n_iter=max_iter_per_frame,
        ),
    }
    optim: BaseOptim = init_optim(**extra_kwargs_optim).to(device)
    x_cur = optim.fixed_point.init_iterate_fn(data_torch, physics)
    itr = 0
    for _ in range(max_iter_per_frame):
        fig, ax = plt.subplots()
        axis3dcut(abs(x_cur["est"][0].to("cpu").numpy().T), None, None, cuts=(0.5, 0.3, 0.5),ax=ax, width_inches=5)
        plt.show()
        x_cur =  optim.fixed_point.single_iteration(
            x_cur, 
            itr, 
            data_torch, 
            physics, 
            #compute_metrics=False,
          #  x_gt=None,
        )
       # axis3dcut(abs(x_cur["est"][0].to("cpu").numpy().T), None, None, cuts=(0.5, 0.3, 0.5),ax=ax)
    # x_est = optim(
    #     data_torch, physics=physics, compute_metrics=False
    # )
    # # Setup optim iterator and solve for frame
    
    #final_images[i] = x_est.to("cpu").numpy()


# %%
axis3dcut(abs(rec_data[0].T), None, None, cuts=(0.5, 0.3, 0.5))

# %%

# %%

# %%

# %%

# %%
