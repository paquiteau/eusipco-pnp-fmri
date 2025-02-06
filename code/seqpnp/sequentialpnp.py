"""Sequential Reconstructor that uses PnP as prior."""

from numpy.typing import NDArray
from functools import partial
import numpy as np
import torch

from snake.core import SimConfig
from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.reconstructors import BaseReconstructor

from .physic import Nufft
from .drunet import load_drunet_mri
from .utils import (
    PreconditionedHQSIteration,
    PreconditionedPnPIteration,
    get_DPIR_params,
)
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP
from deepinv.optim.optimizers import (
    BaseOptim,
    optim_builder,
)

OPTIMIZERS = {
    "hqs": partial(PreconditionedHQSIteration, precond=None),
    "hqs-cheb": partial(PreconditionedHQSIteration, precond="cheby"),
    "hqs-f1": partial(PreconditionedHQSIteration, precond="static"),
    "pnp": partial(PreconditionedPnPIteration, precond=None),
    "pnp-f1": partial(PreconditionedPnPIteration, precond="static"),
    "pnp-cheb": partial(PreconditionedPnPIteration, precond="cheby"),
}


class SequentialPnPReconstructor(BaseReconstructor):
    """A sequential reconstructor that uses PNP Methods.

    Parameters
    --------
    model_weights: str
        Path to the model weights for the denoiser.
    nufft_backend: str
        Backend to use for NUFFT operations.
    density_compensation: str | bool
        Density compensation to use.
    optimizer: str
        Optimizer to use for the reconstruction.
    max_iter_per_frame: int
        Maximum number of iterations per frame.
    device: str
        Device to use for the reconstruction.
    """

    __reconstructor_name__ = "sequential-pnp"

    model_weights: str
    nufft_backend: str = "cufinufft"
    density_compensation: str | bool = "pipe"
    optimizer: str = "hqs-f1"
    max_iter_per_frame: int = 50
    device: str = "cuda"

    def setup(self, sim_conf: SimConfig = None, shape: tuple[int] = None) -> None:
        """Set up the reconstructor."""
        # Use the Deepinv object interfaces
        # Load the denoiser, setup the pnp stuff
        self.denoiser = load_drunet_mri(self.model_weights, device=self.device)
        try:
            IteratorKlass = OPTIMIZERS[self.optimizer]
        except KeyError as e:
            raise ValueError("Unknown optimizer config.") from e

        kwargs_optim = {
            "early_stop": False,
            "verbose": False,
            "max_iter": self.max_iter,
        } | get_DPIR_params(n_iter=self.max_iter)

        self.init_optim = partial(
            optim_builder,
            iteration=IteratorKlass(),
            prior=PnP(self.denoiser),
            data_fidelity=L2(),
            **kwargs_optim,
        )

    def reconstruct(
        self, data_loader: NonCartesianFrameDataLoader, sim_conf: SimConfig = None
    ) -> NDArray:
        """Reconstruct using PnP method."""
        traj, data = data_loader.get_kspace_frame(0)
        kwargs = {}
        if "stacked" in self.nufft_backend:
            kwargs["z_index"] = "auto"
        if self.nufft_backend == "cufinufft":
            kwargs["smaps_cached"] = True

        final_images = np.zeros(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )

        physics = Nufft(
            data_loader.shape,
            data_loader.get_kspace_frame(0)[0],
            density=self.density_compensation,
            smaps=data_loader.get_smaps(),
        )
        with torch.no_grad():
            for i, traj, data in data_loader.iter_frames():
                physics.fourier_op.samples = traj
                optim: BaseOptim = self.init_optim()
                data_torch = torch.from_numpy(data).to("device")
                x_est, metrics = optim(physics, data_torch)
                # Setup optim iterator and solve for frame
                final_images[i] = x_est

        return final_images
