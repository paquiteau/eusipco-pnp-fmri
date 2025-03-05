"""Sequential Reconstructor that uses PnP as prior."""

from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Callable, ClassVar, Literal

import numpy as np
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

from .drunet import load_drunet_mri
from .physic import Nufft
from .utils import (
    PreconditionedHQSIteration,
    PreconditionedPnPIteration,
    get_DPIR_params,
)


OPTIMIZERS: dict[str, Callable[..., OptimIterator]] = {
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
    x_init: torch.Tensor | NDArray | str
        Initial guess for the reconstruction.
    dpir_params: dict
        Parameters for the DPIR algorithm.
        Available keys: sigma, xi, lamb, stepsize.
        stepsize can be a string that will be evaluated (e.g. "1/lipschitz_cst")
    """

    __reconstructor_name__: ClassVar[str] = "sequential-pnp"

    model_weights: str | Path = ""
    nufft_backend: str = "stacked-cufinufft"
    density_compensation: str | None | bool = "voronoi"
    optimizer: str = "hqs-f1"
    max_iter_per_frame: int = 50
    device: str = "cuda"
    compute_metrics: bool = False
    x_init: str = "adjoint"
    dpir_params: dict[str, float] = field(default_factory=dict)
    norm_factor: float = 1

    def __str__(self) -> str:
        """Return a string representation of the reconstructor."""
        return f"{self.__reconstructor_name__}-{self.optimizer}"

    def setup(
        self, sim_conf: SimConfig | None = None, shape: tuple[int, ...] | None = None
    ) -> None:
        """Set up the reconstructor."""
        # Use the Deepinv object interfaces
        # Load the denoiser, setup the pnp stuff
        self.denoiser = load_drunet_mri(
            self.model_weights, device=self.device, dim=3, norm_factor=self.norm_factor
        )
        try:
            IteratorKlass = OPTIMIZERS[self.optimizer]
        except KeyError as e:
            raise ValueError("Unknown optimizer config.") from e

        kwargs_optim = {
            "early_stop": False,
            "verbose": True,
            "max_iter": self.max_iter_per_frame,
        }

        self.init_optim = partial(
            optim_builder,
            iteration=IteratorKlass(),
            prior=PnP(self.denoiser),
            data_fidelity=L2(),
            **kwargs_optim,
        )

    def reconstruct(
        self,
        data_loader: NonCartesianFrameDataLoader,
        constant=True,
    ) -> NDArray[np.complex64] | tuple[NDArray[np.complex64], list]:
        """Reconstruct using PnP method."""
        self.setup(shape=data_loader.shape)
        traj, data = data_loader.get_kspace_frame(0)
        kwargs = {}
        if "stacked" in self.nufft_backend:
            kwargs["z_index"] = "auto"
        if self.nufft_backend == "cufinufft":
            kwargs["smaps_cached"] = True

        final_images = np.zeros(
            (data_loader.n_frames, *data_loader.shape), dtype=np.complex64
        )
        metrics = [None] * data_loader.n_frames
        with torch.no_grad():
            physics = Nufft(
                data_loader.shape,
                data_loader.get_kspace_frame(0)[0],
                density=self.density_compensation,
                smaps=data_loader.get_smaps(),
                backend=self.nufft_backend,
            ).to(self.device)
            if constant:
                lipchitz_cst = physics.nufft.get_lipschitz_cst(max_iter=100)
                print("lipschitz_cst", lipchitz_cst)
            for i, traj, data in data_loader.iter_frames():
                physics.nufft.samples = traj
                if not constant:
                    lipchitz_cst = physics.nufft.get_lipschitz_cst(max_iter=100)

                data_torch = torch.from_numpy(data).to(self.device)
                extra_kwargs_optim = {
                    "custom_init": get_custom_init(
                        self.x_init, physics, data_torch, self.device
                    ),
                    "params_algo": get_DPIR_params(
                        **self.dpir_params,
                        lipschitz_cst=lipchitz_cst,
                        n_iter=self.max_iter_per_frame,
                    ),
                }
                optim: BaseOptim = self.init_optim(**extra_kwargs_optim).to(self.device)
                optim.fixed_point.show_progress_bar = True
                x_est = optim(
                    data_torch, physics=physics, compute_metrics=self.compute_metrics
                )
                # Setup optim iterator and solve for frame
                if self.compute_metrics:
                    final_images[i] = x_est[0].to("cpu").numpy()
                    metrics[i] = x_est[1]
                else:
                    final_images[i] = x_est.to("cpu").numpy()
            if self.compute_metrics:
                return final_images, metrics
        return final_images


def get_custom_init(
    x_init: str | NDArray | torch.Tensor,
    physics: Nufft,
    data: torch.Tensor,
    device: str,
):
    """Get the custom initialization function."""
    if isinstance(x_init, np.ndarray):
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
