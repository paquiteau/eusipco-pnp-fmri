import torch
from deepinv.physics import LinearPhysics
from mrinufft.density.geometry_based import voronoi

import mrinufft

# NufftOperator = mrinufft.get_operator("finufft")

# NufftOperator = mrinufft.get_operator("cufinufft")


class Nufft(LinearPhysics):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def __init__(
        self,
        img_size,
        samples_loc,
        density=None,
        real=False,
        smaps=None,
        backend="cufinufft",
        **kwargs
    ):
        super(Nufft, self).__init__(**kwargs)
        
        self.real = real  # Whether to project the data on real images
        n_coils = 1
        if smaps is not None:
            n_coils = len(smaps)
        self.nufft = mrinufft.get_operator(backend)(samples_loc.reshape(-1, len(img_size)), shape=img_size, density=density, n_coils=n_coils, squeeze_dims=False, smaps = smaps)

    def A(self, x):
        return self.nufft.op(x)

    def A_adjoint(self, kspace):
        return self.nufft.adj_op(kspace)

