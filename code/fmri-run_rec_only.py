# +
from snake.toolkit.reconstructors import ZeroFilledReconstructor
from snake.mrd_utils import NonCartesianFrameDataLoader

zer_rec = ZeroFilledReconstructor(
    nufft_backend="gpunufft", density_compensation="pipe"
)
with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    adj_data = zer_rec.reconstruct(data_loader)
    traj, _ = data_loader.get_kspace_frame(0)


# -

len(adj_data)

axis3dcut(abs(adj_data[0].T), None, None, cuts=(0.5, 0.5, 0.5))

# +
model_weights = "weights/drunet_3d_0602.pth"

denoiser = load_drunet_mri(
     model_weights, device=device, dim=3, norm_factor= 1
)
# -

device="cuda"
density_compensation=None #"pipe"
nufft_backend="stacked-cufinufft"
x_init="pinv"
dpir_params = {"xi":0.75, "stepsize":"1", "lamb":100, "sigma":1e-3}
max_iter_per_frame=8
optimizer="hqs"

import matplotlib.pyplot as plt
from time import perf_counter
from seqpnp.sequentialpnp import SequentialPnPReconstructor

with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:
    rec = SequentialPnPReconstructor(
        model_weights=model_weights,
        nufft_backend=nufft_backend,
        x_init="adjoint",
        dpir_params= dpir_params,
        max_iter_per_frame=8)
    rec_data = rec.reconstruct(data_loader)





