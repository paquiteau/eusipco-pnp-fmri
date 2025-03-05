"""A minimal working example of a PnP reconstruction using DeepInv and SNAKE."""

# +
from pathlib import Path

from snake.core import GreConfig, Phantom, SimConfig
from snake.core.engine import NufftAcquisitionEngine
from snake.core.handlers import BlockActivationHandler
from snake.core.sampling import StackOfSpiralSampler
from snake.core.simulation import default_hardware
from snake.core.smaps import get_smaps
from snake.mrd_utils import NonCartesianFrameDataLoader
from snake.toolkit.plotting import axis3dcut


from seqpnp.sequentialpnp import SequentialPnPReconstructor

# +
# ## Setup the simulation configuration
# +
sim_conf = SimConfig(
    max_sim_time=300,
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
sim_conf.hardware.n_coils = 8  # Update to get multi coil results.
sim_conf.hardware.field = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")
phantom = phantom.resample(new_affine=sim_conf.fov.affine, new_shape=sim_conf.fov.shape, use_gpu=True)


sampler = StackOfSpiralSampler(
    accelz=4,
    acsz=0.1,
    orderz="top-down",
    nb_revolutions=12,
    obs_time_ms=25,
    constant=True,
    spiral_name="archimedes"
)
activation_handler = BlockActivationHandler(block_on=20, block_off=20, duration=300,atlas_label=48)


# +


engine = NufftAcquisitionEngine(model="T2s", snr=500)

engine(
    "example_spiral_full.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[activation_handler],
    worker_chunk_size=60,
    n_workers=6,
    nufft_backend="stacked-cufinufft",
)


# +
from snake.toolkit.reconstructors import ZeroFilledReconstructor

zer_rec = ZeroFilledReconstructor(
    nufft_backend="stacked-gpunufft", density_compensation="pipe"
)
with NonCartesianFrameDataLoader("example_spiral_full.mrd") as data_loader:
    adj_data = zer_rec.reconstruct(data_loader)
    traj, _ = data_loader.get_kspace_frame(0)


# -

model_weights = "weights/drunet_3d_2602.pth"

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
import numpy as np


# +
density_compensation=None #"pipe"
nufft_backend="stacked-cufinufft"
x_init="pinv"
model_weights = "weights/drunet_3d_0303.pth"

with NonCartesianFrameDataLoader("example_spiral_full.mrd") as data_loader:
    rec = SequentialPnPReconstructor(
        model_weights=model_weights,
        nufft_backend=nufft_backend,
        x_init=x_init,
        dpir_params = {'xi': 0.9735980562499394, 'stepsize': 12.797325331814873, 'sigma': 0.0038954273339963873},
        density_compensation=density_compensation,
        max_iter_per_frame=20,
        optimizer="hqs-f1",
    )
    rec_data=rec.reconstruct(data_loader)
np.save("rec_pnp.npy", rec_data)
# -

import numpy as np
np.save("rec_pnp.npy", rec_data)

from snake.toolkit.analysis.stats import contrast_zscore, get_scores 
rec_data = np.load("rec_pnp.npy")


# +
with NonCartesianFrameDataLoader("example_spiral_full.mrd") as data_loader:
    sim_conf = data_loader.get_sim_conf()
    n_shots = data_loader.get_kspace_frame(0,shot_dim=True)[0].shape[0]
    TR_vol = n_shots * sim_conf.seq.TR /1000
    bold_sample_time = sim_conf.seq.TR
    dyn_datas = data_loader.get_all_dynamic()
    print(dyn_datas)
    waveform_name = f"activation-block_on"
    good_d = None
    for d in dyn_datas:
        if d.name == waveform_name:
            good_d = d
    if good_d is None:
        raise ValueError("No dynamic data found matching waveform name")

    bold_signal = good_d.data[0]
    bold_sample_time = np.arange(len(bold_signal)) * sim_conf.seq.TR / 1000

    phantom = data_loader.get_phantom()
    roi = phantom.masks[-1]

    
z_score = contrast_zscore(
    rec_data, TR_vol, bold_signal, bold_sample_time, "block_on"
)
np.save("rec_pnp_z_score.npy", z_score)

# -

n_shots

axis3dcut(abs(np.mean(rec_data,0)).T, z_score.T, roi.T)

ts_roi = abs(rec_data[..., roi> 0.2])

import matplotlib.pyplot as plt
plt.plot(np.mean(ts_roi,-1))
plt.title("Average temporal signal in ROI")

plt.subplot(211)
plt.plot(np.arange(76)*TR_vol, np.mean(ts_roi, -1))
plt.subplot(212)
plt.plot(bold_sample_time, bold_signal)

len(bold_signal)

TR_vol


