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
    max_sim_time=3,
    seq=GreConfig(TR=50, TE=22, FA=12),
    hardware=default_hardware,
    fov_mm=(181, 217, 181),
    shape=(60, 72, 60),
)
sim_conf.hardware.n_coils = 1  # Update to get multi coil results.
sim_conf.hardware.field_strength = 7
phantom = Phantom.from_brainweb(sub_id=4, sim_conf=sim_conf, tissue_file="tissue_7T")


sampler = StackOfSpiralSampler(
    accelz=2,
    acsz=0.1,
    orderz="top-down",
    nb_revolutions=12,
    obs_time_ms=30,
    constant=True,
)

smaps = None
if sim_conf.hardware.n_coils > 1:
    smaps = get_smaps(sim_conf.shape, n_coils=sim_conf.hardware.n_coils)

fov_handler = FOVHandler(
    center=(90, 110, 100),
    size=(192, 192, 128),
    angles=(5, 0, 0),
    target_res=(2.0, 2.0, 2.0),
)

# %%
#

engine = NufftAcquisitionEngine(model="T2s", snr=30000)

engine(
    "example_spiral.mrd",
    sampler,
    phantom,
    sim_conf,
    handlers=[fov_handler],
    smaps=smaps,
    worker_chunk_size=60,
    n_workers=1,
    nufft_backend="cufinufft",
)

with NonCartesianFrameDataLoader("example_spiral.mrd") as data_loader:

    rec_pnp = SequentialPnPReconstructor(
        model_weights=Path(__file__).parent / "weights" / "drunet_3d_0602.pth"
    )

    rec_data = rec_pnp(data_loader)

axis3dcut(abs(rec_data[0]), None, None, cuts=(0.5, 0.3, 0.5))
