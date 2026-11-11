
import hoomd
from hoomd import md

import numpy as np
from datetime import datetime
import argparse

# -----------------------------------
# System parameters
# -----------------------------------

N = 8100
phi_1 = 0.20
phi_2 = 0.90

dt = 1e-4
t_eq_1 = 100
t_eq_2 = 0
t_samp = 1e-4
N_image = 1

# Compression time from command line
parser = argparse.ArgumentParser()
parser.add_argument("tcomp", type=float)
args = parser.parse_args()

t_resize = args.tcomp

if t_resize <= 0:
    raise ValueError("tcomp must be positive")

# Output file
fileprefix = f"HS_N{N}_phi{phi_2:.2f}_tcomp{t_resize}"

# -----------------------------------
# Time steps
# -----------------------------------

N_eq_1 = int(np.round(t_eq_1 / dt))
N_resize = int(np.round(t_resize / dt))
N_eq_2 = int(np.round(t_eq_2 / dt))
N_samp = int(np.round(t_samp / dt))

N_imageperiod = max(1, int(np.round(N_samp / N_image)))

if N_resize < 1:
    raise ValueError("Compression time is shorter than one timestep")

# -----------------------------------
# Initialize simulation
# -----------------------------------

device = hoomd.device.CPU()
sim = hoomd.Simulation(
    device=device,
    seed=datetime.now().microsecond or 1
)

# Square lattice
N_unit = int(np.sqrt(N))

if N_unit**2 != N:
    raise ValueError("N must be a perfect square")

L_unit = np.sqrt(np.pi / phi_1)
L_1 = L_unit * N_unit
L_2 = L_1 * np.sqrt(phi_1 / phi_2)

# Create HOOMD snapshot
snapshot = hoomd.Snapshot()

if snapshot.communicator.rank == 0:
    snapshot.configuration.box = [L_1, L_1, 0, 0, 0, 0]
    snapshot.particles.N = N
    snapshot.particles.types = ["A"]

    # Center particles in the periodic box
    x = (np.arange(N_unit) + 0.5) * L_unit - L_1 / 2
    y = (np.arange(N_unit) + 0.5) * L_unit - L_1 / 2

    xx, yy = np.meshgrid(x, y, indexing="ij")

    positions = np.zeros((N, 3))
    positions[:, 0] = xx.ravel()
    positions[:, 1] = yy.ravel()

    snapshot.particles.position[:] = positions
    snapshot.particles.diameter[:] = 2.0
    snapshot.particles.typeid[:] = 0

sim.create_state_from_snapshot(snapshot)

# -----------------------------------
# Pair potential
# -----------------------------------

nl = md.nlist.Cell(buffer=0.4)

# Harmonic repulsive potential:
# U(r) = (2-r)^2 / (4*dt), r < 2
# F(r) = (2-r) / (2*dt), r < 2

r_min = 0.0
r_cut = 2.0
width = 1000

r = np.linspace(r_min, r_cut, width, endpoint=False)

U = (2.0 - r)**2 / (4.0 * dt)
F = (2.0 - r) / (2.0 * dt)

hs = md.pair.Table(nlist=nl)
hs.params[("A", "A")] = {
    "r_min": r_min,
    "U": U,
    "F": F
}
hs.r_cut[("A", "A")] = r_cut

# -----------------------------------
# Brownian dynamics
# -----------------------------------

brownian = md.methods.Brownian(
    filter=hoomd.filter.All(),
    kT=1.0
)
brownian.gamma["A"] = 1.0

integrator = md.Integrator(
    dt=dt,
    methods=[brownian],
    forces=[hs]
)

sim.operations.integrator = integrator

# -----------------------------------
# Equilibrate at phi_1
# -----------------------------------

print(f"Equilibrating at phi = {phi_1}")

sim.run(N_eq_1)

# -----------------------------------
# Compress box to phi_2
# -----------------------------------

print(f"Compressing from phi={phi_1} to phi={phi_2}")

box_initial = hoomd.Box(Lx=L_1, Ly=L_1, Lz=0)
box_final = hoomd.Box(Lx=L_2, Ly=L_2, Lz=0)

start_step = sim.timestep

ramp = hoomd.variant.Ramp(
    A=0.0,
    B=1.0,
    t_start=start_step,
    t_ramp=N_resize
)

box_variant = hoomd.variant.box.Interpolate(
    initial_box=box_initial,
    final_box=box_final,
    variant=ramp
)

resizer = hoomd.update.BoxResize(
    trigger=hoomd.trigger.Periodic(10),
    box=box_variant,
    filter=hoomd.filter.All()
)

sim.operations.updaters.append(resizer)

sim.run(N_resize)

# Remove updater after compression
sim.operations.updaters.remove(resizer)

# Ensure exact final box size
hoomd.update.BoxResize.update(
    state=sim.state,
    box=box_final,
    filter=hoomd.filter.All()
)

# -----------------------------------
# Equilibrate at phi_2
# -----------------------------------

print(f"Equilibrating at phi = {phi_2}")

if N_eq_2 > 0:
    sim.run(N_eq_2)

# -----------------------------------
# Save configurations
# -----------------------------------

gsd_writer = hoomd.write.GSD(
    filename=fileprefix + ".gsd",
    trigger=hoomd.trigger.Periodic(N_imageperiod),
    mode="wb",
    filter=hoomd.filter.All(),
    dynamic=["property"]
)

sim.operations.writers.append(gsd_writer)

# Sample
sim.run(N_samp + 1)

gsd_writer.flush()

print("Simulation complete")
print("Final timestep:", sim.timestep)
print("Output file:", fileprefix + ".gsd")
