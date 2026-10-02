# CDM_Ewald

**Quickstart**

* To install, run `pip install -e .` from this directory

**Electrochemical doping: density and eta scans**

`test/electrochemical_doping_multilayers/constant_ne_vary_eta.py` now scans both
electron-density ratios (strictly greater than 1) and eta for a single layer.
The adjacent `vary_ne_eta_multilayers.py` scans 11 layers with lateral shift
constants from 0 to 0.5 in steps of 0.1. Original output formats are retained in
separate density/shift folders, with complete geometry metadata and source
snapshots. See [the scan guide](docs/electrochemical_doping_scan.md) for commands,
defaults, output layout, and tests.

**Finite silver twisted bilayer (no Ewald summation)**

Open `test/bilayers/silver_twisted_bilayer_direct_50x50.ipynb` with a Python
kernel containing NumPy, SciPy >= 1.12, and Matplotlib. It uses the local silver
CSV and `src/cdm_ewald/finite_direct.py` to solve two finite 50 × 50 primitive
lattice patches (5,000 particles), with the original twist, 50 nm sphere radius,
200 nm spacing, and 100 nm layer separation. All free-space particle pairs are
included through a blocked direct Green-function application and GMRES; work
per iteration is quadratic in particle count. The full spectrum can be slow;
`WAVELENGTHS_NM` selects a shorter spectrum and `SIDE` a smaller validation array.

The notebook saves complex dipoles, optical-theorem extinction cross sections,
convergence diagnostics, and plots under
`outputs/silver_twisted_bilayer_direct_50x50/`. Extinction is reported in nm² and
normalized by one layer's nominal area; it is not the periodic `1 - |t|²` metric.
Each completed wavelength is saved before the next solve. Unconverged solves
raise an error. Test the direct solver with
`PYTHONPATH=src python -m unittest discover -s tests -p 'test_finite_direct.py'`.
