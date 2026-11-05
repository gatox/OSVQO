# OS-VQO challenge module

This is the challenge-focused extraction of the OS-VQO path from the larger `NOFVQE` implementation. It exposes a single public calculator class:

```python
from osvqo import OSVQO

geometry = """units angstrom
0 1
H 0.0 0.0 0.0
H 0.0 0.0 0.735
"""

calc = OSVQO(
    geometry,
    functional="pnof7",
    basis="sto-3g",
    dev="simulator",
    opt_circ="slsqp",
    max_iterations=300,
)
result = calc.run()
energy_total, params, rdm1, occupations, orbital_vectors, cj12, ck12, C_MO, lagrange = result
analytic_gradient = calc.nuclear_gradient()
```

`run()` performs OS-VQO occupation optimization together with the classical natural-orbital optimization. The returned energy is the selected PNOF total energy. `nuclear_gradient()` uses the retained analytical Psi4 gradient; finite-difference gradient paths are not included.

## Devices

- `dev="simulator"` uses the analytic OS-VQO occupation parametrization.
- `dev="noise_simulator"` and `dev="real"` use the existing PennyLane Qiskit-device integration and OS-VQO occupation sampling.
- `dev="hybrid_real"` or `dev="hybrid_noise_simulator"` optimizes on the simulator and evaluates the optimized parameters on the selected device.

The finite-shot path requires a configured IBM Quantum Runtime account and the compatible PennyLane Qiskit plugin. OS-VQO readout, logical-subspace, response, and residual-floor options remain available; the old ON-VQE error-mitigation path is removed.

## Dependencies

Install compatible versions of PennyLane, JAX, Psi4, NumPy, SciPy, Qiskit, `pennylane-qiskit`, and `qiskit-ibm-runtime`. `qiskit-aer` is additionally required for the backend-derived noise simulator. The local analytic path does not import IBM Runtime modules until a remote device is selected.

## Run from the terminal

The module can be run directly with the same positional argument order as the previous script:

```bash
python ../../osvqo.py h2_bohr.xyz pnof7 sto-3g T
```

Arguments are `geometry_file functional basis save_params [layout]`. `save_params` is `T` to keep the `.npy` output files or `F` to remove `.npy` files and `timer.dat` after the calculation. The optional layout accepts `h2o`, `li2`, or `lih` and supplies the corresponding hardware qubit layout. The CLI defaults to `dev="simulator"` and uses SLSQP; edit `_main()` to select a hardware/noise device, for which the optimizer switches to COBYLA.

## Printed occupation-space and circuit information

At the start of a calculation, `OSVQO` prints each pair subspace, its orbital indices, occupation-space dimension `d=m+1`, `m`, and its logical qubit requirement `ceil(log2(d))`. It also prints total logical qubits and OS parameters. The reported logical ansatz gate count and depth are computed from the parameterized two-level rotations without running a circuit, so this diagnostic adds negligible work when `dev="simulator"`. Depth assumes distinct pair registers can run in parallel. These are pre-transpilation logical resources; routing and decomposition for a particular backend can increase physical gate count and depth.

The run header includes the manuscript citation:

> Implemented by E. X. Salazar, M. Piris, *Occupation-space variational quantum optimization* (2026). doi:10.26434/chemrxiv.15008226

## IBM Quantum Runtime account profile

For `dev="real"`, `dev="noise_simulator"`, `dev="hybrid_real"`, or `dev="hybrid_noise_simulator"`, the code creates the Runtime service internally; you do not need to add a `QiskitRuntimeService(...)` call to the calculation script. The selected profile must already be saved in the local Qiskit Runtime account configuration. The default profile name for the default `eu-de` region is `qff_2026`, matching `QiskitRuntimeService(name="qff_2026")`. You can choose another saved profile in Python with `runtime_service_name="your_profile"`, or from the command line with `--runtime-service-name your_profile`. For `us-east`, the default remains `us-east-ibm-quantum-platform`.

The command-line interface accepts a device option, for example:

```bash
python osvqo.py n2_bohr.xyz pnof8m sto-3g T --device noise_simulator
python osvqo.py n2_bohr.xyz pnof8m sto-3g T --device real --runtime-service-name qff_2026
```

The default remains `--device simulator`.

## Occupation-space and circuit diagnostics

The calculation header prints a citation to the OS-VQO manuscript. After Psi4 determines the orbital space, the program reports each occupation pair's orbital indices, dimension `d`, `m=d-1`, and logical qubit requirement `ceil(log2(d))`, followed by totals. It also prints a lightweight logical ansatz resource estimate: the parameterized two-level rotation count and scheduled depth assuming different pair registers can run in parallel. This estimate is computed from the layout and does not construct or execute a circuit, so it does not add meaningful runtime to `dev="simulator"`. These are logical, pre-transpilation resources; hardware compilation, routing, and decomposition can increase gate counts and depth.

## IBM Runtime profile

For hardware and hybrid modes, `OSVQO` initializes `QiskitRuntimeService` internally. You do not need to add a separate service construction call to your calculation script. The credentials must be saved in the local Qiskit Runtime configuration under the profile selected by the code. For the default `eu-de` region, that profile is `qff_2026` (equivalent to selecting it with `QiskitRuntimeService(name="qff_2026")`). The default `us-east` profile is `us-east-ibm-quantum-platform`.

You can override the selected saved profile in Python by passing `runtime_service_name="profile_name"`, or from the CLI with `--runtime-service-name profile_name`. For example:

```bash
python osvqo.py n2_bohr.xyz pnof8m sto-3g T --device noise_simulator
python osvqo.py n2_bohr.xyz pnof8m sto-3g T --device real --runtime-service-name qff_2026
```

Qiskit Runtime supports saving credentials under a named account profile and selecting that name when initializing the service; see the [IBM Quantum account setup documentation](https://docs.quantum.ibm.com/guides/setup-channel).
