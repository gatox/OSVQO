import time
import pennylane as qml
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

import psi4
psi4.core.be_quiet()
from scipy.linalg import eigh, expm
from scipy.optimize import minimize
import re






class OSVQO:
    """Occupation-Space Variational Quantum Optimization with PNOF orbital optimization.

    This challenge-focused class exposes only OS-VQO runs and the analytical
    nuclear gradient. Quantum circuit execution uses the existing PennyLane
    Qiskit-device integration; Psi4 supplies integrals and analytic derivatives.
    """

    def __init__(self, geometry, functional='pnof4', conv_tol=1e-5, init_param=None, basis='sto-3g', max_iterations=1000, C_MO=None, dev='simulator', opt_circ='slsqp', initial_layout=None, n_shots=None, optimization_level=3, resilience_level=None, os_readout_mitigation=False, os_error_mitigation=True, noisy_parameter_tol=1e-3, noisy_rhobeg=0.1, runtime_service_name=None):
        self.units, self.charge, self.mul, self.symbols, self.crd, self.xyz_str, self.geom_jax = self._read_mol(geometry)
        self.basis = basis
        self.functional = functional
        self.natoms = len(self.symbols)
        self.ipnof = None
        if functional in ['pnof4', 'pnof5', 'pnof7', 'pnof8', 'pnof8m']:
            self.ipnof = self._func_indix(functional)
        self.conv_tol = conv_tol
        self.max_iterations = max_iterations
        self.opt_param = None
        self.opt_rdm1 = None
        self.opt_n = None
        self.opt_vecs = None
        self.opt_cj12 = None
        self.opt_ck12 = None
        self.C_MO = C_MO
        self.H_ao = None
        self.I_ao = None
        self.b_mnl = None
        self.C = None
        self.initial_layout = initial_layout
        self.runtime_service_name = runtime_service_name
        self.os_use_specialized_small_m = True
        self.os_subspace_postselection = True
        self.os_residual_floor_mitigation = False
        self.os_residual_floor_dominant_threshold = 0.995
        self.os_residual_floor_sigma = 2.0
        self.os_residual_floor_angle_eps = 1e-06
        self._os_residual_floor = None
        self.os_readout_mitigation = os_readout_mitigation
        self.os_subspace_postselection = True
        self.os_response_mitigation = True
        self.os_response_m_values = (3,)
        self._os_response_matrices = None
        self._label = 'OS-VQO'
        self._sub_label = 'VQO'
        self.os_error_mitigation = bool(os_error_mitigation)
        self.noisy_parameter_tol = float(noisy_parameter_tol)
        self.noisy_rhobeg = float(noisy_rhobeg)
        if not self.os_error_mitigation:
            self.os_readout_mitigation = False
            self.os_subspace_postselection = False
            self.os_response_mitigation = False
            self.os_residual_floor_mitigation = False
        self.mps = False
        self.region = 'eu-de'
        self.verbose = 0
        self.nthreshl = 4
        self.threshl = 10.0 ** (-self.nthreshl)
        self.maxloop = 10
        self.energy_scale = 1000.0
        self.pair_doubles = True
        if hasattr(dev, '_main_answer'):
            self.dev = dev._main_answer
        else:
            self.dev = dev
        if self.dev != 'simulator':
            self.n_shots = n_shots
            self.optimization_level = optimization_level
            self.resilience_level = resilience_level
        else:
            self.n_shots = None
        self.opt_circ = opt_circ
        if self.functional not in ['pnof4', 'pnof5', 'pnof7', 'pnof8', 'pnof8m']:
            raise ValueError('OSVQO supports PNOF4, PNOF5, PNOF7, PNOF8, or PNOF8m.')
        self.init_param = init_param

    def _run_osvqo(self, max_outer=120):
        tol = self.conv_tol
        eva_start = time.perf_counter()
        "Self-consistent OS-VQO loop: classical orbitals and occupations."
        table_width = 95
        print('\n' + '-' * table_width)
        print(' ' * 47 + self._label)
        print(' ' * 28 + '(Natural-orbital and occupation optimization)')
        print('-' * table_width)
        print(' ' * 31 + 'Implemented by E. X. Salazar, M. Piris\n')
        print('Ref. Occupation-space variational quantum optimization (2026). doi:10.26434/chemrxiv.15008226')
        print('')
        print('functional:', self.functional)
        print('basis_set:', self.basis)
        print('OS-VQO optimizer:', self.opt_circ)
        print('NO optimizer: ADAM')
        E_old = None
        E_orb_old = None
        E_total_old = None
        E_HF = None
        rdm1 = None
        E_total_old = None
        S, H, I, E_nuc, wfn_mol = self._ao_integrals(self.crd)
        self._job_time(eva_start, 'Total AO integrals time')
        self.E_nuc = E_nuc
        self._global_parameters(wfn_mol)
        init_param = self.init_param
        print('init_param:', init_param)
        C_MO = jnp.array(self.C_MO) if self.C_MO is not None else None
        if C_MO is None:
            print('Computing HF energy and C_MO')
            hf_start = time.perf_counter()
            psi4.set_options({'basis': self.basis, 'scf_type': 'pk', 'reference': 'rhf', 'maxiter': 600, 'guess': 'sad', 'diis': True, 'damping_percentage': 20.0, 'level_shift': 0.5})
            E_HF, wfn_HF = psi4.energy('HF', return_wfn=True)
            C_HF = wfn_HF.Ca().np
            print('HF energy:', E_HF)
            self._job_time(hf_start, 'HF energy time')
            C_MO = C_HF
        else:
            print('Using provided C_MO')
            C_old = C_MO.copy()
            for i in range(self.ndoc):
                for j in range(self.ncwo):
                    k = self.no1 + self.ndns + (self.ndoc - i - 1) * self.ncwo + j
                    l = self.no1 + self.ndns + (self.ndoc - i - 1) + j * self.ndoc
                    C_MO = C_MO.at[:, k].set(C_old[:, l])
        C_MO = self._check_ortho(C_MO, S)
        self._prepare_backend()
        table_width = 95
        print('\n' + '=' * table_width)
        print(' ' * 35 + 'SC-OS-VQO OPTIMIZATION')
        print('=' * table_width)
        print(f"{'It':>4} {'OccIt':>7} {'OrbIt':>7} {'MaxOrb':>7} {'Total Energy':>20} {'Delta Etot':>14} {'Delta Eocc':>14} {'MaxDiff':>12}")
        print('-' * table_width)
        for it in range(max_outer):
            delta_E_occ = None
            delta_E_total = None
            if self.verbose > 1:
                print(f'\n==== SC-OS-VQO Iteration {it} ====')
            self.init_param = self._initial_params(init_param)
            self.J_MO, self.K_MO, self.H_core = self._JKH_MO_Full(C_MO, H, I)
            if self.verbose > 1:
                print('OS-VQO parameter optimization')
            E_vqe, params, rdm1, n, vecs, cj12, ck12, occ_nit = self._ene_vqe(self.init_param)
            if self.verbose > 1:
                print(f'OS-VQO energy: {E_vqe:.12f} Ha')
            if E_orb_old is not None:
                E_orb_old_total = E_orb_old + self.E_nuc
                delta_E_occ = E_vqe - E_orb_old_total
                if self.verbose > 1:
                    print(f'Occupation energy change: {delta_E_occ:.10e} Ha')
                if delta_E_occ > -1e-06:
                    self.maxloop += 10
                    if self.verbose > 1:
                        print(f'Increasing maximum orbital iterations to {self.maxloop}')
            if self.verbose > 1:
                print(f'Orbital Optimization (ADAM), max iterations = {self.maxloop}')
            E_orb, C_new, orb_nit, sumdiff, maxdiff = self._orbital_optimization(n, cj12, ck12, C_MO, H, I)
            if self.verbose > 1:
                print(f'Orbital optimization energy: {self.E_nuc + E_orb:.10f} Ha')
            E_total = self.E_nuc + E_orb
            if E_total_old is None:
                delta_E_total = None
                convg_energy = False
            else:
                delta_E_total = E_total - E_total_old
                convg_energy = abs(delta_E_total) < tol
            convg_delag = maxdiff < self.threshl
            if self.verbose > 1:
                print(f'Orbital Lagrange convergence: maxdiff = {maxdiff:.6e}, threshold = {self.threshl:.1e}, CONVGDELAG = {convg_delag}')
                print('Updating initial parameters with optimal parameters', params)
            init_param = params
            if delta_E_total is None:
                delta_total_str = '---------'
            else:
                delta_total_str = f'{delta_E_total:.3e}'
            if delta_E_occ is None:
                delta_occ_str = '---------'
            else:
                delta_occ_str = f'{delta_E_occ:.3e}'
            print(f'{it + 1:4d} {occ_nit:7d} {orb_nit:7d} {self.maxloop:7d} {E_total:20.12f} {delta_total_str:>14} {delta_occ_str:>14} {maxdiff:12.3e}')
            if convg_delag and convg_energy:
                print('-' * table_width)
                print(f'SC-OS-VQO converged: MaxDiff = {maxdiff:.3e} < {self.threshl:.1e}, |Delta Etot| = {abs(delta_E_total):.3e} < {tol:.1e}')
                break
            E_total_old = E_total
            E_old = E_vqe
            E_orb_old = E_orb
            self.C_MO = C_new
            C_MO = C_new
        E_sim, elag_sim, _, _ = self._ENERGY1r(C_new, n, H, I, cj12, ck12)
        if self._is_hybrid_mode() or self.dev == 'simulator':
            np.save('osvqo_sim_params.npy', params)
            np.save('osvqo_sim_n.npy', n)
        else:
            np.save('osvqo_qc_params.npy', params)
            np.save('osvqo_qc_n.npy', n)
        np.save('osvqo_C.npy', C_MO)
        if not self._is_hybrid_mode():
            self.init_param = init_param
            self._write_results(n=n, elag=elag_sim, E=E_sim, E_HF=E_HF, E_nuc=E_nuc, source=self.dev, case_label='Optimized occupations', corrected=self._is_corrected_run())
            self._job_time(eva_start, 'Total job time', True)
            return (E_nuc + E_sim, params, rdm1, n, vecs, cj12, ck12, C_new, elag_sim)
        else:
            n_sim = n
            self.dev_old = self.dev
            target_dev = self._hybrid_target_dev()
            self.dev = target_dev
            self.opt_param = init_param
            print('==== Hybrid mode activated ====')
            print('Optimization device: simulator')
            print('Evaluation device:', self.dev)
            print('===============================')
            self._prepare_backend()
            n_eva, vecs_eva, cj12_eva, ck12_eva, rdm1_eva = self._pnof(self.opt_param, self.n_elec, self.norb, rdm1=None)
            E_eva, elag_eva, _, _ = self._ENERGY1r(C_new, n_eva, H, I, cj12_eva, ck12_eva)
            self.dev = self.dev_old
            self._prepare_backend()
            self.opt_rdm1 = rdm1_eva
            self.opt_n = n_eva
            self.opt_vecs = vecs_eva
            self.opt_cj12 = cj12_eva
            self.opt_ck12 = ck12_eva
            self._write_results(n=n_eva, elag=elag_eva, E=E_eva, E_HF=E_HF, E_nuc=E_nuc, source=f'{self.dev_old}->{target_dev}', case_label=f'{target_dev}-evaluated occupations', corrected=self._is_corrected_run(), E_sim=E_sim, n_sim=n_sim, elag_sim=elag_sim)
            self._job_time(eva_start, 'Total job time', True)
            return (E_nuc + E_eva, init_param, rdm1_eva, n_eva, vecs_eva, cj12_eva, ck12_eva, C_new, elag_eva)

    def _pnof7(self, params, n_elec, norb, rdm1=None, n_opt=None, n_raw=None):
        if rdm1 is None:
            rdm1 = self._evaluate_rdm1(params)
        n = rdm1
        vecs = jnp.eye(norb)
        if self.pair_doubles:
            assert n.ndim == 1, f'Occupation numbers have wrong shape {n.shape}'
            assert vecs.ndim == 2, f'Orbital matrix has wrong shape {vecs.shape}'
        if self.ista == 0:
            fi = n * (1 - n)
            fi = fi.at[fi <= 0].set(0)
            fi = jnp.sqrt(fi)
        else:
            fi = 2 * n * (1 - n)
        cj12 = 2.0 * jnp.outer(n, n)
        ck12 = jnp.outer(n, n) + jnp.outer(fi, fi)
        if self.MSpin == 0 and self.nsoc > 1:
            ck12 = ck12.at[self.nbeta:self.nalpha, self.nbeta:self.nalpha].set(2 * jnp.outer(n[self.nbeta:self.nalpha], n[self.nbeta:self.nalpha]))
        for l in range(self.ndoc):
            ldx = self.no1 + l
            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo
            n_strong = n[ldx]
            n_weak = n[ll:ul]
            cj12 = cj12.at[ldx, ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul, ldx].set(0.0)
            cj12 = cj12.at[ll:ul, ll:ul].set(0.0)
            ck12 = ck12.at[ldx, ll:ul].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul, ldx].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul, ll:ul].set(-jnp.sqrt(jnp.outer(n_weak, n_weak)))
        return (n, vecs, cj12, ck12, rdm1)

    def _vqe_opt_scipy(self, E_fn, method, params, max_iterations):
        """Keep each objective energy and its occupations from the same measurement."""
        noisy = self.dev in ('real', 'noise_simulator')
        method = method.lower()
        if noisy and method in ('slsqp', 'l-bfgs-b'):
            raise ValueError("Shot-based OS-VQO optimization: select opt_circ='cobyla'. The previous microscopic finite-difference step is unsuitable.")
        params_np = np.asarray(params, dtype=float).reshape(-1)
        bounds = self._os_parameter_bounds()
        self._validate_occupation_parameters(params_np)
        if len(bounds) != len(params_np):
            raise ValueError('Parameter and bound counts differ.')
        records = []

        def key(x):
            return np.asarray(x, dtype=float).reshape(-1).tobytes()

        def E_scipy(x):
            values = E_fn(jnp.asarray(x))
            records.append((key(x), np.asarray(x, dtype=float).copy(), values))
            if self.verbose > 1:
                print(f'Evaluation = {len(records) - 1}, Energy = {float(values[0]):.12f} Ha')
            return float(values[0]) * self.energy_scale
        analytic = False

        def grad_scipy(x):
            return self.energy_scale * np.asarray(jax.grad(lambda theta: E_fn(theta)[0])(jnp.asarray(x)), dtype=float)
        step = 0

        def callback(xk):
            nonlocal step
            matches = [r for r in records if r[0] == key(xk)]
            if matches and self.verbose > 1:
                print(f'Step = {step}, Energy = {float(matches[-1][2][0]):.12f} Ha (stored objective evaluation)')
            step += 1
        options = {'maxiter': max_iterations}
        tolerance = self.conv_tol
        if method == 'cobyla':
            tolerance = self.noisy_parameter_tol if noisy else self.conv_tol
            options.update(rhobeg=self.noisy_rhobeg if noisy else 0.1, tol=tolerance, catol=1e-08)
        elif method in ('slsqp', 'l-bfgs-b'):
            options.update(ftol=self.conv_tol, eps=1e-06)
        res = minimize(E_scipy, params_np, method=method.upper(), jac=grad_scipy if analytic else None, bounds=bounds, tol=tolerance, options=options, callback=callback)
        final_params = np.asarray(res.x, dtype=float).reshape(-1)
        self._validate_occupation_parameters(final_params)
        matches = [r for r in records if r[0] == key(res.x)]
        if not matches:
            raise RuntimeError('Optimizer returned an unevaluated parameter vector.')
        chosen = min(matches, key=lambda r: abs(float(r[2][0]) * self.energy_scale - float(res.fun)))
        E_val, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = chosen[2]
        self.optimization_history = [{'evaluation': i, 'params': r[1].copy(), 'energy_total': float(r[2][0]), 'n': np.asarray(r[2][2]).copy()} for i, r in enumerate(records)]
        self.last_optimizer_result = res
        if self.verbose > 1:
            print(f'Optimizer success = {res.success}; message = {res.message}')
            print(f'Optimizer stored total energy = {float(E_val):.12f} Ha')
        return ([E_val], [final_params], [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val], int(getattr(res, 'nit', step)))

    def _evaluate_rdm1(self, params):
        """Return orbital occupations from the analytic OS map or measured pair registers."""
        if self.dev in ['simulator', 'hybrid_real', 'hybrid_noise_simulator']:
            return self._os_occupations_from_theta(params)
        probabilities_raw = self._os_qnode(params)
        self._last_os_raw_probabilities = probabilities_raw
        if not self.os_error_mitigation:
            return self._decode_os_probabilities(probabilities_raw)
        probabilities = probabilities_raw
        if self.os_readout_mitigation:
            probabilities = self._mitigate_os_readout(probabilities)
        self._last_os_readout_probabilities = probabilities
        self._report_os_readout_mitigation(probabilities_raw, probabilities)
        if self.os_subspace_postselection:
            probabilities, stats = self._os_postselect_probabilities(probabilities)
            self._last_os_probability_stats = stats
            self._report_os_subspace_stats(stats)
        self._last_os_postselected_probabilities = probabilities
        if self.os_response_mitigation:
            matrices = getattr(self, '_os_response_matrices', None)
            if matrices is None:
                matrices = self._build_os_response_matrices(params)
            probabilities = self._apply_os_response_mitigation(probabilities, matrices)
        self._last_os_response_probabilities = probabilities
        if self.os_residual_floor_mitigation:
            floor = self._build_os_residual_floor(params)
            probabilities = self._apply_os_residual_floor(probabilities, floor)
        self._last_os_corrected_probabilities = probabilities
        return self._decode_os_probabilities(probabilities)

    def _pnof8(self, params, n_elec, norb, rdm1=None, n_opt=None, n_raw=None):
        if rdm1 is None:
            rdm1 = self._evaluate_rdm1(params)
        n = rdm1
        vecs = jnp.eye(norb)
        if self.pair_doubles:
            assert n.ndim == 1, f'Occupation numbers have wrong shape {n.shape}'
            assert vecs.ndim == 2, f'Orbital matrix has wrong shape {vecs.shape}'
        h_cut = 0.02 * jnp.sqrt(2.0)
        n_d = jnp.zeros_like(n)
        for i in range(self.ndoc):
            idx = self.no1 + i
            ll = self.no1 + self.ndns + (self.ndoc - i - 1) * self.ncwo
            ul = ll + self.ncwo
            n_strong = n[idx]
            n_weak = n[ll:ul]
            h = 1.0 - n_strong
            coc = h / h_cut
            F = jnp.exp(-coc ** 2)
            n_d = n_d.at[idx].set(n_strong * F)
            n_d = n_d.at[ll:ul].set(n_weak * F)
        n_d12 = jnp.sqrt(n_d)
        fi = n * (1 - n)
        fi = fi.at[fi <= 0].set(0)
        fi = jnp.sqrt(fi)
        cj12 = 2.0 * jnp.outer(n, n)
        if self.imod == 0:
            ck12 = jnp.outer(n, n)
            ck12 = ck12.at[self.no1:self.nbeta, self.nalpha:].add(jnp.outer(fi[self.no1:self.nbeta], fi[self.nalpha:]))
            ck12 = ck12.at[self.nalpha:, self.no1:self.nbeta].add(jnp.outer(fi[self.nalpha:], fi[self.no1:self.nbeta]))
            ck12 = ck12.at[self.nalpha:, self.nalpha:].add(jnp.outer(fi[self.nalpha:], fi[self.nalpha:]))
            if self.MSpin == 0 and self.nsoc > 0:
                half = jnp.full((self.nsoc,), 0.5)
                ck12 = ck12.at[self.no1:self.nbeta, self.nbeta:self.nalpha].add(0.5 * jnp.outer(fi[self.no1:self.nbeta], half))
                ck12 = ck12.at[self.nbeta:self.nalpha, self.no1:self.nbeta].add(0.5 * jnp.outer(half, fi[self.no1:self.nbeta]))
                ck12 = ck12.at[self.nbeta:self.nalpha, self.nalpha:].add(jnp.outer(half, fi[self.nalpha:]))
                ck12 = ck12.at[self.nalpha:, self.nbeta:self.nalpha].add(jnp.outer(fi[self.nalpha:], half))
            if self.MSpin == 0 and self.nsoc > 1:
                ck12 = ck12.at[self.nbeta:self.nalpha, self.nbeta:self.nalpha].set(0.5)
        elif self.imod == 1:
            ck12 = jnp.outer(n, n) + jnp.outer(fi, fi)
        else:
            raise ValueError(f'Unsupported GNOF version: imod={self.imod}')
        ck12 = ck12.at[self.no1:self.nbeta, self.nalpha:].add(jnp.outer(n_d12[self.no1:self.nbeta], n_d12[self.nalpha:]) - jnp.outer(n_d[self.no1:self.nbeta], n_d[self.nalpha:]))
        ck12 = ck12.at[self.nalpha:, self.no1:self.nbeta].add(jnp.outer(n_d12[self.nalpha:], n_d12[self.no1:self.nbeta]) - jnp.outer(n_d[self.nalpha:], n_d[self.no1:self.nbeta]))
        ck12 = ck12.at[self.nalpha:, self.nalpha:].add(-jnp.outer(n_d12[self.nalpha:], n_d12[self.nalpha:]) - jnp.outer(n_d[self.nalpha:], n_d[self.nalpha:]))
        for l in range(self.ndoc):
            ldx = self.no1 + l
            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo
            n_strong = n[ldx]
            n_weak = n[ll:ul]
            cj12 = cj12.at[ldx, ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul, ldx].set(0.0)
            cj12 = cj12.at[ll:ul, ll:ul].set(0.0)
            exch = jnp.sqrt(n_strong * n_weak)
            ck12 = ck12.at[ldx, ll:ul].set(exch)
            ck12 = ck12.at[ll:ul, ldx].set(exch)
            ck12 = ck12.at[ll:ul, ll:ul].set(-jnp.sqrt(jnp.outer(n_weak, n_weak)))
        return (n, vecs, cj12, ck12, rdm1)

    def _validate_occupation_parameters(self, params):
        params = np.asarray(params, dtype=float).reshape(-1)
        bounds = self._os_parameter_bounds()
        if len(params) != len(bounds):
            raise ValueError(f'Expected {len(bounds)} OS-VQO parameters; received {len(params)}.')
        for idx, (value, (lower, upper)) in enumerate(zip(params, bounds)):
            if value < lower - 1e-10 or value > upper + 1e-10:
                raise ValueError(f'OS-VQO parameter {idx}={value} outside [{lower}, {upper}].')

    def _pnof5(self, params, n_elec, norb, rdm1=None, n_opt=None, n_raw=None):
        if rdm1 is None:
            rdm1 = self._evaluate_rdm1(params)
        n = rdm1
        vecs = jnp.eye(norb)
        if self.pair_doubles:
            assert n.ndim == 1, f'Occupation numbers have wrong shape {n.shape}'
            assert vecs.ndim == 2, f'Orbital matrix has wrong shape {vecs.shape}'
        cj12 = 2.0 * jnp.outer(n, n)
        ck12 = jnp.outer(n, n)
        if self.MSpin == 0 and self.nsoc > 1:
            ck12 = ck12.at[self.nbeta:self.nalpha, self.nbeta:self.nalpha].set(2 * jnp.outer(n[self.nbeta:self.nalpha], n[self.nbeta:self.nalpha]))
        for l in range(self.ndoc):
            ldx = self.no1 + l
            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo
            n_strong = n[ldx]
            n_weak = n[ll:ul]
            cj12 = cj12.at[ldx, ll:ul].set(0.0)
            cj12 = cj12.at[ll:ul, ldx].set(0.0)
            cj12 = cj12.at[ll:ul, ll:ul].set(0.0)
            ck12 = ck12.at[ldx, ll:ul].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul, ldx].set(jnp.sqrt(n_strong * n_weak))
            ck12 = ck12.at[ll:ul, ll:ul].set(-jnp.sqrt(jnp.outer(n_weak, n_weak)))
        return (n, vecs, cj12, ck12, rdm1)

    def _initial_params(self, params):
        if params is None:
            return np.full(self._count_os_parameters(), 0.1, dtype=float)
        params = np.asarray(params, dtype=float).reshape(-1)
        self._validate_occupation_parameters(params)
        return params

    def _is_corrected_run(self):
        noisy = self._is_hybrid_mode() or self.dev in ('real', 'noise_simulator')
        return bool(noisy and self.os_error_mitigation and any((self.os_readout_mitigation, self.os_subspace_postselection, self.os_response_mitigation, self.os_residual_floor_mitigation)))

    def _ene_vqe(self, params=None):
        """Optimize the PNOF energy over the OS-VQO occupation angles."""
        out = self._vqe(self.ene_nof, params)
        histories = out
        energies, parameters, rdm1s, occupations, vecs, cj12s, ck12s, *extra = histories
        self.opt_param = parameters[-1]
        self.opt_rdm1 = rdm1s[-1]
        self.opt_n = occupations[-1]
        self.opt_vecs = vecs[-1]
        self.opt_cj12 = cj12s[-1]
        self.opt_ck12 = ck12s[-1]
        return (energies[-1], parameters[-1], rdm1s[-1], occupations[-1], vecs[-1], cj12s[-1], ck12s[-1], extra[0] if extra else len(energies))

    def _build_qnodes(self):
        """Build the OS-VQO probability circuit for finite-shot devices."""
        dev = self.dev_instance
        if not hasattr(self, '_os_layout'):
            self._os_layout = self._build_os_layout()
        if self.dev in ['simulator', 'hybrid_real', 'hybrid_noise_simulator']:
            print('OS-VQO analytic simulator activated; occupations are evaluated analytically.')
            return
        if not hasattr(self, '_os_qnode'):
            layouts = self._os_layout

            @qml.qnode(dev, interface='jax')
            def os_qnode(theta):
                theta = jnp.asarray(theta)
                for layout in layouts:
                    self._prepare_os_pair(theta[layout['parameter_slice']], layout)
                return tuple((qml.probs(wires=layout['wires']) for layout in layouts))
            self._os_qnode = os_qnode
        print('OS-VQO quantum circuit activated')
        print('OS logical qubits:', self._count_os_qubits())
        print('OS parameters:', self._count_os_parameters())

    def _vqe(self, E_fn, params, method=None, max_iterations=None):
        method = (self.opt_circ if method is None else method).lower()
        max_iterations = self.max_iterations if max_iterations is None else max_iterations
        params = self._initial_params(params)
        if method in ('slsqp', 'l-bfgs-b', 'cobyla'):
            return self._vqe_opt_scipy(E_fn, method, params, max_iterations)
        if method == 'spsa':
            return self._vqe_opt_pennylane(E_fn, method, params, max_iterations)
        raise ValueError("Choose opt_circ from 'slsqp', 'l-bfgs-b', 'cobyla', or 'spsa'.")

    def _device(self, allow_fallback=False):
        region = self.region
        max_retries = 10
        retry_delay = 5
        qubits = self._count_os_qubits()
        self._logical_qubits = qubits
        if self.dev in ['simulator', 'hybrid_real', 'hybrid_noise_simulator']:
            if self.mps:
                dev = qml.device('default.tensor', wires=qubits, method='mps', max_bond_dim=10, contract='auto-mps')
            else:
                dev = qml.device('lightning.qubit', wires=qubits)
        else:
            # Keep cloud-only dependencies optional for local analytic runs.
            from qiskit_ibm_runtime import QiskitRuntimeService
            from ibm_cloud_sdk_core.api_exception import ApiException
            from qiskit_ibm_runtime.accounts.exceptions import InvalidAccountError
            if not hasattr(self, '_ibm_service'):
                if region not in ('eu-de', 'us-east'):
                    raise RuntimeError(f'Unsupported region: {region}')
                service_name = self.runtime_service_name or (
                    'qff_2026' if region == 'eu-de' else 'us-east-ibm-quantum-platform'
                )
                self._ibm_service = QiskitRuntimeService(name=service_name)
            for attempt in range(max_retries):
                try:
                    if self.dev == 'noise_simulator':
                        print('Calling noise_simulator:')
                        from qiskit_aer import AerSimulator
                        if region == 'eu-de':
                            real_backend = self._ibm_service.backend('ibm_basquecountry')
                        else:
                            real_backend = self._ibm_service.backend('ibm_pittsburgh')
                        self._ibm_backend = real_backend
                        backend = AerSimulator.from_backend(real_backend)
                        self._execution_backend = backend
                        backend.set_options(max_parallel_threads=0, max_parallel_experiments=0, max_parallel_shots=1, statevector_parallel_threshold=16)
                    elif self.dev == 'real':
                        print('Calling real QC:')
                        if region == 'eu-de':
                            backend = self._ibm_service.backend('ibm_basquecountry')
                        else:
                            backend = self._ibm_service.backend('ibm_kingston')
                        self._ibm_backend = backend
                        self._execution_backend = backend
                    print('')
                    print('Qiskit physical initial layout:')
                    print(self.initial_layout)
                    print('')
                    device_kwargs = {'wires': qubits, 'backend': backend, 'optimization_level': self.optimization_level, 'shots': self.n_shots, 'seed_transpiler': 12345}
                    if self.initial_layout is not None:
                        device_kwargs['initial_layout'] = self.initial_layout
                    dev = qml.device('qiskit.remote', **device_kwargs)
                    self._capture_compiled_circuits(dev)
                    if self.initial_layout is None:
                        print('Layout selection: automatic Qiskit transpiler layout (no user initial_layout supplied).')
                    break
                except (ApiException, InvalidAccountError) as e:
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] IBM Q call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(retry_delay)
            else:
                if allow_fallback:
                    print('IBM Q unavailable after retries. Falling back to local AerSimulator.')
                    from qiskit_aer import AerSimulator
                    backend = AerSimulator()
                    dev = qml.device('qiskit.aer', wires=backend.configuration().num_qubits, backend=backend, shots=self.n_shots)
                else:
                    raise RuntimeError('IBM Q unavailable after retries')
        return dev

    def _capture_compiled_circuits(self, device):
        """Keep the latest backend-transpiled Qiskit circuits for diagnostics.

        PennyLane-Qiskit compiles the circuit immediately before Runtime submission.
        Wrapping this device method records those compiled circuits without adding
        another transpilation or quantum execution.
        """
        if not hasattr(device, 'compile_circuits'):
            print('Transpiled circuit diagnostics unavailable: device exposes no compile_circuits method.')
            return
        if getattr(device, '_osvqo_compile_capture_installed', False):
            return

        original_compile = device.compile_circuits

        def compile_and_capture(circuits):
            compiled = original_compile(circuits)
            self._last_transpiled_circuits = compiled
            self._last_transpiled_backend = getattr(device, 'backend', None)
            return compiled

        try:
            device.compile_circuits = compile_and_capture
            device._osvqo_compile_capture_installed = True
        except Exception as exc:
            print(f'Transpiled circuit diagnostics unavailable ({type(exc).__name__}: {exc}).')

    def _report_transpiled_resources(self):
        """Print resources and final physical placement from the last compiled circuits."""
        circuits = getattr(self, '_last_transpiled_circuits', None)
        if not circuits:
            if self.dev != 'simulator':
                print('No transpiled Qiskit circuit was captured; check PennyLane-Qiskit compatibility.')
            return

        backend = getattr(self, '_last_transpiled_backend', None)
        if backend is not None:
            backend_name = getattr(backend, 'name', None)
            try:
                backend_name = backend_name() if callable(backend_name) else backend_name
            except Exception:
                pass
            print('Transpiled circuit backend:', backend_name or type(backend).__name__)

        print('OS-VQO transpiled circuit resources (latest compiled circuit batch):')
        for circuit_index, circuit in enumerate(circuits):
            print(f'  Circuit {circuit_index + 1}:')
            print(f'    Transpiled circuit width: {circuit.num_qubits} qubits')
            print(f'    Transpiled depth: {circuit.depth()}')
            print(f'    Transpiled gate counts: {dict(circuit.count_ops())}')
            layout = getattr(circuit, 'layout', None)
            if layout is None:
                print('    Final logical-to-physical mapping: unavailable (transpiler returned no layout metadata)')
                continue
            try:
                final_mapping = layout.final_index_layout(filter_ancillas=True)
                print('    Final logical-to-physical mapping:', dict(enumerate(final_mapping)))
                print('    Physical qubits used:', sorted(set(final_mapping)))
            except Exception:
                try:
                    initial_mapping = layout.initial_layout.get_virtual_bits()
                    print('    Initial virtual-to-physical mapping:', {str(q): p for q, p in initial_mapping.items()})
                    print('    Final mapping: unavailable from this Qiskit layout object')
                except Exception as exc:
                    print(f'    Layout metadata could not be decoded ({type(exc).__name__}: {exc})')

    def _global_parameters(self, wfn):
        """Set closed-shell system dimensions and construct OS-VQO pair spaces."""
        self.nalpha = wfn.nalpha()
        self.nbeta = wfn.nbeta()
        self.n_elec = self.nalpha + self.nbeta
        self.norb = wfn.nmo()
        self.nbf = self.norb
        self.no1 = max(0, self.n_elec - self.nbf)
        self.ndoc = self.nbeta - self.no1
        self.nsoc = self.nalpha - self.nbeta
        self.ndns = self.ndoc + self.nsoc
        self.nvir = self.nbf - self.nalpha
        self.ncwo = -1
        if self.ndns != 0:
            self.ncwo = 0 if self.ndoc <= 0 else int(self.nvir / self.ndoc)
        self.closed = self.nalpha == self.nbeta
        self.ista = 0
        self.imod = 1 if self.functional == 'pnof8m' else 0
        self.nac = self.ndoc * (1 + self.ncwo)
        self.nbf5 = self.no1 + self.nac + self.nsoc
        self.no0 = self.nbf - self.nbf5
        self.noptorb = self.nbf
        self.nvar = int(self.nbf * (self.nbf - 1) / 2) - int(self.no0 * (self.no0 - 1) / 2)
        self.nv = self.ncwo * self.ndoc
        self.HighSpin = False
        self.MSpin = 0
        self.alpha = 0.02
        self.doubles = self._build_pnof5e_omega()
        self._print_os_resource_summary()

    def _print_os_resource_summary(self):
        """Print occupation-space dimensions and lightweight logical resources."""
        layouts = self._build_os_layout()
        print('OS-VQO occupation-space resources:')
        for index, layout in enumerate(layouts, start=1):
            print(
                f"  Pair {index}: orbitals={layout['orbitals']}, "
                f"d={layout['dimension']}, m={layout['m']}, "
                f"logical qubits={layout['n_qubits']}"
            )
        logical_gates = sum(layout['m'] for layout in layouts)
        scheduled_depth = max((layout['m'] for layout in layouts), default=0)
        print(f'  Number of pair subspaces: {len(layouts)}')
        print(f'  Total OS parameters / logical rotation gates: {self._count_os_parameters()} / {logical_gates}')
        print(f'  Total logical qubits: {self._count_os_qubits()}')
        print(
            '  Logical ansatz depth estimate: '
            f'{scheduled_depth} layers (assuming independent pair registers run in parallel)'
        )
        print('  Resource counts are pre-transpilation; hardware gate counts/depth may be larger.')

    def _pnof4(self, params, n_elec, norb, rdm1=None, n_opt=None, n_raw=None):
        if rdm1 is None:
            rdm1 = self._evaluate_rdm1(params)
        n = rdm1
        vecs = jnp.eye(norb)
        if self.pair_doubles:
            assert n.ndim == 1, f'Occupation numbers have wrong shape {n.shape}'
            assert vecs.ndim == 2, f'Orbital matrix has wrong shape {vecs.shape}'
        F = int(n_elec / 2)
        h = 1 - n
        S_F = jnp.sum(n[F:])
        Delta = jnp.zeros((norb, norb))
        for p in range(norb):
            for q in range(norb):
                val = 0
                if p < F and q < F:
                    val = h[q] * h[p]
                if p < F and q >= F:
                    val = (1 - S_F) / S_F * n[q] * h[p]
                if p >= F and q < F:
                    val = (1 - S_F) / S_F * h[q] * n[p]
                if p >= F and q >= F:
                    val = n[q] * n[p]
                Delta = Delta.at[q, p].set(val)
        Pi = jnp.zeros((norb, norb))
        for p in range(norb):
            for q in range(norb):
                val = 0
                if p < F and q < F:
                    val = -jnp.sqrt(jnp.abs(h[q] * h[p]))
                if p < F and q >= F:
                    val = -jnp.sqrt(jnp.abs(n[q] * h[p] / S_F * (n[p] - n[q] + n[q] * h[p] / S_F)))
                if p >= F and q < F:
                    val = -jnp.sqrt(jnp.abs(h[q] * n[p] / S_F * (n[q] - n[p] + h[q] * n[p] / S_F)))
                if p >= F and q >= F:
                    val = jnp.sqrt(jnp.abs(n[q] * n[p]))
                Pi = Pi.at[q, p].set(val)
        n_outer = jnp.outer(n, n)
        cj12 = 2.0 * (n_outer - Delta)
        ck12 = n_outer - Delta - Pi
        return (n, vecs, cj12, ck12, rdm1)

    def _ao_integrals(self, crd):
        """Compute overlap, one-electron and two-electron AO integrals with Psi4."""
        return self._ao_integrals_psi4()

    def run(self, max_outer=120):
        """Run OS-VQO and return energy, parameters, occupations, orbitals, and multipliers."""
        self.result = self._run_osvqo(max_outer=max_outer)
        self._report_transpiled_resources()
        return self.result

    def nuclear_gradient(self, result=None):
        """Evaluate the retained analytic nuclear gradient after run()."""
        result = self.result if result is None else result
        if result is None:
            raise RuntimeError("Call run() before requesting the nuclear gradient.")
        _, _, _, n, _, cj12, ck12, C, elag = result
        return self._nuclear_gradient_analytics(n, C, cj12, ck12, elag)

    @staticmethod
    def _job_time(start_time, label="Total job time", header=False):

        elapsed_time = time.perf_counter() - start_time

        day = elapsed_time // (24 * 3600)
        elapsed_time %= (24 * 3600)

        hour = elapsed_time // 3600
        elapsed_time %= 3600

        minutes = elapsed_time // 60
        elapsed_time %= 60

        seconds = elapsed_time

        formatted_time = (
            f"{day:02.0f}:"
            f"{hour:02.0f}:"
            f"{minutes:02.0f}:"
            f"{seconds:02.0f}"
        )
        if header:
            print("")
            print(f"{label}: {formatted_time}")
            print("Format: DD:HH:MM:SS")
            print(f"Finished at:    {time.ctime()}")
        else:
            print(f"{label}: {formatted_time}")
            print("Format: DD:HH:MM:SS")

        return formatted_time


    @staticmethod
    def _read_mol(inputdata):
        """
        Reads either a filepath (XYZ-like file) or a geometry string, 
        and builds a psi4.Molecule object.

        Returns:
            unit (str): "bohr"
            charge (int)
            multiplicity (int)
            symbols (list[str])
            geometry (pnp.array): shape (natoms, 3)
            xyz_str (str): full xyz string representation (for later use)
            mol (psi4.core.Molecule): Psi4 molecule object
        """
        # --- Read XYZ data ---
        if "\n" in inputdata or "units" in inputdata:
            lines = [line.strip() for line in inputdata.splitlines() if line.strip()]
        else:
            with open(inputdata, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]

        # Build xyz string (Psi4 expects full string with header)
        xyz_str = "\n".join(lines)
        xyz_str = "symmetry c1\n" + "no_reorient\n" + "no_com\n" + xyz_str
        #xyz_str = "symmetry c1\n" + xyz_str
        units = lines[0].split()[1]  # first line: "units bohr" or "units angstrom"
        charge, multiplicity = map(int, lines[1].split())

        symbols = []
        geometry = []
        for line in lines[2:]:
            parts = line.split()
            symbols.append(parts[0])
            geometry.append([float(x) for x in parts[1:4]])

        geom_jax = jnp.array(geometry)
        #geometry = pnp.array(geometry, requires_grad=False)
        geometry = geom_jax

        

        return units, charge, multiplicity, symbols, geometry, xyz_str, geom_jax


    @staticmethod
    def _mitigate_readout_distribution(
        measured_probs,
        assignment_matrix,
        ):
        """
        Recover a corrected probability distribution from

            p_meas = A p_true

        through constrained least-squares optimization:

            p_i >= 0
            sum_i p_i = 1.
        """

        p_meas = np.asarray(
            measured_probs,
            dtype=float,
        )

        A = np.asarray(
            assignment_matrix,
            dtype=float,
        )

        if A.shape != (
            len(p_meas),
            len(p_meas),
        ):
            raise ValueError(
                "Readout assignment-matrix dimension does not "
                "match the probability vector."
            )

        p0 = np.clip(
            p_meas,
            0.0,
            None,
        )

        norm = np.sum(p0)

        if norm <= 0.0:
            raise RuntimeError(
                "Measured probability distribution has zero norm."
            )

        p0 = p0 / norm

        def objective(p):

            residual = A @ p - p_meas

            return float(
                residual @ residual
            )

        constraints = [
            {
                "type": "eq",
                "fun": lambda p:
                    np.sum(p) - 1.0,
            }
        ]

        bounds = [
            (0.0, 1.0)
            for _ in range(len(p_meas))
        ]

        result = minimize(
            objective,
            p0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": 1e-12,
                "maxiter": 500,
            },
        )

        if not result.success:
            raise RuntimeError(
                "OS-VQO readout mitigation failed: "
                + result.message
            )

        p_corr = np.asarray(
            result.x,
            dtype=float,
        )

        p_corr = np.clip(
            p_corr,
            0.0,
            1.0,
        )

        p_corr /= np.sum(p_corr)

        return p_corr


    @staticmethod
    def _build_pair_subspaces_from_doubles(doubles):
        """
        Returns {g: [g, p1, p2, ...]} using self.doubles.

        Each double is [2g, 2g+1, 2p, 2p+1].
        """
        pair_map = {}

        for d in doubles:
            g = d[0] // 2
            p = d[2] // 2

            if g not in pair_map:
                pair_map[g] = [g]

            if p not in pair_map[g]:
                pair_map[g].append(p)

        return dict(sorted(pair_map.items()))


    @staticmethod
    def _func_indix(functional):
        index = re.findall(r'\d+',functional)
        return int(index[0])




    @staticmethod
    def _gray_code_bitstrings(dimension):
        """
        Generate the first `dimension` binary-reflected Gray-code states.

        Consecutive states differ in exactly one bit, which allows each
        OS-VQO logical two-level rotation to be implemented as a
        localized controlled RY operation.

        Parameters
        ----------
        dimension : int
            Number of logical occupation states in the pair subspace:
                d = m + 1

        Returns
        -------
        list[tuple[int]]
            Gray-code bitstrings of length ceil(log2(d)).
        """

        if dimension < 2:
            return [tuple()]

        n_qubits = int(np.ceil(np.log2(dimension)))

        bitstrings = []

        for i in range(dimension):

            gray = i ^ (i >> 1)

            bits = tuple(
                (gray >> (n_qubits - 1 - k)) & 1
                for k in range(n_qubits)
            )

            bitstrings.append(bits)

        return bitstrings


    @staticmethod
    def _os_two_level_rotation(theta, bit_a, bit_b, wires):
        """
        Apply a localized two-level RY rotation between two
        Gray-adjacent computational-basis states.

        The two states must differ in exactly one bit.

        Example
        -------
            |001> <-> |011>

        The differing qubit is the RY target.
        All remaining bits define the control pattern.
        """

        if len(bit_a) != len(bit_b):
            raise ValueError(
                "OS basis states must have the same number of bits."
            )

        if len(bit_a) != len(wires):
            raise ValueError(
                "OS bitstring size does not match the number of wires."
            )

        different = [
            k
            for k, (a, b) in enumerate(zip(bit_a, bit_b))
            if a != b
        ]

        if len(different) != 1:
            raise ValueError(
                "Localized OS rotation requires Gray-adjacent states, "
                f"but received {bit_a} and {bit_b}."
            )

        target_index = different[0]
        target_wire = wires[target_index]

        control_indices = [
            k
            for k in range(len(wires))
            if k != target_index
        ]

        control_wires = [
            wires[k]
            for k in control_indices
        ]

        control_values = [
            bool(bit_a[k])
            for k in control_indices
        ]

        # One-qubit case: ordinary RY
        if len(control_wires) == 0:

            qml.RY(theta, wires=target_wire)

        else:

            qml.ctrl(
                qml.RY,
                control=control_wires,
                control_values=control_values,
            )(
                theta,
                wires=target_wire,
            )


    @staticmethod
    def _os_valid_basis_indices(layout):
        """
        Return computational-basis indices corresponding to valid
        occupation-space states for one OS-VQO pair register.
        """

        indices = []

        for bits in layout["bitstrings"]:

            index = 0

            for bit in bits:
                index = 2 * index + int(bit)

            indices.append(index)

        return indices


    @staticmethod
    def _report_os_subspace_stats(stats):

        print("")
        print("==========================================")
        print("     OS-VQO LOGICAL-SUBSPACE STATS")
        print("==========================================")

        leakages = []

        for s in stats:

            leakages.append(
                s["leakage"]
            )

            print("")
            print(
                f"Pair {s['pair']}: "
                f"m={s['m']}, "
                f"d={s['dimension']}, "
                f"wires={s['wires']}"
            )

            print(
                f"  P(valid) = {s['p_valid']:.8f}"
            )

            print(
                f"  leakage  = {s['leakage']:.8f}"
            )

            print(
                f"  unused   = {s['unused_indices']}"
            )

        if len(leakages) > 0:

            print("")
            print(
                "Mean leakage    =",
                np.mean(leakages),
            )

            print(
                "Maximum leakage =",
                np.max(leakages),
            )


    def _orthonormalize(self, C, S):

        # overlap in MO basis
        M = C.T @ S @ C

        # eigen decomposition
        eigval, eigvec = eigh(M)

        # inverse sqrt
        M_inv_sqrt = eigvec @ jnp.diag(eigval**(-0.5)) @ eigvec.T

        # orthonormalized orbitals
        C_new = C @ M_inv_sqrt

        return C_new


    def _check_ortho(self, C, S):

        CTSC = C.T @ S @ C
        ortho_deviation = jnp.abs(CTSC - jnp.eye(self.nbf))

        violation_mask = ortho_deviation > 1e-6

        if jnp.any(violation_mask):

            print(
                "Orthonormality violations {:d}, Maximum Violation {:f}".format(
                    violation_mask.sum(), ortho_deviation.max()
                )
            )
            print("Trying to orthonormalize")

            C = self._orthonormalize(C, S)
            C = self._check_ortho(C, S)          
            # # re-check
            # CTSC = C.T @ S @ C
            # ortho_deviation = jnp.abs(CTSC - jnp.eye(self.nbf))

        else:
            print("No violations of the orthonormality")

        # ----------------------------
        # Sign convention correction
        # ----------------------------

        # index of largest |C| in each MO column
        idxmax = jnp.argmax(jnp.abs(C), axis=0)

        # values at those indices
        col_indices = jnp.arange(self.nbf)
        max_vals = C[idxmax, col_indices]

        # sign of largest coefficient
        signs = jnp.sign(max_vals)

        # broadcast sign correction
        C = C * signs

        return C


    def _ao_integrals_psi4(self):
        """
        Compute S, T, V, H and ERIs (I) in atomic orbitals from Psi4
        """
        print("\nGeometry sent to Psi4:")
        print(self.xyz_str)
        print()
        # Build molecule
        self.mol = mol = psi4.geometry(self.xyz_str)

        psi4.set_options({
            "basis": self.basis,
        })

        # Build an empty wavefunction object containing molecule/basis metadata
        self.wfn = wfn = psi4.core.Wavefunction.build(
            mol,
            psi4.core.get_global_option("basis"),
        )

        # AO integral engine
        mints = psi4.core.MintsHelper(wfn.basisset())

        # One-electron integrals
        S = jnp.asarray(mints.ao_overlap())
        T = jnp.asarray(mints.ao_kinetic())
        V = jnp.asarray(mints.ao_potential())
        H_core = T + V

        # Two-electron integrals
        ERI = jnp.asarray(mints.ao_eri())

        # Nuclear repulsion
        E_nuc = mol.nuclear_repulsion_energy()

        return S, H_core, ERI, E_nuc, wfn


    def _pnof(self, params, n_elec, norb, rdm1, n_opt=None, n_raw=None):
        if self.ipnof == 4:
            n, vecs, cj12, ck12, rdm1 = self._pnof4(params, n_elec, norb, rdm1, n_opt, n_raw)
        elif self.ipnof == 5:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof5(params, n_elec, norb, rdm1, n_opt, n_raw)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof5")
        elif self.ipnof == 7:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof7(params, n_elec, norb, rdm1, n_opt, n_raw)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof8") 
        elif self.ipnof == 8:
            if self.pair_doubles:
                n, vecs, cj12, ck12, rdm1 = self._pnof8(params, n_elec, norb, rdm1, n_opt, n_raw)
            else:
                raise ValueError(f"Pair_doubles must be True for pnof8")  
        else:
            raise ValueError(f"Unsupported PNOF level: {self.ipnof}")
        return n, vecs, cj12, ck12, rdm1


    def ene_nof(self, params, rdm1=None):
        
        E_nuc, J_MO, K_MO, H_core, n_elec, norb = self.E_nuc, self.J_MO, self.K_MO, self.H_core, self.n_elec, self.norb
        
        n, vecs, cj12, ck12, rdm1 = self._pnof(params, n_elec, norb, rdm1)
        
        E = self._calce(n,cj12,ck12,J_MO,K_MO,H_core)

        return E_nuc + E, rdm1, n, vecs, cj12, ck12


    def _os_parameter_bounds(self):
        """
        Bounds for the hierarchical OS-VQO parameters.

        For every pair subspace:

            theta_g in [0, pi/2]
            theta_p in [0, pi]
        """
        
        pair_map = self._build_pair_subspaces_from_doubles(self.doubles)

        bounds = []
        
        angle_eps = 1e-6

        for g, omega in pair_map.items():
            weak_orbs = [p for p in omega if p != g]

            if len(weak_orbs) == 0:
                continue

            # Root angle:
            # n_g = cos^2(theta_g / 2)
            bounds.append((angle_eps, 0.5 * np.pi))

            # Residual-distribution angles
            for _ in weak_orbs[:-1]:
                bounds.append((angle_eps, np.pi - angle_eps))
        return bounds


    def _build_os_layout(self):
        """
        Build the logical-qudit layout for all OS-VQO pair subspaces.

        For each pair Omega_g = {g, p1, ..., pm}, store:

            - orbital indices
            - number of logical qubits
            - assigned PennyLane wires
            - Gray-code basis states
            - slice of the global OS parameter vector

        Returns
        -------
        list[dict]
            One layout dictionary per electron-pair subspace.
        """

        pair_map = self._build_pair_subspaces_from_doubles(self.doubles)

        layouts = []

        wire_offset = 0
        parameter_offset = 0

        for g, omega in pair_map.items():

            orbitals = list(omega)
            dimension = len(orbitals)

            if dimension <= 1:
                continue

            m = dimension - 1

            n_qubits = int(np.ceil(np.log2(dimension)))

            wires = list(
                range(
                    wire_offset,
                    wire_offset + n_qubits,
                )
            )
            
            # ---------------------------------------------------------
            # Computational encoding
            #
            # m = 2 follows the explicit hierarchical encoding used
            # in Appendix B of the OS-VQO manuscript:
            #
            #     g  -> |00>
            #     p1 -> |11>
            #     p2 -> |10>
            #
            # For the remaining dimensions, use the Gray-code
            # construction.
            # ---------------------------------------------------------

            if m == 2:
                bitstrings = [
                    (0, 0),
                    (1, 1),
                    (1, 0),
                ]
            else:
                bitstrings = self._gray_code_bitstrings(
                    dimension
                )

            parameter_slice = slice(
                parameter_offset,
                parameter_offset + m,
            )

            layouts.append(
                {
                    "g": g,
                    "orbitals": orbitals,
                    "dimension": dimension,
                    "m": m,
                    "n_qubits": n_qubits,
                    "wires": wires,
                    "bitstrings": bitstrings,
                    "parameter_slice": parameter_slice,
                }
            )

            wire_offset += n_qubits
            parameter_offset += m

        if parameter_offset != self._count_os_parameters():
            raise RuntimeError(
                "OS layout parameter count is inconsistent: "
                f"layout uses {parameter_offset}, "
                f"expected {self._count_os_parameters()}."
            )

        if wire_offset != self._count_os_qubits():
            raise RuntimeError(
                "OS layout qubit count is inconsistent: "
                f"layout uses {wire_offset}, "
                f"expected {self._count_os_qubits()}."
            )

        return layouts


    def _os_postselect_probabilities(
        self,
        probabilities,
        tol=1e-12,
        ):
        """
        Postselect each OS-VQO register onto its valid logical
        occupation-space subspace.

        Invalid computational states are assigned zero probability.
        Valid probabilities are renormalized per pair.

        For d = 2^q, such as m = 1 or m = 3, all computational
        states are valid and this operation changes nothing.

        Returns
        -------
        corrected_probabilities : tuple
        stats : list[dict]
        """
        print(">>> OS POSTSELECT PORB. ACTIVATED")
        corrected_probabilities = []
        stats = []

        for pair_index, (probs, layout) in enumerate(
            zip(probabilities, self._os_layout)
        ):

            probs = jnp.asarray(probs)

            valid_indices = self._os_valid_basis_indices(
                layout
            )

            all_indices = set(
                range(len(probs))
            )

            unused_indices = sorted(
                all_indices - set(valid_indices)
            )

            p_valid = jnp.sum(
                probs[jnp.array(valid_indices)]
            )

            leakage = 1.0 - p_valid

            probs_corr = jnp.zeros_like(probs)

            if float(p_valid) > tol:

                for idx in valid_indices:

                    probs_corr = probs_corr.at[idx].set(
                        probs[idx] / p_valid
                    )

            else:
                raise RuntimeError(
                    f"OS pair {pair_index} has effectively "
                    "zero valid probability."
                )

            corrected_probabilities.append(
                probs_corr
            )

            stats.append(
                {
                    "pair": pair_index,
                    "m": layout["m"],
                    "dimension": layout["dimension"],
                    "wires": layout["wires"],
                    "valid_indices": valid_indices,
                    "unused_indices": unused_indices,
                    "p_valid": float(p_valid),
                    "leakage": float(leakage),
                }
            )

        return tuple(corrected_probabilities), stats


    def _build_os_readout_assignment_matrices(self):
        """
        Build one pair-local readout assignment matrix for every
        OS-VQO pair register.

        For one register,

            A[i,j] = P(measure i | prepare j)

        where i and j run over the computational basis states.

        The current implementation supports OS registers containing
        at most two logical qubits, corresponding to m = 1, 2, 3.
        """

        if not hasattr(self, "_os_layout"):
            self._os_layout = self._build_os_layout()

        layouts = self._os_layout
        dev = self.dev_instance

        max_q = max(
            layout["n_qubits"]
            for layout in layouts
        )

        if max_q > 2:
            raise NotImplementedError(
                "OS-VQO readout mitigation is currently restricted "
                "to one- and two-qubit pair registers."
            )

        matrices = []

        for layout in layouts:

            dim_comp = 2 ** layout["n_qubits"]

            matrices.append(
                np.zeros(
                    (dim_comp, dim_comp),
                    dtype=float,
                )
            )

        unique_q = sorted(
            set(
                layout["n_qubits"]
                for layout in layouts
            )
        )

        for q in unique_q:

            target_indices = [
                i
                for i, layout in enumerate(layouts)
                if layout["n_qubits"] == q
            ]

            target_layouts = [
                layouts[i]
                for i in target_indices
            ]

            n_states = 2 ** q

            @qml.qnode(dev)
            def calibration_qnode(state_index):

                bits = np.array(
                    [
                        (state_index >> (q - 1 - k)) & 1
                        for k in range(q)
                    ],
                    dtype=int,
                )

                # Prepare the same known computational state
                # in every register of the same size.
                for layout in target_layouts:

                    qml.BasisState(
                        bits,
                        wires=layout["wires"],
                    )

                return tuple(
                    qml.probs(
                        wires=layout["wires"]
                    )
                    for layout in target_layouts
                )

            for prepared_state in range(n_states):

                measured_probabilities = (
                    calibration_qnode(
                        prepared_state
                    )
                )

                for local_index, probs in enumerate(
                    measured_probabilities
                ):

                    global_pair_index = (
                        target_indices[local_index]
                    )

                    matrices[
                        global_pair_index
                    ][:, prepared_state] = np.asarray(
                        probs,
                        dtype=float,
                    )

        self._os_readout_assignment_matrices = matrices

        return matrices


    def _mitigate_os_readout(
        self,
        probabilities,
        matrices=None,
        ):
        """
        Apply pair-local readout mitigation to every OS-VQO
        pair-register probability distribution.
        """

        if matrices is None:

            matrices = getattr(
                self,
                "_os_readout_assignment_matrices",
                None,
            )

        if matrices is None:

            print("")
            print("Building OS-VQO readout calibration matrices...")

            matrices = (
                self._build_os_readout_assignment_matrices()
            )

        if len(probabilities) != len(matrices):
            raise ValueError(
                "Number of OS probability vectors does not match "
                "the number of readout assignment matrices."
            )

        corrected = []

        for probs, A in zip(
            probabilities,
            matrices,
        ):

            p_corr = (
                self._mitigate_readout_distribution(
                    probs,
                    A,
                )
            )

            corrected.append(
                jnp.asarray(p_corr)
            )

        return tuple(corrected)


    def _prepare_os_pair_generic(self, pair_params, layout):
        """
        Generic Gray-code/localized-Givens OS-VQO preparation.
        This construction is exact for arbitrary m but can become
        expensive after decomposition because every localized rotation
        is conditioned on the remaining register qubits.

        The convention matches _os_occupations_from_theta():

            n_g = cos^2(theta_g / 2)

            n_p1 = residual * sin^2(theta_p1 / 2)
            residual <- residual * cos^2(theta_p1 / 2)

            ...

        Therefore:

            - the root rotation uses theta_g
            - each residual-distribution rotation uses pi - theta_p
        """

        pair_params = jnp.asarray(pair_params)

        bitstrings = layout["bitstrings"]
        wires = layout["wires"]
        m = layout["m"]

        if len(pair_params) != m:
            raise ValueError(
                f"Pair requires {m} OS parameters, "
                f"but received {len(pair_params)}."
            )

        # -------------------------------------------------
        # Root:
        #
        # |b0> -> cos(theta_g/2)|b0>
        #       + sin(theta_g/2)|b1>
        #
        # b0 corresponds to g.
        # -------------------------------------------------

        theta_g = pair_params[0]

        self._os_two_level_rotation(
            theta_g,
            bitstrings[0],
            bitstrings[1],
            wires,
        )

        # -------------------------------------------------
        # Residual hierarchy
        #
        # To reproduce the analytical convention
        #
        #   n_pk = residual * sin^2(theta_pk/2)
        #   residual *= cos^2(theta_pk/2)
        #
        # use phi = pi - theta_pk.
        #
        # -------------------------------------------------

        for k in range(1, m):

            theta_p = pair_params[k]

            effective_angle = jnp.pi - theta_p

            self._os_two_level_rotation(
                effective_angle,
                bitstrings[k],
                bitstrings[k + 1],
                wires,
            )


    def _prepare_os_pair_m1(self, pair_params, layout):
        """
        Optimized OS-VQO circuit for

            m = 1
            d = 2
            |0> -> g
            |1> -> p1

        Requires one qubit and one RY rotation.
        """

        if layout["m"] != 1:
            raise ValueError("_prepare_os_pair_m1 requires m = 1.")

        if len(pair_params) != 1:
            raise ValueError("m = 1 requires exactly one parameter.")

        theta_g = pair_params[0]
        wire = layout["wires"][0]

        qml.RY(
            theta_g,
            wires=wire,
        )


    def _prepare_os_pair_m2(self, pair_params, layout):
        """
        Optimized OS-VQO circuit for

            m = 2
            d = 3

        Encoding:

            g  -> |00>
            p1 -> |11>
            p2 -> |10>

        |01> is unused.

        Analytical convention:

            n_g  = cos^2(theta_g / 2)

            n_p1 = sin^2(theta_g / 2)
                sin^2(theta_p1 / 2)

            n_p2 = sin^2(theta_g / 2)
                cos^2(theta_p1 / 2)
        """

        if layout["m"] != 2:
            raise ValueError(
                "_prepare_os_pair_m2 requires m = 2."
            )

        if len(pair_params) != 2:
            raise ValueError(
                "m = 2 requires exactly two parameters."
            )

        expected_bits = [
            (0, 0),   # g
            (1, 1),   # p1
            (1, 0),   # p2
        ]

        if layout["bitstrings"] != expected_bits:
            raise RuntimeError(
                "Unexpected encoding convention for m = 2."
            )

        q0, q1 = layout["wires"]

        theta_g = pair_params[0]
        theta_p1 = pair_params[1]

        # --------------------------------------------------------
        # Root branching
        #
        # |00> -> cos(theta_g/2)|00>
        #       + sin(theta_g/2)|10>
        # --------------------------------------------------------

        qml.RY(
            theta_g,
            wires=q0,
        )

        # --------------------------------------------------------
        # Residual branching
        #
        # |10> -> cos(theta_p1/2)|10>
        #       + sin(theta_p1/2)|11>
        # --------------------------------------------------------

        qml.CRY(
            theta_p1,
            wires=[q0, q1],
        )


    def _prepare_os_pair_m3(self, pair_params, layout):
        """
        Optimized OS-VQO circuit for

            m = 3
            d = 4

        Gray-code encoding:

            g  -> |00>
            p1 -> |01>
            p2 -> |11>
            p3 -> |10>

        Three parameters produce four occupations.
        """

        if layout["m"] != 3:
            raise ValueError("_prepare_os_pair_m3 requires m = 3.")

        if len(pair_params) != 3:
            raise ValueError("m = 3 requires exactly three parameters.")

        expected_bits = [
            (0, 0),
            (0, 1),
            (1, 1),
            (1, 0),
        ]

        if layout["bitstrings"] != expected_bits:
            raise RuntimeError(
                "Unexpected Gray-code convention for m = 3."
            )

        q0, q1 = layout["wires"]

        theta_g = pair_params[0]
        theta_p1 = pair_params[1]
        theta_p2 = pair_params[2]

        # First branching:
        # |00> -> |00> + |01>
        qml.RY(
            theta_g,
            wires=q1,
        )

        # Second branching:
        # |01> <-> |11>
        qml.CRY(
            jnp.pi - theta_p1,
            wires=[q1, q0],
        )

        # Third branching:
        # |11> <-> |10>
        qml.CRY(
            jnp.pi - theta_p2,
            wires=[q0, q1],
        )


    def _prepare_os_pair(self, pair_params, layout):
        """
        Prepare one OS-VQO pair register.

        Optimized circuits are used for m = 1, 2, 3.
        The exact generic Gray-code construction is retained
        as a fallback for larger occupation spaces.
        """

        m = layout["m"]

        if getattr(
            self,
            "os_use_specialized_small_m",
            True,
        ):

            if m == 1:
                print(">>> USING SPECIALIZED m=1 KERNEL")
                return self._prepare_os_pair_m1(
                    pair_params,
                    layout,
                )

            if m == 2:
                print(">>> USING SPECIALIZED m=2 KERNEL")
                return self._prepare_os_pair_m2(
                    pair_params,
                    layout,
                )

            if m == 3:
                print(">>> USING SPECIALIZED m=3 KERNEL")
                return self._prepare_os_pair_m3(
                    pair_params,
                    layout,
                )

        # Generic exact fallback
        print(f">>> USING GENERIC OS KERNEL, m={m}")
        
        return self._prepare_os_pair_generic(
            pair_params,
            layout,
        )


    def _decode_os_probabilities(self, probabilities):
        """
        Convert OS-VQO pair-register probabilities into the full
        natural-occupation vector.

        Parameters
        ----------
        probabilities : sequence
            One probability vector per OS pair register.

        Returns
        -------
        jax.Array
            Full occupation-number vector of length self.norb.
        """

        layouts = self._os_layout

        if len(probabilities) != len(layouts):
            raise ValueError(
                "Number of OS probability vectors does not match "
                "the number of pair layouts."
            )

        n_full = jnp.zeros(self.norb)

        # Frozen/core orbitals
        if self.no1 > 0:
            core_orbs = jnp.arange(self.no1)
            n_full = n_full.at[core_orbs].set(1.0)

        for probs, layout in zip(probabilities, layouts):

            probs = jnp.asarray(probs)

            orbitals = layout["orbitals"]
            bitstrings = layout["bitstrings"]
            n_qubits = layout["n_qubits"]

            expected_size = 2 ** n_qubits

            if probs.shape[0] != expected_size:
                raise ValueError(
                    f"Expected {expected_size} probabilities "
                    f"for OS register {layout['wires']}, "
                    f"received {probs.shape[0]}."
                )

            for orbital, bits in zip(orbitals, bitstrings):

                basis_index = 0

                for bit in bits:
                    basis_index = 2 * basis_index + int(bit)

                n_full = n_full.at[orbital].set(
                    probs[basis_index]
                )

        return n_full


    def _build_pnof5e_omega(self):
        Omega_mo = self._build_pnof5e_omega_mo()
        
        Omega_spin = []
        for omega in Omega_mo:
            g = omega[0]          # strong orbital
            weak_orbs = omega[1:] # weak orbitals in Omega_g

            g_spin = [2*g, 2*g + 1]

            for q in weak_orbs:
                q_spin = [2*q, 2*q + 1]
                Omega_spin.append(g_spin + q_spin)
        return Omega_spin


    def _build_pnof5e_omega_mo(self):
        """
        Build Omega_g exactly as in PyNOF5 using ll / ul logic.
        Returns Omega in MO indices.
        """
        Omega = []
        
        for l in range(self.ndoc):
            g = self.no1 + l

            ll = self.no1 + self.ndns + (self.ndoc - l - 1) * self.ncwo
            ul = ll + self.ncwo

            if ul > ll:
                weak = list(range(ll, ul))
                Omega.append([g] + weak)
            elif ul == ll:
                Omega.append([ul-1]+[ul])
                break
            
        return Omega


    def _JKH_MO_Full(self, C,H,I):
        #denmatj
        D = jnp.einsum('mi,ni->imn', C[:,0:self.nbf5], C[:,0:self.nbf5],optimize=True)
        #QJMATm
        J = jnp.tensordot(D, I, axes=([1,2],[2,3]))
        J_MO = jnp.tensordot(J, D,axes=((1,2),(1,2)))
        #QKMATm
        K = jnp.tensordot(D, I, axes=([1,2],[1,3]))
        K_MO = jnp.tensordot(K, D, axes=([1,2],[1,2]))
        #QHMATm
        H_core = jnp.tensordot(D, H, axes=([1,2],[0,1]))
        
        return J_MO,K_MO,H_core


    def _calcorbg(self, y,n,cj12,ck12,C,H,I):

        Cnew = self._rotate_orbital(y,C)

        elag,_ = self._computeLagrange2(n,cj12,ck12,Cnew,H,I)

        grad = 4*(elag - elag.T)

        grads = np.zeros((self.nvar))

        k = 0
        for i in range(self.nbf5):
            for j in range(i+1,self.nbf):
                grads[k] = grad[i,j]
                k += 1
        grad = grads

        return grad


    def _computeLagrange2(self,n,cj12,ck12,C,H,I):
        
        H_mat = jnp.einsum("mi,mn,nj->ij",C,H,C[:,:self.nbf5],optimize=True)
        I_MO = jnp.einsum("mp,nq,mnsl,sr,lt->pqrt",C,C[:,:self.nbf5],I,C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)

        cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
        ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
        
        elag = np.zeros((self.nbf,self.nbf))
        if(self.MSpin==0):
            # 2ndH/dy_ab
            
            #elag[:,:self.nbf5] +=  jnp.einsum('b,ab->ab',n,H_mat[:,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] +=  jnp.einsum('b,ab->ab',n[:self.nbf5],H_mat[:,:self.nbf5],optimize=True)

            # dJ_pp/dy_ab
            elag[:,:self.nbeta] +=  jnp.einsum('b,abbb->ab',n[:self.nbeta],I_MO[:,:self.nbeta,:self.nbeta,:self.nbeta],optimize=True)
            elag[:,self.nalpha:self.nbf5] +=  jnp.einsum('b,abbb->ab',n[self.nalpha:self.nbf5],I_MO[:,self.nalpha:self.nbf5,self.nalpha:self.nbf5,self.nalpha:self.nbf5],optimize=True)
            
            # C^J_pq dJ_pq/dy_ab 
            #elag[:,:self.nbf5] +=  jnp.einsum('bq,abqq->ab',cj12,I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] +=  jnp.einsum('bq,abqq->ab',cj12[:self.nbf5,:self.nbf5],I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)

            # -C^K_pq dK_pq/dy_ab 
            #elag[:,:self.nbf5] += -jnp.einsum('bq,aqbq->ab',ck12,I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
            elag[:,:self.nbf5] += -jnp.einsum('bq,aqbq->ab',ck12[:self.nbf5,:self.nbf5],I_MO[:,:self.nbf5,:self.nbf5,:self.nbf5],optimize=True)
        else:
            raise RuntimeError("MSpin != 0 is not implemented yet")
        return elag, H_mat


    def _rotate_orbital(self, y,C):
        
        ynew = np.zeros((self.nbf,self.nbf))

        n = 0
        for i in range(self.nbf5):
            for j in range(i+1,self.nbf):
                ynew[i,j] =  y[n]
                ynew[j,i] = -y[n]
                n += 1

        U = expm(ynew)
        Cnew = jnp.einsum("mr,rp->mp",C,U,optimize=True)

        return Cnew


    def _calcorbe(self, y,n,cj12,ck12,C,H,I):

        Cnew = self._rotate_orbital(y,C)
        J_MO,K_MO,H_core = self._JKH_MO_Full(Cnew,H,I)
        E = self._calce(n,cj12,ck12,J_MO,K_MO,H_core)

        return E


    def _orbopt_adam(self, n,cj12,ck12,C,H,I):
        y = np.zeros((self.nvar))
        E = self._calcorbe(y, n,cj12,ck12,C,H,I)
        
        alpha = self.alpha
        beta1 = 0.7
        beta2 = 0.999

        y = np.zeros((self.nvar))

        m = 0.0 * y
        v = 0.0 * y
        vhat_max = 0.0 * y

        improved = False
        success = False
        best_E, best_C = E, C
        nit = 0
        for i in range(self.maxloop):
            nit += 1
            
            grad = self._calcorbg(y*0, n,cj12,ck12,C,H,I)
            if np.linalg.norm(grad) < 10**-4 and improved:
                success = True
                break

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad**2)
            mhat = m / (1.0 - beta1**(i+1))
            vhat = v / (1.0 - beta2**(i+1))
            vhat_max = np.maximum(vhat_max, vhat)
            y = - alpha * mhat / (jnp.sqrt(vhat_max + 10**-8)) #AMSgrad
            C = self._rotate_orbital(y,C)
            
            E = self._calcorbe(y*0, n,cj12,ck12,C,H,I)
            if self.verbose > 1:
                print(f"Step = {i}, "
                        f"Etot = {float(self.E_nuc+E):.8f} Ha, "
                        f"E < best_E : {E < best_E}")
            if E < best_E:
                best_C = C
                best_E = E
                improved = True

        if not improved:
            self.alpha = self.alpha/10
            #print("      alpha ",self.alpha)
        
        return best_E,best_C,nit,success


    def _orbital_optimization(self, n, cj12,ck12, C,H,I):
        """
        Perform classical orbital optimization using PyNOF.
        H_MO and I_MO
        """
        
        E_orb, C_new, nit, success = self._orbopt_adam(
            n, cj12,ck12, C,H,I
        )
        
        # Evaluate orbital stationarity at the optimized orbitals
        _, _, sumdiff, maxdiff = self._ENERGY1r(
            C_new, n, H, I, cj12, ck12
        )
        
        # if not success:
        #     print("Warning: orbital optimization did not fully converge")
        if self.verbose>1:
            print(
                f"Orbital Lagrange convergence: "
                f"sumdiff = {sumdiff:.6e}, "
                f"maxdiff = {maxdiff:.6e}"
            )
        
        # Save updated orbitals
        self.C_MO = C_new

        return E_orb, C_new, nit, sumdiff, maxdiff


    def _computeF_RC_CPU(self, J,K,n,H,cj12,ck12):
        
        # PyNOF/DoNOF Fock construction operates in the NBF5
        # NOF orbital space. OS-VQO may provide full-NBF arrays.
        n = n[:self.nbf5]
        cj12 = cj12[:self.nbf5, :self.nbf5]
        ck12 = ck12[:self.nbf5, :self.nbf5]

        J = J[:self.nbf5, :, :]
        K = K[:self.nbf5, :, :]

        F = jnp.zeros(
            (self.nbf5, self.nbf, self.nbf),
            dtype=H.dtype
        )

        ini = 0
        if(self.no1>0):
            ini = self.no1

        # nH
        F += jnp.einsum(
            'i,mn->imn',
            n,
            H,
            optimize=True
        )        # i = [1,nbf5]

        # # nJ
        
        # F[ini:self.nbeta,:,:] += jnp.einsum('i,imn->imn',n[ini:self.nbeta],J[ini:self.nbeta,:,:],optimize=True)        # i = [ini,nbeta]
        # F[self.nalpha:self.nbf5,:,:] += jnp.einsum('i,imn->imn',n[self.nalpha:self.nbf5],J[self.nalpha:self.nbf5,:,:],optimize=True)  # i = [nalpha,nbf5]

        # nJ
        F = F.at[ini:self.nbeta,:,:].add(
            jnp.einsum('i,imn->imn', n[ini:self.nbeta], J[ini:self.nbeta,:,:], optimize=True)
        )

        F = F.at[self.nalpha:self.nbf5,:,:].add(
            jnp.einsum('i,imn->imn', n[self.nalpha:self.nbf5], J[self.nalpha:self.nbf5,:,:], optimize=True)
        )
        
        # ---- Remove diagonal of cj12 block ----
        sub = cj12[ini:, ini:]
        sub = sub - jnp.diag(jnp.diag(sub))
        cj12 = cj12.at[ini:, ini:].set(sub)
        # C^J J
        F += jnp.einsum('ij,jmn->imn',cj12,J,optimize=True)                                                # i = [1,nbf5]
        #F[ini:self.nbf5,:,:] -= jnp.einsum('ii,imn->imn',cj12[ini:self.nbf5,ini:self.nbf5],J[ini:self.nbf5,:,:],optimize=True) # quita i==j

        
        # ---- Remove diagonal of ck12 block ----
        sub = ck12[ini:, ini:]
        sub = sub - jnp.diag(jnp.diag(sub))
        ck12 = ck12.at[ini:, ini:].set(sub)
        # -C^K K
        F -= jnp.einsum('ij,jmn->imn',ck12,K,optimize=True)                                                # i = [1,nbf5]
        #F[ini:self.nbf5,:,:] += jnp.einsum('ii,imn->imn',ck12[ini:self.nbf5,ini:self.nbf5],K[ini:self.nbf5,:,:],optimize=True) # quita i==j

        return F


    def _JKj_Full(self, C,I):
        #denmatj
        D = jnp.einsum('mi,ni->imn', C[:,0:self.nbf5], C[:,0:self.nbf5], optimize=True)
        #hstarj
        J = jnp.tensordot(D, I, axes=([1,2],[2,3]))
        #hstark
        K = jnp.tensordot(D, I, axes=([1,2],[1,3]))
        
        return J,K


    def _computeLagrange(self, F,C):

        G = jnp.einsum('imn,ni->mi',F,C[:,0:self.nbf5],optimize=True)

        #Compute Lagrange multipliers
        elag = jnp.zeros((self.nbf,self.nbf))
        elag = elag.at[0:self.noptorb,0:self.nbf5].add(
            jnp.einsum('mi,mj->ij',C[:,0:self.noptorb],G,optimize=True)[0:self.noptorb,0:self.nbf5]
        )
        
        # elag[0:self.noptorb,0:self.nbf5] = jnp.einsum('mi,mj->ij',C[:,0:self.noptorb],G,optimize=True)[0:self.noptorb,0:self.nbf5]

        return elag


    def _computeE_elec(self, H,C,n,elag):
        #EELECTRr
        E = 0

        E = E + jnp.einsum('ii',elag[:self.nbf5,:self.nbf5],optimize=True)
        E = E + jnp.einsum('i,mi,mn,ni',n[:self.nbeta],C[:,:self.nbeta],H,C[:,:self.nbeta],optimize=True)
        if(not self.HighSpin):
            E = E + jnp.einsum('i,mi,mn,ni',n[self.nbeta:self.nalpha],C[:,self.nbeta:self.nalpha],H,C[:,self.nbeta:self.nalpha],optimize=True)
        elif(self.HighSpin):
            E = E + 0.5*jnp.einsum('mi,mn,ni',C[:,self.nbeta:self.nalpha],H,C[:,self.nbeta:self.nalpha],optimize=True)

        E = E + jnp.einsum('i,mi,mn,ni',n[self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],H,C[:,self.nalpha:self.nbf5],optimize=True)

        return E


    def _computeLagrangeConvergency(self, elag):
        # Convergency

        sumdiff = np.sum(np.abs(elag-elag.T))
        maxdiff = np.max(np.abs(elag-elag.T))

        return sumdiff,maxdiff


    def _ENERGY1r(self, C,n,H,I,cj12,ck12):
        if(self.no1==0):
            elag,Hmat = self._computeLagrange2(n,cj12,ck12,C,H,I)
            E = jnp.einsum('ii',elag[:self.nbf5,:self.nbf5],optimize=True)
            E = E + jnp.einsum('i,ii',n[:self.nbf5],Hmat[:self.nbf5,:self.nbf5],optimize=True)
        else:
            
            J,K = self._JKj_Full(C,I)
            if(self.MSpin==0):
                F = self._computeF_RC_CPU(J,K,n,H,cj12,ck12)
            elif(not self.MSpin==0):
                raise RuntimeError("MSpin != 0 is not implemented yet")
            elag = self._computeLagrange(F,C)
            E = self._computeE_elec(H,C,n,elag)

        sumdiff,maxdiff = self._computeLagrangeConvergency(elag)
        
        return E,elag,sumdiff,maxdiff


    def _reset_qnode_cache(self):
        for attr in [
            "_rdm1_qnode_diag",
            "_rdm1_qnode_full",
            "_rdm1_qnode_samples",
            "_os_qnode",

            # OS calibration QNodes
            "_os_response_qnodes",

            # Calibration matrices are backend/layout dependent
            "_os_response_matrices",
            "_os_response_matrix_params",
            "_os_readout_assignment_matrices",

            # Residual-floor calibration is also hardware dependent
            "_os_residual_floor",
            "_last_os_residual_floor",
            "_last_os_residual_floor_raw",

            "dev_instance",
        ]:

            if hasattr(self, attr):
                delattr(self, attr)


    def _prepare_backend(self):
        self._reset_qnode_cache()
        self.dev_instance = self._device()
        self._build_qnodes()


    def _os_occupations_from_theta(self, theta):
        """
        Generate OS-VQO occupations analytically.

        For each pair subspace:

            Omega_g = {g, p1, ..., pm}

        the occupations are generated as:

            n_g = cos^2(theta_g / 2)

        The remaining occupation is distributed recursively:

            n_p1 = residual * sin^2(theta_p1 / 2)
            residual <- residual * cos^2(theta_p1 / 2)

            n_p2 = residual * sin^2(theta_p2 / 2)
            residual <- residual * cos^2(theta_p2 / 2)

            ...

        The last weak orbital receives the remaining residual.
        """

        pair_map = self._build_pair_subspaces_from_doubles(self.doubles)

        theta = jnp.asarray(theta)
        n_full = jnp.zeros(self.norb)

        # Core orbitals
        if self.no1 > 0:
            core_orbs = jnp.arange(self.no1)
            n_full = n_full.at[core_orbs].set(1.0)

        parameter_index = 0
        
        for g, omega in pair_map.items():

            weak_orbs = [p for p in omega if p != g]

            if len(weak_orbs) == 0:
                n_full = n_full.at[g].set(1.0)
                continue

            # Root parameter
            theta_g = theta[parameter_index]
            parameter_index += 1

            n_g = jnp.cos(theta_g / 2.0) ** 2
            residual = jnp.sin(theta_g / 2.0) ** 2

            n_full = n_full.at[g].set(n_g)

            # Distribute residual among all weak orbitals
            # except the final one.
            for p in weak_orbs[:-1]:

                theta_p = theta[parameter_index]
                parameter_index += 1

                n_p = residual * jnp.sin(theta_p / 2.0) ** 2
                residual = residual * jnp.cos(theta_p / 2.0) ** 2

                n_full = n_full.at[p].set(n_p)

            # Last weak orbital gets the remaining occupation
            p_last = weak_orbs[-1]
            n_full = n_full.at[p_last].set(residual)

        if parameter_index != len(theta):
            raise ValueError(
                "OS parameter-count mismatch: "
                f"used {parameter_index} parameters but received {len(theta)}."
            )

        return n_full


    def _count_os_parameters(self):
        pair_map = self._build_pair_subspaces_from_doubles(self.doubles)

        n_params = 0

        for g, omega in pair_map.items():
            weak_orbs = [p for p in omega if p != g]
            n_params += len(weak_orbs)

        return n_params


    def _count_os_qubits(self):
        """
        Count the total number of logical qubits required by OS-VQO.

        A pair subspace with d = m + 1 occupation states requires:

            ceil(log2(d))

        qubits.
        """

        pair_map = self._build_pair_subspaces_from_doubles(self.doubles)

        n_qubits = 0

        for g, omega in pair_map.items():
            dimension = len(omega)

            if dimension <= 1:
                continue

            n_qubits += int(np.ceil(np.log2(dimension)))

        return n_qubits


    def _is_hybrid_mode(self):
        return self.dev.startswith("hybrid_")


    def _hybrid_target_dev(self):
        if not self._is_hybrid_mode():
            raise RuntimeError(f"Not a hybrid mode: {self.dev}")

        target = self.dev.replace("hybrid_", "", 1)

        valid_targets = ["real", "noise_simulator"]
        if target not in valid_targets:
            raise RuntimeError(f"Unsupported hybrid target: {target}")

        return target


    def _calce(self, n,cj12,ck12,J_MO,K_MO,H_core):
        E = 0
        n_test = n
        if(self.MSpin==0):
            # 2H + J
            #E = E + 2*jnp.einsum('i,i',n,H_core,optimize=True)
            E = E + 2*jnp.einsum('i,i',n[:self.nbf5],H_core,optimize=True)
            
            E = E + jnp.einsum('i,i',n[:self.nbeta],jnp.diagonal(J_MO)[:self.nbeta],optimize=True)
            E = E + jnp.einsum('i,i',n[self.nalpha:self.nbf5],jnp.diagonal(J_MO)[self.nalpha:self.nbf5],optimize=True)
            #C^J JMO
            cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
            E = E + jnp.einsum('ij,ji->',cj12[:self.nbf5,:self.nbf5],J_MO,optimize=True) # sum_ij
            #C^K KMO
            ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
            E = E - jnp.einsum('ij,ji->',ck12[:self.nbf5,:self.nbf5],K_MO,optimize=True) # sum_ij
        else:
            print(f"Mspin:{self.MSpin} must be cero")
        return E


    def _vqe_opt_pennylane(self, E_fn, method, params, max_iterations):
        """OS-VQO parameter optimization using SPSA."""
        # Choose optimizer
        # The selected hyperparameters are Pennylane's default.

        if method == "spsa":
            
            opt = qml.SPSAOptimizer(
                    maxiter=max_iterations, 
                    alpha=0.602, 
                    gamma=0.101, 
                    c=0.15, 
                    A=10, 
                    a=0.2
                    )
        else:
            raise ValueError(f"Unknown optax method: {method}")
        E_history = []
        params_history = []

        def cost_fn(p):
            E_val, _, _, _, _, _ = E_fn(p)
            return E_val
        
        for n in range(max_iterations):
            params, E_val = opt.step_and_cost(cost_fn, params)
            E_history.append(E_val)
            params_history.append(params)

            print(f"SPSA Step {n}: Energy = {E_val:.8f}")

            if n > 0 and abs(E_history[-1] - E_history[-2]) < self.conv_tol:
                break

        # Evaluate final rdm1 and other quantities
        _, rdm1_val, n_val, vecs_val, cj12_val, ck12_val = E_fn(params)
        return E_history, params_history, [rdm1_val], [n_val], [vecs_val], [cj12_val], [ck12_val]


    def _write_results(
        self,
        n,
        elag,
        E,
        E_HF,
        E_nuc,
        source="simulation",
        case_label=None,
        corrected=False,
        stats=None,
        status=None,
        flags=None,
        E_sim=None,
        n_sim=None,
        elag_sim=None,
        E_raw=None,
        n_raw=None,
        ):
        """
        Print occupation numbers and final energies in a unified format.

        Parameters
        ----------
        n : array
            Occupation numbers.
        elag : array
            Lagrange multiplier matrix.
        E : float
            Electronic NOF energy, without nuclear repulsion.
        E_HF : float
            HF total energy.
        E_nuc : float
            Nuclear repulsion energy.
        source : str
            One of: "simulation", "real", "noise_simulator", "hybrid-real",
            "hybrid-noise_simulator", etc.
        case_label : str
            Description of the occupations, e.g. "Raw QC occupations",
            "Physically projected occupations".
        corrected : bool
            Whether the occupations have been physically corrected/projected.
        stats : dict
            Optional diagnostic dictionary.
        status : str
            Optional status: "accepted", "flagged", etc.
        flags : list
            Optional list of diagnostic flags.
        E_sim : float
            Optional: Electronic NOF energy from simulator (Exact).
        n_sim : array
            Optional: Occupation numbers from simulator (Exact).
        elag_sim : array
            Optional: Lagrange multiplier matrix from simulator (Exact).
        E_raw : float
            Optional: Electronic NOF energy from QC without physically corrected/projected.
        n_sim : array
            Optional: Occupation numbers from QC without physically corrected/projected.
        """

        if case_label is None:
            case_label = "Occupation numbers"

        E_total = E_nuc + E

        print("")
        print("           RESULTS OF THE OCCUPATION OPTIMIZATION          ")
        print("===========================================================")
        print("")
        print(f"                 Source : {source}")
        print(f"            occupations : {case_label}")
        print(f"NOF physical projection : {corrected}")

        if status is not None:
            print(f"QC status   : {status}")

        if flags is not None:
            print(f"QC flags    : {flags}")
                
        if stats is not None and "accepted_for_statistics" in stats:
            print(f"Accepted for statistics : {stats['accepted_for_statistics']}")
                
        if n_sim is not None:
            print("")
            print("Reference simulator occupation vector:")
            print(n_sim)
            print("")

        print("")
        print("Evaluated/corrected occupation vector:")
        print(n)
        print("")

        print("")
        print(" {:^3}    {:^9}   {:>12}".format("Idx", "n", "E (Hartree)"))

        e_val = elag[np.diag_indices(self.nbf5)]

        for i in range(self.nbeta):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(
                i + 1, 2 * n[i], e_val[i]
            ))

        for i in range(self.nbeta, self.nalpha):
            if not self.HighSpin:
                print(" {:3d}    {:9.7f}  {:12.8f}".format(
                    i + 1, 2 * n[i], e_val[i]
                ))
            else:
                print(" {:3d}    {:9.7f}  {:12.8f}".format(
                    i + 1, n[i], e_val[i]
                ))

        for i in range(self.nalpha, self.nbf5):
            print(" {:3d}    {:9.7f}  {:12.8f}".format(
                i + 1, 2 * n[i], e_val[i]
            ))

        print("")
        print("--------------------------------------------------")
        print("                   Final Energies                 ")
        print("--------------------------------------------------")
        if E_HF is not None:
            print("")
            print("            HF Total Energy = {:15.12f}".format(E_HF))
        print("")
        print("     Final NOF Total Energy = {:15.12f}".format(E_total))
        if E_HF is not None:
            E_corr = E_total - E_HF
            print("         Correlation Energy = {:15.12f}".format(E_corr))
        print("")
        if E_sim is not None:
            E_total_sim = E_nuc + E_sim
            print("Sim. Final NOF Total Energy = {:15.12f}".format(E_total_sim))
            if E_HF is not None:
                E_corr_sim = E_total_sim - E_HF
                print("Sim.     Correlation Energy = {:15.12f}".format(E_corr_sim))
            print("")
            print("==================== Comparison ==================")
            print("")
            print("                 Abs_Diff_E = {:15.12f}".format(jnp.abs(E_total - E_total_sim)))
            print("                 ||Diff_n|| = {:15.12f}".format(jnp.linalg.norm(n - n_sim)))
            print("")


    def _binomial_upper_bound(
            self,
            p,
            shots,
            z=1.96,
        ):
        """
        Wilson upper confidence bound for a binomial probability.
        """

        p = float(
            np.clip(
                p,
                0.0,
                1.0,
            )
        )

        N = float(shots)

        denominator = (
            1.0
            + z**2 / N
        )

        center = (
            p
            + z**2 / (2.0 * N)
        )

        radius = z * np.sqrt(
            p * (1.0 - p) / N
            + z**2 / (4.0 * N**2)
        )

        upper = (
            center + radius
        ) / denominator

        return min(
            1.0,
            upper,
        )


    def _os_pair_is_residual_floor_eligible(
            self,
            p_logical,
        ):
        """
        Determine whether an OS pair is sufficiently close to the
        idempotent limit to allow residual-floor suppression.

        The first logical state corresponds to the strongly occupied
        orbital g.

        Residual-floor suppression is only allowed when the measured
        dominant occupation is very close to one. This protects
        genuinely fractional pairs from having physical weak
        occupations removed.
        """

        p = np.asarray(
            p_logical,
            dtype=float,
        )

        p = np.clip(
            p,
            0.0,
            1.0,
        )

        total = float(
            np.sum(p)
        )

        if total <= 0.0:
            return False

        p = p / total

        dominant = float(
            p[0]
        )

        weak_total = float(
            np.sum(p[1:])
        )

        threshold = float(
            getattr(
                self,
                "os_residual_floor_dominant_threshold",
                0.995,
            )
        )

        eligible = (
            dominant >= threshold
        )

        return eligible


    def _build_os_residual_floor_params(
            self,
            params,
            angle_eps=None,
        ):
        """
        Build OS-VQO parameters for residual-floor calibration.

        For every pair, theta_g is replaced by a very small angle so that

            n_g ~ 1
            n_p ~ 0

        while all secondary angles theta_p are kept equal to their
        optimized values.

        This preserves as much as possible of the actual gate sequence
        and rotation-angle structure used by the target circuit.
        """

        if angle_eps is None:
            angle_eps = self.os_residual_floor_angle_eps

        params_floor = np.asarray(
            params,
            dtype=float,
        ).copy()

        if not hasattr(self, "_os_layout"):
            self._os_layout = self._build_os_layout()

        for layout in self._os_layout:

            # Your layout already stores the parameter slice used
            # by each pair.
            param_slice = layout["parameter_slice"]

            start = param_slice.start

            # First angle of every pair is theta_g.
            params_floor[start] = angle_eps

        return params_floor


    def _os_valid_probability_indices(self, layout):
        """
        Return computational-basis probability indices corresponding
        to physical OS states in this pair.
        """

        indices = []

        for bits in layout["bitstrings"]:

            index = 0

            for bit in bits:
                index = (index << 1) | int(bit)

            indices.append(index)

        return indices


    def _apply_os_residual_floor(
            self,
            probabilities,
            residual_floor,
        ):
        """
        Suppress weak OS populations that cannot be statistically
        distinguished from the calibrated residual noise floor.

        The correction is applied only to near-idempotent pair spaces.

        For an eligible pair, a non-dominant logical population i
        is suppressed when its measured value is statistically
        compatible with the calibrated residual floor.

        Genuinely fractional pairs are left untouched.
        """

        corrected = []

        k_sigma = float(
            self.os_residual_floor_sigma
        )

        shots = float(
            self.n_shots
        )

        print("")
        print("==========================================")
        print("     OS-VQO RESIDUAL-FLOOR MITIGATION")
        print("==========================================")
        print("")

        for pair_index, layout in enumerate(
            self._os_layout
        ):

            # =====================================================
            # Full computational-basis probability vectors
            # =====================================================

            p_full = np.asarray(
                probabilities[pair_index],
                dtype=float,
            ).copy()

            floor_full = np.asarray(
                residual_floor[pair_index],
                dtype=float,
            ).copy()

            # =====================================================
            # Logical states belonging to this OS subspace
            # =====================================================

            valid_indices = (
                self._os_valid_probability_indices(
                    layout
                )
            )

            p = p_full[
                valid_indices
            ].copy()

            floor = floor_full[
                valid_indices
            ].copy()

            # =====================================================
            # Normalize logical distributions
            # =====================================================

            p_norm = np.sum(p)

            if p_norm <= 0.0:
                raise RuntimeError(
                    f"Pair {pair_index}: "
                    "zero probability in logical subspace."
                )

            p /= p_norm

            floor_norm = np.sum(floor)

            if floor_norm > 0.0:
                floor /= floor_norm

            # =====================================================
            # Dominant logical state
            #
            # In the OS encoding, logical state 0 corresponds
            # to the strongly occupied orbital g.
            # =====================================================

            dominant = 0

            p_dominant = float(
                p[dominant]
            )

            weak_total = float(
                np.sum(
                    p[1:]
                )
            )

            print(
                f"Pair {pair_index}: "
                f"Omega={layout['orbitals']}"
            )

            print(
                f"  dominant population = "
                f"{p_dominant:.8f}"
            )

            print(
                f"  total weak population = "
                f"{weak_total:.8e}"
            )

            print(
                f"  eligibility threshold = "
                f"{self.os_residual_floor_dominant_threshold:.6f}"
            )

            # =====================================================
            # Determine whether residual-floor suppression is
            # allowed for this pair.
            # =====================================================

            eligible = (
                self._os_pair_is_residual_floor_eligible(
                    p
                )
            )

            print(
                f"  residual-floor eligible = "
                f"{eligible}"
            )

            # =====================================================
            # Genuinely fractional pair:
            # leave the distribution untouched.
            # =====================================================

            if not eligible:

                print(
                    "  -> pair left unchanged"
                )

                # Reconstruct the full probability vector.
                #
                # This is slightly safer than simply appending
                # p_full because p has been explicitly normalized
                # over the valid logical subspace.

                p_corrected_full = np.zeros_like(
                    p_full
                )

                for logical_i, full_i in enumerate(
                    valid_indices
                ):

                    p_corrected_full[
                        full_i
                    ] = p[
                        logical_i
                    ]

                corrected.append(
                    p_corrected_full
                )

                print(
                    "  after  =",
                    p,
                )

                print("")

                continue

            # =====================================================
            # Near-idempotent pair:
            # test weak populations individually.
            # =====================================================

            print(
                "  near-idempotent pair -> "
                "testing weak populations"
            )

            original_p = p.copy()

            for i in range(
                len(p)
            ):

                if i == dominant:
                    continue

                # -------------------------------------------------
                # Calibrated residual floor
                # -------------------------------------------------

                f = float(
                    np.clip(
                        floor[i],
                        0.0,
                        1.0,
                    )
                )

                # -------------------------------------------------
                # Measured logical population
                # -------------------------------------------------

                p_i = float(
                    np.clip(
                        p[i],
                        0.0,
                        1.0,
                    )
                )

                # -------------------------------------------------
                # Approximate shot uncertainty of measured p_i
                # -------------------------------------------------

                sigma_measured = np.sqrt(
                    max(
                        p_i
                        * (1.0 - p_i)
                        / shots,
                        0.0,
                    )
                )

                # -------------------------------------------------
                # Wilson upper confidence bound for residual floor
                # -------------------------------------------------

                floor_upper = (
                    self._binomial_upper_bound(
                        f,
                        shots,
                        z=1.96,
                    )
                )

                threshold = (
                    floor_upper
                    + k_sigma
                    * sigma_measured
                )

                print(
                    f"    state {i}: "
                    f"p={p_i:.8e}, "
                    f"floor={f:.8e}, "
                    f"floor_upper={floor_upper:.8e}, "
                    f"sigma_meas={sigma_measured:.3e}, "
                    f"limit={threshold:.8e}",
                    end="",
                )

                # -------------------------------------------------
                # Suppress statistically unresolved occupation
                # -------------------------------------------------

                if p_i <= threshold:

                    p[i] = 0.0

                    print(
                        "  -> SUPPRESSED"
                    )

                else:

                    p_excess = max(
                        p_i - f,
                        0.0,
                    )

                    p[i] = p_excess

                    print(
                        f"  -> FLOOR SUBTRACTED "
                        f"(corrected={p_excess:.8e})"
                    )

            # =====================================================
            # Renormalize corrected logical pair distribution
            # =====================================================

            norm = np.sum(
                p
            )

            if norm <= 0.0:

                raise RuntimeError(
                    "Residual-floor mitigation removed "
                    "all probability from a pair."
                )

            p /= norm

            # =====================================================
            # Put logical probabilities back into complete
            # computational-basis vector
            # =====================================================

            p_corrected_full = np.zeros_like(
                p_full
            )

            for logical_i, full_i in enumerate(
                valid_indices
            ):

                p_corrected_full[
                    full_i
                ] = p[
                    logical_i
                ]

            corrected.append(
                p_corrected_full
            )

            print(
                "  before =",
                original_p,
            )

            print(
                "  after  =",
                p,
            )

            print("")

        return corrected


    def _build_os_residual_floor(
            self,
            params,
        ):
        """
        Measure the residual population generated by the OS-VQO
        circuit when every pair is prepared close to its dominant
        occupation state.

        The same backend, physical layout, circuit topology,
        readout mitigation, and logical-subspace postselection
        are used as in the production calculation.
        """

        print("")
        print("==========================================")
        print("   OS-VQO RESIDUAL-FLOOR CALIBRATION")
        print("==========================================")
        print("")

        # -----------------------------------------------------
        # Build near-idempotent parameters
        # -----------------------------------------------------

        params_floor = (
            self._build_os_residual_floor_params(
                params
            )
        )

        # -----------------------------------------------------
        # Raw calibration probabilities
        # IMPORTANT: call QNode directly, not _evaluate_rdm1()
        # -----------------------------------------------------

        probabilities_floor_raw = (
            self._os_qnode(
                params_floor
            )
        )

        # -----------------------------------------------------
        # Same readout mitigation as production calculation
        # -----------------------------------------------------

        if getattr(
            self,
            "os_readout_mitigation",
            False,
        ):

            probabilities_floor_readout = (
                self._mitigate_os_readout(
                    probabilities_floor_raw
                )
            )

        else:

            probabilities_floor_readout = (
                probabilities_floor_raw
            )

        # -----------------------------------------------------
        # Same logical-subspace postselection
        # -----------------------------------------------------

        if getattr(
            self,
            "os_subspace_postselection",
            False,
        ):

            probabilities_floor_final, _ = (
                self._os_postselect_probabilities(
                    probabilities_floor_readout
                )
            )

        else:

            probabilities_floor_final = (
                probabilities_floor_readout
            )

        # -----------------------------------------------------
        # Store calibration
        # -----------------------------------------------------

        self._last_os_residual_floor_raw = (
            probabilities_floor_raw
        )

        self._last_os_residual_floor = (
            probabilities_floor_final
        )

        # -----------------------------------------------------
        # Report
        # -----------------------------------------------------

        for pair_index, layout in enumerate(
            self._os_layout
        ):

            p = np.asarray(
                probabilities_floor_final[
                    pair_index
                ],
                dtype=float,
            )

            valid_indices = (
                self._os_valid_probability_indices(
                    layout
                )
            )

            logical_p = p[
                valid_indices
            ]

            print(
                f"Pair {pair_index}: "
                f"orbitals={layout['orbitals']}"
            )

            print(
                "  calibrated floor =",
                logical_p,
            )

            print(
                "  total weak residual = "
                f"{np.sum(logical_p[1:]):.8e}"
            )

            print("")

        return probabilities_floor_final


    def _build_os_response_matrices(
            self,
            params_target,
        ):
        """
        Experimentally construct pair-local logical response matrices.

        B[i,j] = probability of measuring logical state i
                when logical calibration state j was prepared.

        IMPORTANT:
        For the dominant g-state calibration (column 0), use the
        optimized secondary angles of the actual target circuit and
        replace only theta_g by a very small value.

        Thus, for an m=3 pair with optimized parameters

            [theta_g*, theta_p1*, theta_p2*]

        the g-state calibration uses

            [eps, theta_p1*, theta_p2*]

        rather than a generic basis-state parameter set.

        This tests whether the residual populations observed on hardware
        depend on the actual CRY angles of the target OS-VQO circuit.

        For logical states 1, 2, and 3, the existing OS-circuit
        basis-targeting parameter sets are retained.

        Readout mitigation is applied to the calibration data before
        constructing the response matrix.
        """

        # =========================================================
        # Target parameters
        # =========================================================

        params_target = np.asarray(
            params_target,
            dtype=float,
        )

        # =========================================================
        # Build response QNodes if necessary
        # =========================================================

        if not hasattr(
            self,
            "_os_response_qnodes",
        ):
            self._build_os_response_qnodes()

        response_matrices = []

        print("")
        print("==========================================")
        print("       OS-VQO CIRCUIT RESPONSE")
        print("          MATRIX CALIBRATION")
        print("==========================================")
        print("")

        # =========================================================
        # Loop over OS pair spaces
        # =========================================================

        for pair_index, layout in enumerate(
            self._os_layout
        ):

            orbitals = layout["orbitals"]
            m = len(orbitals) - 1

            # -----------------------------------------------------
            # Only calibrate the requested pair-space dimensions
            # -----------------------------------------------------

            if m not in self.os_response_m_values:

                response_matrices.append(
                    None
                )

                continue

            # -----------------------------------------------------
            # Current optimized parameters for this pair
            # -----------------------------------------------------

            target_pair_params = np.asarray(
                params_target[
                    layout["parameter_slice"]
                ],
                dtype=float,
            ).copy()

            qnode = self._os_response_qnodes[
                pair_index
            ]

            valid_indices = (
                self._os_valid_probability_indices(
                    layout
                )
            )

            d = len(
                valid_indices
            )

            B = np.zeros(
                (d, d),
                dtype=float,
            )

            print(
                f"Pair {pair_index}: "
                f"Omega={orbitals}"
            )

            print(
                "  target optimized params =",
                target_pair_params,
            )

            print("")

            # =====================================================
            # Build each response-matrix column
            # =====================================================

            for logical_j in range(
                d
            ):

                # =================================================
                # Construct calibration parameters
                # =================================================

                if logical_j == 0:

                    # -------------------------------------------------
                    # TARGET-MATCHED dominant-state calibration
                    #
                    # Preserve the optimized secondary rotations:
                    #
                    #   theta_p1 = theta_p1*
                    #   theta_p2 = theta_p2*
                    #
                    # and force only:
                    #
                    #   theta_g -> eps
                    #
                    # Ideally this still prepares approximately |g>,
                    # but now the hardware executes the secondary
                    # rotations relevant to the actual target circuit.
                    # -------------------------------------------------

                    params_cal = (
                        target_pair_params.copy()
                    )

                    params_cal[0] = (
                        self.os_residual_floor_angle_eps
                    )

                    calibration_type = (
                        "TARGET-MATCHED g"
                    )

                else:

                    # -------------------------------------------------
                    # Retain the OS-circuit logical-basis calibration
                    # for p1, p2, p3.
                    # -------------------------------------------------

                    params_cal = (
                        self._os_response_basis_params_m3(
                            logical_j,
                            eps=1.0e-6,
                        )
                    )

                    calibration_type = (
                        "logical-basis"
                    )

                print(
                    f"  logical {logical_j} "
                    f"[{calibration_type}] "
                    f"params = {params_cal}"
                )

                # =================================================
                # Hardware calibration measurement
                # =================================================

                p_raw = np.asarray(
                    qnode(
                        params_cal
                    ),
                    dtype=float,
                )

                # =================================================
                # Apply the same readout mitigation used for
                # the production OS-VQO calculation
                # =================================================

                if getattr(
                    self,
                    "os_readout_mitigation",
                    False,
                ):

                    if not hasattr(
                        self,
                        "_os_readout_assignment_matrices",
                    ):

                        self._build_os_readout_assignment_matrices()

                    A = (
                        self._os_readout_assignment_matrices[
                            pair_index
                        ]
                    )

                    p_readout = (
                        self._mitigate_readout_distribution(
                            p_raw,
                            A,
                        )
                    )

                else:

                    p_readout = p_raw

                # =================================================
                # Extract the valid logical OS states
                # =================================================

                p_logical = p_readout[
                    valid_indices
                ].copy()

                # =================================================
                # Normalize logical distribution
                # =================================================

                norm = np.sum(
                    p_logical
                )

                if norm <= 0.0:

                    raise RuntimeError(
                        f"Response calibration failed "
                        f"for pair {pair_index}, "
                        f"logical state {logical_j}."
                    )

                p_logical /= norm

                # =================================================
                # Special diagnostic for the target-matched g state
                # =================================================

                if logical_j == 0:

                    print(
                        "  >>> target-matched g residual "
                        "weak populations =",
                        p_logical[1:],
                    )

                    print(
                        "  >>> total target-matched g "
                        "weak residual =",
                        np.sum(
                            p_logical[1:]
                        ),
                    )

                # =================================================
                # Store column j
                #
                # B[i,j] =
                # P(measured logical i | prepared logical j)
                # =================================================

                B[
                    :,
                    logical_j
                ] = p_logical

                print(
                    f"  prepared logical {logical_j}: "
                    f"{p_logical}"
                )

                print("")

            # =====================================================
            # Store/report this pair's response matrix
            # =====================================================

            response_matrices.append(
                B
            )

            print("")
            print(
                "  Response matrix B:"
            )
            print(B)

            print("")

            print(
                "  column sums =",
                np.sum(
                    B,
                    axis=0,
                ),
            )

            print("")
            print("------------------------------------------")
            print("")

        # =========================================================
        # Cache response calibration
        # =========================================================

        self._os_response_matrices = (
            response_matrices
        )

        # The response matrix now depends on the target parameters,
        # not only on backend/layout.
        self._os_response_matrix_params = (
            np.asarray(
                params_target,
                dtype=float,
            ).copy()
        )

        return response_matrices


    def _os_response_basis_params_m3(
            self,
            logical_index,
            eps=1.0e-6,
        ):
        """
        Parameters for preparing each m=3 OS logical basis state
        using the actual specialized OS circuit.

        Mapping:
            0 -> g  -> |00>
            1 -> p1 -> |01>
            2 -> p2 -> |11>
            3 -> p3 -> |10>

        Specialized m=3 circuit:
            RY(theta_g) on q1
            CRY(pi - theta_p1) q1 -> q0
            CRY(pi - theta_p2) q0 -> q1

        Small eps values avoid exact zero/pi boundary parameters.
        """

        eps = float(eps)

        if logical_index == 0:
            # Ideally |00>
            return np.array([
                eps,
                0.5 * np.pi,
                0.5 * np.pi,
            ])

        elif logical_index == 1:
            # Ideally |01>
            return np.array([
                np.pi - eps,
                np.pi - eps,
                0.5 * np.pi,
            ])

        elif logical_index == 2:
            # Ideally |11>
            return np.array([
                np.pi - eps,
                eps,
                np.pi - eps,
            ])

        elif logical_index == 3:
            # Ideally |10>
            return np.array([
                np.pi - eps,
                eps,
                eps,
            ])

        else:
            raise ValueError(
                f"Invalid m=3 logical index: {logical_index}"
            )


    def _apply_os_response_mitigation(
            self,
            probabilities,
            response_matrices,
        ):
        """
        Apply pair-local circuit-response mitigation.

        Input probabilities are assumed to have already undergone
        readout mitigation and logical-subspace postselection.
        """

        corrected = []

        print("")
        print("==========================================")
        print("      OS-VQO RESPONSE MITIGATION")
        print("==========================================")
        print("")

        for pair_index, layout in enumerate(
            self._os_layout
        ):

            p_full = np.asarray(
                probabilities[
                    pair_index
                ],
                dtype=float,
            ).copy()

            B = response_matrices[
                pair_index
            ]

            # No response mitigation for this pair.
            if B is None:

                corrected.append(
                    p_full
                )

                continue

            valid_indices = (
                self._os_valid_probability_indices(
                    layout
                )
            )

            p_logical = p_full[
                valid_indices
            ].copy()

            norm = np.sum(
                p_logical
            )

            if norm <= 0.0:

                raise RuntimeError(
                    f"Pair {pair_index}: "
                    "zero logical probability before "
                    "response mitigation."
                )

            p_logical /= norm

            p_corrected = (
                self._invert_os_response_distribution(
                    p_logical,
                    B,
                )
            )

            p_corrected_full = np.zeros_like(
                p_full
            )

            for logical_i, full_i in enumerate(
                valid_indices
            ):

                p_corrected_full[
                    full_i
                ] = p_corrected[
                    logical_i
                ]

            corrected.append(
                p_corrected_full
            )

            print(
                f"Pair {pair_index}: "
                f"Omega={layout['orbitals']}"
            )

            print(
                "  before =",
                p_logical,
            )

            print(
                "  after  =",
                p_corrected,
            )

            print(
                "  ||Delta response|| =",
                np.linalg.norm(
                    p_corrected
                    - p_logical
                ),
            )

            print("")

        return corrected


    def _invert_os_response_distribution(
            self,
            p_measured,
            B,
        ):
        """
        Recover the most likely physical logical distribution from

            p_measured ~= B @ p_true

        subject to

            p_true[i] >= 0
            sum(p_true) = 1.
        """

        p_measured = np.asarray(
            p_measured,
            dtype=float,
        )

        B = np.asarray(
            B,
            dtype=float,
        )

        d = len(
            p_measured
        )

        # Start from measured distribution.
        x0 = np.clip(
            p_measured,
            0.0,
            1.0,
        )

        x0 /= np.sum(
            x0
        )

        def objective(
                x,
            ):

            residual = (
                B @ x
                - p_measured
            )

            return float(
                np.dot(
                    residual,
                    residual,
                )
            )

        constraints = [
            {
                "type": "eq",
                "fun": lambda x: (
                    np.sum(x) - 1.0
                ),
            }
        ]

        bounds = [
            (0.0, 1.0)
            for _ in range(d)
        ]

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "ftol": 1.0e-12,
                "maxiter": 1000,
            },
        )

        if not result.success:

            raise RuntimeError(
                "OS response-matrix inversion failed: "
                + result.message
            )

        p_corrected = np.clip(
            result.x,
            0.0,
            1.0,
        )

        p_corrected /= np.sum(
            p_corrected
        )

        return p_corrected


    def _build_os_response_qnodes(self):
        """
        Build pair-local response-calibration QNodes.

        IMPORTANT:
        The logical calibration states are prepared through the
        SAME specialized OS circuit used in the production
        calculation, rather than directly with X gates.

        This allows the response matrix to probe errors introduced
        by the actual RY/CRY state-preparation circuit.
        """

        response_qnodes = []

        for pair_index, layout in enumerate(
            self._os_layout
        ):

            orbitals = layout["orbitals"]
            m = len(orbitals) - 1

            if m not in self.os_response_m_values:

                response_qnodes.append(None)
                continue

            # For this first implementation we explicitly support
            # the H2O-type m=3 pair.
            if m != 3:

                raise NotImplementedError(
                    "Circuit-aware response calibration is "
                    "currently implemented only for m=3."
                )

            layout_local = layout
            dev = self.dev_instance

            def make_qnode(layout_fixed):

                wires_fixed = layout_fixed["wires"]

                @qml.qnode(dev)
                def response_qnode(params_cal):

                    # IMPORTANT:
                    # use exactly the same m=3 OS preparation
                    # as the production circuit.
                    self._prepare_os_pair_m3(
                        params_cal,
                        layout_fixed,
                    )

                    return qml.probs(
                        wires=wires_fixed
                    )

                return response_qnode

            response_qnodes.append(
                make_qnode(layout_local)
            )

        self._os_response_qnodes = response_qnodes

        return response_qnodes


    def _report_os_readout_mitigation(
        self,
        raw_probabilities,
        corrected_probabilities,
        ):
        """
        Compact report of pair-local OS-VQO readout mitigation.
        """

        print("")
        print("==========================================")
        print("       OS-VQO READOUT MITIGATION")
        print("==========================================")

        for pair_index, (
            raw,
            corrected,
            layout,
        ) in enumerate(
            zip(
                raw_probabilities,
                corrected_probabilities,
                self._os_layout,
            )
        ):

            raw = np.asarray(
                raw,
                dtype=float,
            )

            corrected = np.asarray(
                corrected,
                dtype=float,
            )

            print("")
            print(
                f"Pair {pair_index}: "
                f"m={layout['m']}, "
                f"wires={layout['wires']}"
            )

            print(
                "  ||Delta p|| =",
                np.linalg.norm(
                    corrected - raw
                ),
            )


    def _nuclear_gradient_analytics(self, n,C,cj12,ck12,elag):
        
        mints = psi4.core.MintsHelper(self.wfn.basisset())
        
        RDM1 = 2*jnp.einsum('p,mp,np->mn',n[:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        lag = 2*jnp.einsum('mq,qp,np->mn',C,elag,C,optimize=True)
        
        grad = jnp.zeros((self.natoms,3))
        
        grad += jnp.array(self.mol.nuclear_repulsion_energy_deriv1())
        
        for i in range(self.natoms):
            dSx,dSy,dSz = jnp.array(mints.ao_oei_deriv1("OVERLAP",i))
            
            grad = grad.at[i,0].add(-jnp.einsum('mn,mn->',lag,dSx,optimize=True))
            grad = grad.at[i,1].add(-jnp.einsum('mn,mn->',lag,dSy,optimize=True))
            grad = grad.at[i,2].add(-jnp.einsum('mn,mn->',lag,dSz,optimize=True))
            
            dTx,dTy,dTz = jnp.array(mints.ao_oei_deriv1("KINETIC",i))
            grad = grad.at[i,0].add(jnp.einsum('mn,mn->',RDM1,dTx,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum('mn,mn->',RDM1,dTy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum('mn,mn->',RDM1,dTz,optimize=True))

            dVx,dVy,dVz = jnp.array(mints.ao_oei_deriv1("POTENTIAL",i))
            grad = grad.at[i,0].add(jnp.einsum('mn,mn->',RDM1,dVx,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum('mn,mn->',RDM1,dVy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum('mn,mn->',RDM1,dVz,optimize=True))
        
        cj12 = cj12 - jnp.diag(jnp.diag(cj12)) # Remove diag.
        ck12 = ck12 - jnp.diag(jnp.diag(ck12)) # Remove diag.
        
        #RDM2 = jnp.einsum('pq,mp,np,sq,lq->mnsl',cj12,C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 = jnp.einsum('pq,mp,np,sq,lq->mnsl',cj12[:self.nbf5,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 += jnp.einsum('p,mp,np,sp,lp->mnsl',n[:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],C[:,:self.nbeta],optimize=True)
        RDM2 += jnp.einsum('p,mp,np,sp,lp->mnsl',n[self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],C[:,self.nalpha:self.nbf5],optimize=True)
        #RDM2 -= jnp.einsum('pq,mp,lp,sq,nq->mnsl',ck12,C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)
        RDM2 -= jnp.einsum('pq,mp,lp,sq,nq->mnsl',ck12[:self.nbf5,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],C[:,:self.nbf5],optimize=True)

        for i in range(self.natoms):
            derix,deriy,deriz = jnp.array(mints.ao_tei_deriv1(i))
            grad = grad.at[i,0].add(jnp.einsum("mnsl,mnsl->",RDM2,derix,optimize=True))
            grad = grad.at[i,1].add(jnp.einsum("mnsl,mnsl->",RDM2,deriy,optimize=True))
            grad = grad.at[i,2].add(jnp.einsum("mnsl,mnsl->",RDM2,deriz,optimize=True))
        return grad

def _main(argv=None):
    """Command-line entry point for OS-VQO single-geometry calculations.

    Positional arguments preserve the useful legacy order:
        xyz_file functional basis save_params [layout]

    Example:
        python osvqo.py h2_bohr.xyz pnof7 sto-3g T
    """
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Run an OS-VQO/PNOF calculation for one molecular geometry."
    )
    parser.add_argument("xyz_file", help="XYZ-like geometry file (including units, charge, and multiplicity)")
    parser.add_argument("functional", help="PNOF functional, e.g. pnof7")
    parser.add_argument("basis", help="Psi4 basis set, e.g. sto-3g")
    parser.add_argument(
        "save_params",
        choices=("T", "F", "t", "f"),
        help="T keeps generated .npy files; F removes .npy files and timer.dat after the run",
    )
    parser.add_argument(
        "layout",
        nargs="?",
        default=None,
        help="Optional named hardware layout: h2o, li2, or lih",
    )
    parser.add_argument(
        "--device",
        choices=("simulator", "real", "noise_simulator", "hybrid_real", "hybrid_noise_simulator"),
        default="simulator",
        help="Execution mode (default: simulator)",
    )
    parser.add_argument(
        "--runtime-service-name",
        default=None,
        help="Saved IBM Quantum Runtime account profile name (default selected from region)",
    )
    args = parser.parse_args(argv)

    if args.layout == "h2o":
        initial_layout = [18, 31, 61, 62, 33, 39, 41, 42, 148, 149]
    elif args.layout == "li2":
        initial_layout = [18, 31, 61, 62, 33, 39]
    elif args.layout == "lih":
        initial_layout = [21, 22, 60, 61]
    elif args.layout is None:
        initial_layout = None
    else:
        parser.error("layout must be one of: h2o, li2, lih")

    dev = args.device
    opt_circ = "cobyla" if dev in ("real", "noise_simulator") else "slsqp"

    calculator = OSVQO(
        args.xyz_file,
        functional=args.functional,
        conv_tol=1e-6,
        init_param=None,
        basis=args.basis,
        max_iterations=200,
        opt_circ=opt_circ,
        dev=dev,
        n_shots=10000,
        optimization_level=3,
        resilience_level=0,
        initial_layout=initial_layout,
        os_readout_mitigation=True,
        os_error_mitigation=True,
        runtime_service_name=args.runtime_service_name,
    )

    result = calculator.run()
    energy = float(result[0])
    print(f"OS-VQO total energy ({args.functional}, {args.basis}): {energy:.12f} Ha")

    if args.save_params.upper() == "F":
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        Path("timer.dat").unlink(missing_ok=True)
        for path in Path(".").glob("*.npy"):
            path.unlink()

if __name__ == "__main__":
    _main()
