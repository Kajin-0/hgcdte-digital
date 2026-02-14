"""
===========================================================================
MCT (HgCdTe) Photoconductor Drift–Diffusion Solver — Stages 0–3
===========================================================================

Coupled Poisson + electron continuity for a biased bulk photoconductor.

Physics (Stage 1 — pure biased resistor, G=R=0):
    Poisson:      ∇·(ε ∇φ) = -q (Nd - n)
    Continuity:   ∇·Jn = 0
    Jn = -q μn n ∇φ + q Dn ∇n    (Dn = μn Vt, Einstein relation)

Nondimensionalization:
    φ̂ = φ / Vt       (thermal voltage)
    x̂ = x / L_norm   (largest device dimension)
    n̂ = n / n_ref     (reference concentration)

    Dimensionless Poisson:
        ∇̂²φ̂ = -α_eff (N̂d - n̂)

    Dimensionless continuity (weak form):
        ∫ (-n̂ ∇̂φ̂ + ∇̂n̂) · ∇̂w dx̂ = 0

Boundary conditions (ohmic contacts):
    φ̂(0) = 0,  φ̂(L) = V_bias/Vt
    n̂(0) = Nd/n_ref,  n̂(L) = Nd/n_ref

Usage:
    # Stage 1: biased resistor (G=R=0)
    mpirun -n 8 python3 photoconductor_dd.py --Nd 1e22 --V_bias 0.1

    # Stage 2: with recombination
    mpirun -n 8 python3 photoconductor_dd.py --Nd 1e22 --V_bias 0.1 --tau_n 1e-6

    # Stage 3: with optical generation
    mpirun -n 8 python3 photoconductor_dd.py --Nd 1e22 --V_bias 0.1 --tau_n 1e-6 --G0 1e25 --alpha_opt 1e6
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI

# ── PETSc init (before importing PETSc) ──
_petsc_argv = [sys.argv[0]]
_app_argv = [sys.argv[0]]
_i = 1
while _i < len(sys.argv):
    arg = sys.argv[_i]
    if arg.startswith("--"):
        _app_argv.append(arg)
        if _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("-"):
            _app_argv.append(sys.argv[_i + 1])
            _i += 2
        else:
            _i += 1
    else:
        _petsc_argv.append(arg)
        if _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("-"):
            _petsc_argv.append(sys.argv[_i + 1])
            _i += 2
        else:
            _i += 1

import petsc4py
petsc4py.init(_petsc_argv)
from petsc4py import PETSc

sys.argv = _app_argv

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.mesh as dmesh
import dolfinx.io
import ufl

# ════════════════════════════════════════════════════════════════════════
#  1.  PHYSICAL CONSTANTS AND SCALING
# ════════════════════════════════════════════════════════════════════════

Q_E   = 1.602176634e-19       # C
K_B   = 1.380649e-23          # J/K
EPS_0 = 8.8541878128e-12      # F/m


@dataclass(frozen=True)
class DeviceParams:
    """Material and operating parameters."""
    temperature: float = 77.0      # K
    eps_r: float = 16.7            # MCT static dielectric constant
    n_ref: float = 1e21            # reference carrier density [m⁻³]
    L_ref: float = 10e-6           # reference length [m] (device thickness)
    mu_n: float = 1.0              # electron mobility [m²/Vs] (MCT ~1 at 77K)
    Nd: float = 1e22               # donor concentration [m⁻³]
    Na: float = 0.0                # acceptor concentration [m⁻³]
    V_bias: float = 0.1            # applied voltage [V]
    tau_n: float = 0.0             # electron lifetime [s] (0 = no recombination)
    G0: float = 0.0                # optical generation rate peak [m⁻³ s⁻¹]
    alpha_opt: float = 0.0         # optical absorption coefficient [m⁻¹]

    @property
    def V_t(self) -> float:
        """Thermal voltage kT/q [V]."""
        return K_B * self.temperature / Q_E

    @property
    def D_n(self) -> float:
        """Electron diffusion coefficient [m²/s] (Einstein relation)."""
        return self.mu_n * self.V_t

    @property
    def alpha(self) -> float:
        """Poisson coupling constant (dimensionless)."""
        return (Q_E * self.n_ref * self.L_ref**2) / (EPS_0 * self.eps_r * self.V_t)

    @property
    def L_D(self) -> float:
        """Debye length [m]."""
        return np.sqrt(EPS_0 * self.eps_r * self.V_t / (Q_E * self.Nd))

    def summary(self, L_norm: float) -> str:
        alpha_eff = self.alpha * (self.L_ref / L_norm)**2
        return (
            "=== Device Parameters ===\n"
            f"  T       = {self.temperature:.1f} K\n"
            f"  V_t     = {self.V_t*1e3:.4f} mV\n"
            f"  eps_r   = {self.eps_r:.2f}\n"
            f"  n_ref   = {self.n_ref:.3e} m⁻³\n"
            f"  L_ref   = {self.L_ref*1e6:.1f} µm\n"
            f"  L_D     = {self.L_D*1e6:.4f} µm\n"
            f"  mu_n    = {self.mu_n:.2f} m²/Vs\n"
            f"  D_n     = {self.D_n:.4e} m²/s\n"
            f"  alpha   = {self.alpha:.6e}\n"
            f"  alpha_eff = {alpha_eff:.6e}\n"
            f"  Nd      = {self.Nd:.3e} m⁻³\n"
            f"  Nd/nref = {self.Nd/self.n_ref:.6e}\n"
            f"  V_bias  = {self.V_bias:.4f} V\n"
            f"  V̂_bias  = {self.V_bias/self.V_t:.4f}\n"
            f"  tau_n   = {self.tau_n:.3e} s\n"
            f"  G0      = {self.G0:.3e} m⁻³s⁻¹\n"
            "========================="
        )


# ════════════════════════════════════════════════════════════════════════
#  2.  GEOMETRY AND MESH
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeviceGeometry:
    """1D photoconductor geometry for drift-diffusion."""
    L: float = 10e-6     # device length [m] (= thickness for photoconductor)


def create_1d_mesh(geom: DeviceGeometry, L_norm: float, n_cells: int,
                   comm: MPI.Comm):
    """Create a 1D mesh in dimensionless coordinates [0, L/L_norm].

    1D simplifies drift-diffusion validation before going to 3D.
    """
    L_hat = geom.L / L_norm
    msh = dmesh.create_interval(comm, n_cells, points=[0.0, L_hat])
    return msh, {"L_hat": L_hat, "L_norm": L_norm, "L_ref": geom.L}


# ════════════════════════════════════════════════════════════════════════
#  3.  WEAK FORM — COUPLED POISSON + CONTINUITY
# ════════════════════════════════════════════════════════════════════════

def build_dd_system(msh, params: DeviceParams, dims: dict, comm: MPI.Comm):
    """
    Build the coupled Poisson + electron continuity weak form.

    Mixed function space: W = (φ̂, n̂) ∈ CG1 × CG1

    Poisson (residual form, tested by v):
        F_phi = ∫ ∇̂φ̂ · ∇̂v dx̂ + α_eff ∫ (n̂ - N̂d) v dx̂ = 0

    Note: Poisson is  ∇²φ = -q/ε (Nd - n)
          Weak form:  -∫ ∇φ · ∇v dx = -q/ε ∫ (Nd - n) v dx
          Rearranged: ∫ ∇φ · ∇v dx = q/ε ∫ (Nd - n) v dx
                    = α_eff ∫ (N̂d - n̂) v dx̂
          So:  ∫ ∇̂φ̂ · ∇̂v dx̂ - α_eff ∫ (N̂d - n̂) v dx̂ = 0
          i.e. ∫ ∇̂φ̂ · ∇̂v dx̂ + α_eff ∫ (n̂ - N̂d) v dx̂ = 0

    Continuity (tested by w):
        ∇·Jn = 0  where  Jn = -q μn n ∇φ + q Dn ∇n
        Dimensionless: Jn ∝ (-n̂ ∇̂φ̂ + ∇̂n̂)
        Weak: -∫ Jn · ∇w dx = 0  (from IBP of ∇·Jn w)
        →  ∫ (n̂ ∇̂φ̂ - ∇̂n̂) · ∇̂w dx̂ = 0
        which is:  ∫ n̂ ∇̂φ̂ · ∇̂w dx̂ - ∫ ∇̂n̂ · ∇̂w dx̂ = 0

    With recombination (Stage 2):
        ∇·Jn = q(G - R) → dimensionless:
        ∫ (n̂ ∇̂φ̂ - ∇̂n̂) · ∇̂w dx̂ = β ∫ (Ĝ - R̂) w dx̂
        where β = L_norm² / (μn Vt) = scaling factor for source terms
        R̂ = (n̂ - n̂_eq) / τ̂_n
        Ĝ = G0_hat * exp(-α_opt_hat * x)

    Returns (W, F_form, J_form, bcs, u_hat) where u_hat = (φ̂, n̂).
    """
    # Mixed function space: CG1 × CG1
    # DOLFINx 0.10 uses basix.ufl for element creation
    try:
        import basix.ufl as bufl
        elem_phi = bufl.element("Lagrange", msh.basix_cell(), 1)
        elem_n = bufl.element("Lagrange", msh.basix_cell(), 1)
        mel = bufl.mixed_element([elem_phi, elem_n])
    except (ImportError, AttributeError):
        # Fallback for older DOLFINx
        mel = ufl.MixedElement([ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1),
                                ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)])
    W = fem.functionspace(msh, mel)

    # Solution and test functions
    u_hat = fem.Function(W, name="u_hat")
    phi_hat, n_hat = ufl.split(u_hat)       # UFL symbolic splits
    v, w = ufl.TestFunctions(W)

    # Scaling
    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = params.alpha * (L_ref / L_norm)**2
    Nd_hat = params.Nd / params.n_ref

    # --- Poisson residual ---
    F_phi = (ufl.inner(ufl.grad(phi_hat), ufl.grad(v)) * ufl.dx
             + alpha_eff * (n_hat - Nd_hat) * v * ufl.dx)

    # --- Continuity residual ---
    # ∫ (n̂ ∇̂φ̂ - ∇̂n̂) · ∇̂w dx̂ = β ∫ (G - R) w dx̂
    F_cont = (ufl.inner(n_hat * ufl.grad(phi_hat) - ufl.grad(n_hat),
                         ufl.grad(w)) * ufl.dx)


    # --- Total residual ---
    F_form = F_phi + F_cont

    # --- Add source terms cleanly ---
    if params.tau_n > 0:
        n_eq_hat = Nd_hat
        tau_hat = params.tau_n * params.mu_n * params.V_t / L_norm**2
        # R̂ = (n̂ - n̂_eq) / τ̂
        # Source Ŝ = Ĝ - R̂ = -R̂ (when G=0)
        # F_cont -= ∫ Ŝ w dx̂ = -∫ (-R̂) w dx̂ = ∫ R̂ w dx̂
        F_form += (n_hat - n_eq_hat) / tau_hat * w * ufl.dx

    if params.G0 > 0:
        # Ĝ = G0 × L_norm² / (n_ref μn Vt)
        G0_hat = params.G0 * L_norm**2 / (params.n_ref * params.mu_n * params.V_t)
        x = ufl.SpatialCoordinate(msh)
        alpha_hat = params.alpha_opt * L_norm
        G_hat = G0_hat * ufl.exp(-alpha_hat * x[0])
        # Ŝ = Ĝ - R̂, so F_cont -= ∫ Ĝ w dx̂ (generation adds carriers)
        F_form -= G_hat * w * ufl.dx

    # Jacobian
    J_form = ufl.derivative(F_form, u_hat)

    # --- Boundary conditions ---
    L_hat = dims["L_hat"]
    V_hat = params.V_bias / params.V_t
    Nd_hat_val = params.Nd / params.n_ref

    # Sub-spaces for φ and n
    W0 = W.sub(0)   # φ̂ sub-space
    W1 = W.sub(1)   # n̂ sub-space
    W0_col, _ = W0.collapse()
    W1_col, _ = W1.collapse()

    tol = 1e-10

    # Left contact: x = 0
    def left_marker(x):
        return np.abs(x[0]) < tol

    left_facets = dmesh.locate_entities_boundary(msh, 0, left_marker)

    # φ(0) = 0
    phi_left = fem.Function(W0_col)
    phi_left.x.array[:] = 0.0
    dofs_phi_left = fem.locate_dofs_topological((W0, W0_col), 0, left_facets)
    bc_phi_left = fem.dirichletbc(phi_left, dofs_phi_left, W0)

    # n(0) = Nd
    n_left = fem.Function(W1_col)
    n_left.x.array[:] = Nd_hat_val
    dofs_n_left = fem.locate_dofs_topological((W1, W1_col), 0, left_facets)
    bc_n_left = fem.dirichletbc(n_left, dofs_n_left, W1)

    # Right contact: x = L
    def right_marker(x):
        return np.abs(x[0] - L_hat) < tol

    right_facets = dmesh.locate_entities_boundary(msh, 0, right_marker)

    # φ(L) = V_bias / Vt
    phi_right = fem.Function(W0_col)
    phi_right.x.array[:] = V_hat
    dofs_phi_right = fem.locate_dofs_topological((W0, W0_col), 0, right_facets)
    bc_phi_right = fem.dirichletbc(phi_right, dofs_phi_right, W0)

    # n(L) = Nd
    n_right = fem.Function(W1_col)
    n_right.x.array[:] = Nd_hat_val
    dofs_n_right = fem.locate_dofs_topological((W1, W1_col), 0, right_facets)
    bc_n_right = fem.dirichletbc(n_right, dofs_n_right, W1)

    bcs = [bc_phi_left, bc_phi_right, bc_n_left, bc_n_right]

    # --- Set initial guess ---
    # φ: linear ramp from 0 to V̂_bias
    # n: uniform Nd
    u_hat.x.array[:] = 0.0  # zero everything first

    # Get sub-function arrays
    phi_sub = u_hat.sub(0)
    n_sub = u_hat.sub(1)

    # Interpolate initial guess for φ (linear ramp)
    phi_init = fem.Function(W0_col)
    phi_init.interpolate(lambda x: V_hat * x[0] / L_hat)
    phi_sub.interpolate(phi_init)

    # Interpolate initial guess for n (uniform Nd)
    n_init = fem.Function(W1_col)
    n_init.x.array[:] = Nd_hat_val
    n_sub.interpolate(n_init)

    u_hat.x.scatter_forward()

    return W, F_form, J_form, bcs, u_hat


# ════════════════════════════════════════════════════════════════════════
#  4.  SOLVER
# ════════════════════════════════════════════════════════════════════════

class SolveError(RuntimeError):
    pass


def solve_dd(F_form, J_form, u_hat, bcs, comm,
             max_newton=50, snes_rtol=1e-8, snes_atol=1e-10,
             verbose=True):
    """
    Solve coupled Poisson-continuity using DOLFINx NonlinearProblem.

    Returns dict with convergence info.
    """
    problem = fem_petsc.NonlinearProblem(
        F_form, u_hat, bcs=bcs, J=J_form,
        petsc_options_prefix="dd_"
    )

    snes = problem._snes
    snes.setType("newtonls")
    snes.setTolerances(rtol=snes_rtol, atol=snes_atol, max_it=max_newton)

    ls = snes.getLineSearch()
    ls.setType("bt")

    ksp = snes.getKSP()
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9, atol=1e-12, max_it=2000)
    pc = ksp.getPC()
    # For mixed systems, use fieldsplit or LU
    pc.setType("lu")

    if verbose:
        snes.setMonitor(
            lambda s, it, fnorm: print(
                f"  SNES iter {it:3d} | ||F|| = {fnorm:.6e}", flush=True
            ) if comm.rank == 0 else None
        )

    snes.setFromOptions()

    if comm.rank == 0:
        rtol_s, atol_s, _, max_it_s = snes.getTolerances()
        print(f"  [SNES config] type={snes.getType()} max_it={max_it_s} "
              f"rtol={rtol_s:.1e} atol={atol_s:.1e} "
              f"ls={ls.getType()}", flush=True)
        print(f"  [KSP  config] type={ksp.getType()} pc={pc.getType()}", flush=True)

    if comm.rank == 0:
        print(f"  Starting SNES solve...", flush=True)

    t0 = time.perf_counter()
    problem.solve()
    u_hat.x.scatter_forward()
    t_solve = time.perf_counter() - t0

    n_iters = snes.getIterationNumber()
    fnorm = snes.getFunctionNorm()
    reason = snes.getConvergedReason()
    converged = reason > 0

    _REASONS = {
        2: "RTOL", 3: "ATOL", 4: "STOL", 5: "ITS",
        -1: "FNORM_NAN", -2: "FUNCTION_COUNT", -3: "LINEAR_SOLVE",
        -4: "FUNCTION_RANGE", -5: "DTOL",
        -6: "DIVERGED_LINE_SEARCH", -7: "DIVERGED_LOCAL_MIN",
        -8: "MAX_IT",
    }
    reason_str = _REASONS.get(reason, f"CODE_{reason}")

    if comm.rank == 0:
        tag = "CONVERGED" if converged else "FAILED"
        print(f"\n=== SNES {tag} ===")
        print(f"  SNES iterations : {n_iters}")
        print(f"  Final ||F||     : {fnorm:.6e}")
        print(f"  Reason          : {reason_str} ({reason})")
        print(f"  Solve time      : {t_solve:.3f} s")
        print(f"====================\n")

    if not converged:
        raise SolveError(
            f"SNES diverged: {n_iters} iters, reason={reason_str}, ||F||={fnorm:.3e}"
        )

    return {"converged": converged, "iterations": n_iters,
            "fnorm": fnorm, "reason": reason_str, "time": t_solve}


# ════════════════════════════════════════════════════════════════════════
#  5.  POST-PROCESSING AND ACCEPTANCE TESTS
# ════════════════════════════════════════════════════════════════════════

def extract_fields_1d(u_hat, W, msh, params, dims, comm):
    """Extract φ, n, Jn from the mixed solution on a 1D mesh.

    Returns dict with 1D arrays (on rank 0) of:
        x_phys: physical x coordinates [m]
        phi_phys: potential [V]
        n_phys: electron density [m⁻³]
        Jn: electron current density evaluated at cell midpoints [A/m²]
    """
    # Collapse sub-spaces and interpolate
    W0_col, dof_map_0 = W.sub(0).collapse()
    W1_col, dof_map_1 = W.sub(1).collapse()

    phi_fn = fem.Function(W0_col, name="phi_hat")
    n_fn = fem.Function(W1_col, name="n_hat")
    phi_fn.x.array[:] = u_hat.x.array[dof_map_0]
    n_fn.x.array[:] = u_hat.x.array[dof_map_1]
    phi_fn.x.scatter_forward()
    n_fn.x.scatter_forward()

    L_norm = dims["L_norm"]
    V_t = params.V_t

    # Get DOF coordinates
    x_dofs = W0_col.tabulate_dof_coordinates()[:, 0]  # 1D: only x
    n_owned = W0_col.dofmap.index_map.size_local

    # Gather on rank 0
    x_local = x_dofs[:n_owned] * L_norm  # physical coordinates
    phi_local = phi_fn.x.array[:n_owned] * V_t  # physical potential
    n_local = n_fn.x.array[:n_owned] * params.n_ref  # physical density

    x_all = comm.gather(x_local, root=0)
    phi_all = comm.gather(phi_local, root=0)
    n_all = comm.gather(n_local, root=0)

    result = {}
    if comm.rank == 0:
        x_phys = np.concatenate(x_all)
        phi_phys = np.concatenate(phi_all)
        n_phys = np.concatenate(n_all)

        # Sort by x
        idx = np.argsort(x_phys)
        x_phys = x_phys[idx]
        phi_phys = phi_phys[idx]
        n_phys = n_phys[idx]

        # Compute Jn at midpoints using finite differences
        # Jn = -q μn n dφ/dx + q Dn dn/dx
        dx = np.diff(x_phys)
        dphi = np.diff(phi_phys)
        dn = np.diff(n_phys)
        n_mid = 0.5 * (n_phys[:-1] + n_phys[1:])
        x_mid = 0.5 * (x_phys[:-1] + x_phys[1:])

        E_mid = -dphi / dx  # electric field [V/m]
        Jn_drift = Q_E * params.mu_n * n_mid * E_mid
        Jn_diff = Q_E * params.D_n * dn / dx
        Jn = Jn_drift + Jn_diff

        result = {
            "x_phys": x_phys, "phi_phys": phi_phys, "n_phys": n_phys,
            "x_mid": x_mid, "Jn": Jn, "E_mid": E_mid,
            "Jn_drift": Jn_drift, "Jn_diff": Jn_diff,
        }

    return result


def run_acceptance_tests(result, params, dims, comm):
    """
    Run Stage 1 acceptance tests.

    Tests (on rank 0 only):
    1. Jn must be spatially constant (relative variation < 1e-3)
    2. Compare Jn to analytic: J = q μn Nd (V_bias / L) for uniform Nd, G=R=0
    3. Verify n ≈ Nd everywhere (charge neutrality for small perturbation)
    """
    if comm.rank != 0:
        return True

    if not result:
        print("  No result to test (not rank 0?)")
        return False

    all_pass = True
    Jn = result["Jn"]
    n_phys = result["n_phys"]

    print("=== ACCEPTANCE TESTS (Stage 1) ===")

    # Test 1: Jn spatial constancy
    Jn_mean = np.mean(Jn)
    Jn_var = np.max(np.abs(Jn - Jn_mean)) / max(abs(Jn_mean), 1e-30)
    t1_pass = Jn_var < 1e-3
    print(f"  1. Jn spatial constancy: var={Jn_var:.6e} "
          f"({'PASS' if t1_pass else 'FAIL'} < 1e-3)")
    if not t1_pass:
        all_pass = False

    # Test 2: Jn vs analytic (ohmic IV)
    L_phys = params.L_ref  # 1D device length
    J_analytic = Q_E * params.mu_n * params.Nd * params.V_bias / L_phys
    J_rel_err = abs(Jn_mean - J_analytic) / max(abs(J_analytic), 1e-30)
    t2_pass = J_rel_err < 0.05
    print(f"  2. Jn vs analytic: J_num={Jn_mean:.6e} A/m², "
          f"J_ana={J_analytic:.6e} A/m², rel_err={J_rel_err:.6e} "
          f"({'PASS' if t2_pass else 'FAIL'} < 5%)")
    if not t2_pass:
        all_pass = False

    # Test 3: n ≈ Nd (quasi-neutrality)
    n_err = np.max(np.abs(n_phys - params.Nd)) / params.Nd
    t3_pass = n_err < 0.01
    print(f"  3. Charge neutrality: max|n-Nd|/Nd={n_err:.6e} "
          f"({'PASS' if t3_pass else 'FAIL'} < 1%)")
    if not t3_pass:
        all_pass = False

    # Test 4: Zero current when V_bias = 0
    if abs(params.V_bias) < 1e-15:
        Jn_max = np.max(np.abs(Jn))
        t4_pass = Jn_max < 1e-10
        print(f"  4. Zero current at V=0: max|Jn|={Jn_max:.6e} "
              f"({'PASS' if t4_pass else 'FAIL'})")
        if not t4_pass:
            all_pass = False

    tag = "ALL PASSED" if all_pass else "SOME FAILED"
    print(f"  Result: {tag}")
    print("==================================\n")

    return all_pass


# ════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ════════════════════════════════════════════════════════════════════════

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank

    parser = argparse.ArgumentParser(description="MCT Photoconductor Drift-Diffusion")
    parser.add_argument("--Nd", type=float, default=1e22, help="Donor conc [m⁻³]")
    parser.add_argument("--Na", type=float, default=0.0, help="Acceptor conc [m⁻³]")
    parser.add_argument("--V_bias", type=float, default=0.1, help="Bias voltage [V]")
    parser.add_argument("--mu_n", type=float, default=1.0, help="Mobility [m²/Vs]")
    parser.add_argument("--tau_n", type=float, default=0.0, help="Lifetime [s]")
    parser.add_argument("--G0", type=float, default=0.0, help="Generation rate [m⁻³s⁻¹]")
    parser.add_argument("--alpha_opt", type=float, default=0.0, help="Absorption coeff [m⁻¹]")
    parser.add_argument("--n_cells", type=int, default=200, help="Number of 1D cells")
    parser.add_argument("--L", type=float, default=10e-6, help="Device length [m]")
    args = parser.parse_args()

    params = DeviceParams(
        Nd=args.Nd, Na=args.Na, V_bias=args.V_bias,
        mu_n=args.mu_n, tau_n=args.tau_n,
        G0=args.G0, alpha_opt=args.alpha_opt,
    )
    geom = DeviceGeometry(L=args.L)
    L_norm = geom.L  # For 1D, normalize by device length

    if rank == 0:
        print("\n" + "=" * 70)
        print("DRIFT-DIFFUSION: Stage 1 — Biased Resistor (G=R=0)")
        if params.tau_n > 0:
            print("  + Stage 2: Recombination (τ_n > 0)")
        if params.G0 > 0:
            print("  + Stage 3: Optical Generation (G0 > 0)")
        print("=" * 70)
        print(params.summary(L_norm))

    # Create mesh
    msh, dims = create_1d_mesh(geom, L_norm, args.n_cells, comm)

    if rank == 0:
        n_dofs = msh.topology.index_map(0).size_global
        print(f"  Mesh: {args.n_cells} cells, {n_dofs} vertices")

    # Build system
    W, F_form, J_form, bcs, u_hat = build_dd_system(msh, params, dims, comm)

    if rank == 0:
        print(f"  DOFs: {W.dofmap.index_map.size_global} "
              f"(φ: {W.sub(0).collapse()[0].dofmap.index_map.size_global}, "
              f"n: {W.sub(1).collapse()[0].dofmap.index_map.size_global})")

    # Solve
    info = solve_dd(F_form, J_form, u_hat, bcs, comm)

    # Post-process
    result = extract_fields_1d(u_hat, W, msh, params, dims, comm)

    if rank == 0 and result:
        phi = result["phi_phys"]
        n = result["n_phys"]
        Jn = result["Jn"]
        print(f"  Solution ranges:")
        print(f"    φ  : [{phi.min():.6f}, {phi.max():.6f}] V")
        print(f"    n  : [{n.min():.6e}, {n.max():.6e}] m⁻³")
        print(f"    Jn : [{Jn.min():.6e}, {Jn.max():.6e}] A/m²")
        print(f"    Jn_mean = {np.mean(Jn):.6e} A/m²")

    # Acceptance tests
    run_acceptance_tests(result, params, dims, comm)


if __name__ == "__main__":
    main()
