"""
===========================================================================
MCT (HgCdTe) Photoconductor 3D Poisson Solver — Bricks 1-3
===========================================================================

Single-file, MPI-safe, production-grade solver for DOLFINx 0.10.x.

Physics:
    Dimensionless Poisson:  nabla^2 phi_hat = -alpha_eff * rho_hat
    Brick 1: rho_hat = 0 (Laplace)
    Brick 2: rho_hat = (Nd - Na)/n_ref (fixed doping, no mobile carriers)
    Brick 3: rho_hat = (Nd-Na)/n_ref + (n_i/n_ref)(p_hat - n_hat)
             with Boltzmann n_hat=exp(+(phi-phi_ref)), p_hat=exp(-(phi-phi_ref))
             Solved via PETSc SNES Newton.

Geometry:
    Rectangular prism photoconductor (structured hexahedral mesh)
    Thickness (x): 10 um  |  Length (y): 3000 um  |  Width (z): 1000 um

Boundary conditions:
    Left contact:   x = 0,      y <= 1000 um  -> phi = 0 V
    Right contact:  x = Lx,     y >= 2000 um  -> phi = V_bias
    All other boundaries: dphi/dn = 0 (insulating Neumann)

Solver:
    Linear (Bricks 1-2): PETSc KSP CG + HYPRE BoomerAMG
    Nonlinear (Brick 3): PETSc SNES Newton + CG/AMG inner solve

Usage:
    # Brick 1: Laplace
    mpirun -n 8 python3 photoconductor_poisson.py
    mpirun -n 8 python3 photoconductor_poisson.py --V_bias 0.5

    # MMS validation (MUST PASS before any new physics)
    mpirun -n 8 python3 photoconductor_poisson.py --test_mms

    # Bias sweep (Brick 1, linearity check)
    mpirun -n 8 python3 photoconductor_poisson.py --sweep_bias --V0 0 --V1 2 --nV 11

    # Brick 2: Fixed doping
    mpirun -n 8 python3 photoconductor_poisson.py --doping_profile uniform --Nd 1e22
    mpirun -n 8 python3 photoconductor_poisson.py --doping_profile uniform --Nd 1e22 --Na 5e21
    mpirun -n 8 python3 photoconductor_poisson.py --doping_profile step_y --Nd 1e22 --y_step 1500e-6

    # Doping sanity sweep (V_bias=0, Nd/n_ref sweep)
    mpirun -n 8 python3 photoconductor_poisson.py --doping_sanity

    # Brick 3: Semi-linear Poisson (Boltzmann carriers)
    mpirun -n 8 python3 photoconductor_poisson.py --nonlinear --Nd 1e22 --V_bias 0.1
    mpirun -n 8 python3 photoconductor_poisson.py --nl_sweep --Nd 1e22 --V0 0 --V1 2 --nV 11

    # PETSc pass-through
    mpirun -n 8 python3 photoconductor_poisson.py -pc_type gamg
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.mesh as dmesh
import dolfinx.io
import dolfinx.nls.petsc
import ufl

# ════════════════════════════════════════════════════════════════════════
#  1.  SCALING  — nondimensionalization
# ════════════════════════════════════════════════════════════════════════
#
#  φ̂ = φ / V_t          V_t = kT/q
#  x̂ = x / L_norm       (coordinates normalized by L_max for conditioning)
#  ρ̂ = ρ / (q n_ref)
#
#  Dimensionless Poisson:  ∇̂²φ̂ = −α_eff ρ̂
#  where  α_eff = α * (L_ref / L_norm)²
#         α = q n_ref L_ref² / (ε₀ εᵣ V_t)

Q_E   = 1.602176634e-19       # C
K_B   = 1.380649e-23          # J/K
EPS_0 = 8.8541878128e-12      # F/m


@dataclass(frozen=True)
class DeviceScaling:
    """Immutable nondimensionalization container."""
    temperature: float    # K
    eps_r: float          # MCT static dielectric constant
    n_ref: float          # reference carrier density  [m⁻³]
    L_ref: float          # reference length  [m]

    @property
    def V_t(self) -> float:
        return K_B * self.temperature / Q_E

    @property
    def alpha(self) -> float:
        return (Q_E * self.n_ref * self.L_ref**2) / (EPS_0 * self.eps_r * self.V_t)

    def phi_to_dimless(self, phi_V: float) -> float:
        return phi_V / self.V_t

    def phi_to_physical(self, phi_hat: float) -> float:
        return phi_hat * self.V_t

    def summary(self) -> str:
        return (
            "=== Nondimensionalization ===\n"
            f"  T       = {self.temperature:.1f} K\n"
            f"  V_t     = {self.V_t*1e3:.4f} mV\n"
            f"  eps_r   = {self.eps_r:.2f}\n"
            f"  n_ref   = {self.n_ref:.3e} m⁻³\n"
            f"  L_ref   = {self.L_ref*1e6:.1f} µm\n"
            f"  alpha   = {self.alpha:.6e}\n"
            "============================="
        )


def make_mct_scaling_77K() -> DeviceScaling:
    """MCT (x≈0.2–0.3 MWIR) at 77 K.  εᵣ≈16.7, L_ref=thickness=10µm."""
    return DeviceScaling(temperature=77.0, eps_r=16.7, n_ref=1e21, L_ref=10e-6)


# ════════════════════════════════════════════════════════════════════════
#  2.  MESH  — structured hexahedral, dimensionless coordinates
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeviceGeometry:
    Lx: float = 10e-6       # thickness  [m]
    Ly: float = 3000e-6     # length     [m]
    Lz: float = 1000e-6     # width      [m]

    @property
    def L_max(self) -> float:
        """Maximum dimension for coordinate normalization."""
        return max(self.Lx, self.Ly, self.Lz)


@dataclass(frozen=True)
class MeshConfig:
    nx: int = 20
    ny: int = 60
    nz: int = 40


def create_device_mesh(
    geom: DeviceGeometry,
    L_ref: float,
    cfg: MeshConfig,
    comm: MPI.Comm,
):
    """Return (mesh, dims_dict) in dimensionless coordinates.
    
    Coordinates normalized by L_max (largest dimension) for better conditioning.
    Physics scaling still uses L_ref.
    """
    L_norm = geom.L_max
    Lx_hat = geom.Lx / L_norm
    Ly_hat = geom.Ly / L_norm
    Lz_hat = geom.Lz / L_norm

    msh = dmesh.create_box(
        comm,
        points=[np.array([0.0, 0.0, 0.0]),
                np.array([Lx_hat, Ly_hat, Lz_hat])],
        n=[cfg.nx, cfg.ny, cfg.nz],
        cell_type=dmesh.CellType.hexahedron,
    )
    dims = {"Lx_hat": Lx_hat, "Ly_hat": Ly_hat, "Lz_hat": Lz_hat, 
            "L_norm": L_norm, "L_ref": L_ref}
    return msh, dims


# ════════════════════════════════════════════════════════════════════════
#  3.  POISSON  — weak form, BCs, assembly
# ════════════════════════════════════════════════════════════════════════

def _locate_contact_dofs(V, msh, x_val, y_lo, y_hi, tol=1e-8):
    """DOFs on partial face: |x - x_val| < tol  AND  y_lo ≤ y ≤ y_hi."""
    def marker(x):
        return (np.abs(x[0] - x_val) < tol) & (x[1] >= y_lo - tol) & (x[1] <= y_hi + tol)
    tdim = msh.topology.dim
    facets = dmesh.locate_entities_boundary(msh, tdim - 1, marker)
    return fem.locate_dofs_topological(V, tdim - 1, facets)


def build_poisson_system(msh, scaling, dims, V_bias, rho_hat_expr=None):
    """
    Assemble dimensionless Poisson:  ∫ ∇̂φ̂·∇̂v dx̂ = α_eff ∫ ρ̂ v dx̂

    Coordinates normalized by L_norm, physics scaled by L_ref.
    α_eff = α * (L_ref / L_norm)²

    Returns (V, A, b, bcs, phi_hat).
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi_hat = fem.Function(V, name="phi_hat")
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # --- LHS ---
    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # --- RHS ---
    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2
    
    if rho_hat_expr is not None:
        rho_fn = fem.Function(V, name="rho_hat")
        rho_fn.interpolate(rho_hat_expr)
        L_form = fem.Constant(msh, PETSc.ScalarType(alpha_eff)) * rho_fn * v * ufl.dx
    else:
        L_form = fem.Constant(msh, PETSc.ScalarType(0.0)) * v * ufl.dx

    # --- BCs ---
    Lx_hat = dims["Lx_hat"]
    Ly_hat = dims["Ly_hat"]
    y_left_max  = 1000e-6 / L_norm   # dimensionless
    y_right_min = 2000e-6 / L_norm

    dofs_L = _locate_contact_dofs(V, msh, 0.0,    0.0,         y_left_max)
    dofs_R = _locate_contact_dofs(V, msh, Lx_hat, y_right_min, Ly_hat)

    bc_L = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_L, V)
    bc_R = fem.dirichletbc(PETSc.ScalarType(scaling.phi_to_dimless(V_bias)), dofs_R, V)
    bcs = [bc_L, bc_R]

    # --- Compile & assemble ---
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    return V, A, b, bcs, phi_hat


# ════════════════════════════════════════════════════════════════════════
#  4.  MMS VALIDATION  — manufactured solution test
# ════════════════════════════════════════════════════════════════════════

def mms_exact_solution(x):
    """Manufactured exact solution: φ̂_exact = x̂²"""
    return x[0]**2


def mms_rhs_charge(alpha_eff):
    """For φ̂ = x̂², ∇̂²φ̂ = 2, so ρ̂ = -2/α_eff"""
    return lambda x: np.full(x.shape[1], -2.0 / alpha_eff)


def build_mms_system(msh, scaling, dims):
    """
    Assemble MMS test system with full Dirichlet BCs.
    
    Exact solution: φ̂ = x̂²
    Strong form: ∇̂²φ̂ = 2 = -α_eff * ρ̂  →  ρ̂ = -2/α_eff
    
    Returns (V, A, b, bcs, phi_hat, phi_exact).
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi_hat = fem.Function(V, name="phi_hat")
    phi_exact = fem.Function(V, name="phi_exact")
    phi_exact.interpolate(mms_exact_solution)
    
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # --- LHS ---
    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

    # --- RHS ---
    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2
    
    rho_fn = fem.Function(V, name="rho_hat")
    rho_fn.interpolate(mms_rhs_charge(alpha_eff))
    L_form = fem.Constant(msh, PETSc.ScalarType(alpha_eff)) * rho_fn * v * ufl.dx

    # --- Full Dirichlet BCs on all boundaries ---
    tdim = msh.topology.dim
    fdim = tdim - 1
    
    # Locate all boundary facets
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    
    # Apply exact solution on boundary
    bc = fem.dirichletbc(phi_exact, boundary_dofs)
    bcs = [bc]

    # --- Compile & assemble ---
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], bcs=[bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs)

    return V, A, b, bcs, phi_hat, phi_exact


def compute_errors(phi_h, phi_exact, comm):
    """
    Compute L2 and H1 seminorm errors.
    
    Returns (L2_error, H1_seminorm_error).
    """
    # Error function
    e_h = phi_h - phi_exact
    
    # L2 error: ||e||_L2 = sqrt(∫ e² dx)
    L2_form = fem.form(e_h**2 * ufl.dx)
    L2_local = fem.assemble_scalar(L2_form)
    L2_global = comm.allreduce(L2_local, op=MPI.SUM)
    L2_error = np.sqrt(L2_global)
    
    # H1 seminorm: ||∇e||_L2 = sqrt(∫ |∇e|² dx)
    H1_form = fem.form(ufl.inner(ufl.grad(e_h), ufl.grad(e_h)) * ufl.dx)
    H1_local = fem.assemble_scalar(H1_form)
    H1_global = comm.allreduce(H1_local, op=MPI.SUM)
    H1_seminorm = np.sqrt(H1_global)
    
    return L2_error, H1_seminorm


# ════════════════════════════════════════════════════════════════════════
#  5.  SOLVER  — PETSc KSP harness
# ════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SolverConfig:
    ksp_type: str   = "cg"
    pc_type: str    = "hypre"
    pc_hypre_type: str = "boomeramg"
    rtol: float     = 1e-10
    atol: float     = 1e-14
    max_iter: int   = 2000


class PoissonSolveError(RuntimeError):
    pass


@dataclass
class SolveResult:
    converged: bool
    iterations: int
    residual_norm: float
    reason: int
    reason_str: str


def solve_poisson(A, b, phi, cfg=SolverConfig(), comm=MPI.COMM_WORLD) -> SolveResult:
    """CG + AMG solve with convergence check.  Raises on divergence."""
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType(cfg.ksp_type)
    ksp.setTolerances(rtol=cfg.rtol, atol=cfg.atol, max_it=cfg.max_iter)

    pc = ksp.getPC()
    pc.setType(cfg.pc_type)
    if cfg.pc_type == "hypre":
        pc.setHYPREType(cfg.pc_hypre_type)

    ksp.setFromOptions()
    ksp.solve(b, phi.x.petsc_vec)
    phi.x.scatter_forward()

    its    = ksp.getIterationNumber()
    rnorm  = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    converged = reason > 0

    # Compute true residual ||A*x - b||
    r = b.duplicate()
    A.mult(phi.x.petsc_vec, r)
    r.axpy(-1.0, b)  # r = A*x - b
    true_rnorm = r.norm()
    b_norm = b.norm()
    r.destroy()

    _REASONS = {
        2: "RTOL", 3: "ATOL", 9: "ATOL_NORMAL",
        -3: "DIVERGED_ITS", -4: "DIVERGED_DTOL",
        -5: "DIVERGED_BREAKDOWN", -9: "DIVERGED_NANORINF",
    }
    reason_str = _REASONS.get(reason, f"CODE_{reason}")

    result = SolveResult(converged, its, rnorm, reason, reason_str)

    if comm.rank == 0:
        tag = "CONVERGED" if converged else "FAILED"
        print(f"\n=== KSP {tag} ===")
        print(f"  Iterations      : {its}")
        print(f"  KSP residual    : {rnorm:.6e}")
        print(f"  True ||Ax-b||   : {true_rnorm:.6e}")
        print(f"  ||b||           : {b_norm:.6e}")
        print(f"  Relative true   : {true_rnorm/max(b_norm, 1e-30):.6e}")
        print(f"  Reason          : {reason_str}")
        print(f"====================\n")

    ksp.destroy()

    if not converged:
        raise PoissonSolveError(
            f"Poisson solve diverged: {its} iters, reason={reason_str}, rnorm={rnorm:.3e}"
        )
    return result


# ════════════════════════════════════════════════════════════════════════
#  6.  MMS CONVERGENCE TEST
# ════════════════════════════════════════════════════════════════════════

def run_mms_convergence_test(base_nx=5, base_ny=15, base_nz=10, n_levels=3):
    """
    Run MMS convergence test with mesh refinement.
    
    Expected convergence rates for P1 elements:
    - L2 error: O(h²) → rate ≈ 2.0
    - H1 seminorm: O(h) → rate ≈ 1.0
    
    Exits with code 1 if convergence rates are outside tolerance.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    
    if rank == 0:
        print("\n" + "="*70)
        print("MMS CONVERGENCE TEST: φ̂_exact = x̂²")
        print("="*70)
        print(scaling.summary())
        print()
    
    results = []
    
    for level in range(n_levels):
        factor = 2**level
        nx = base_nx * factor
        ny = base_ny * factor
        nz = base_nz * factor
        
        mcfg = MeshConfig(nx=nx, ny=ny, nz=nz)
        
        # Create mesh
        msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)
        
        # Compute mesh size h (characteristic element size) - use max for anisotropic domains
        hx = dims["Lx_hat"] / nx
        hy = dims["Ly_hat"] / ny
        hz = dims["Lz_hat"] / nz
        h = max(hx, hy, hz)  # in normalized coordinates
        
        # Build MMS system
        V, A, b, bcs, phi_hat, phi_exact = build_mms_system(msh, scaling, dims)
        
        ndofs = V.dofmap.index_map.size_global
        
        # Solve
        t0 = time.perf_counter()
        result = solve_poisson(A, b, phi_hat, SolverConfig(), comm)
        t_solve = time.perf_counter() - t0
        
        # Compute errors
        L2_err, H1_err = compute_errors(phi_hat, phi_exact, comm)
        
        results.append({
            'level': level,
            'nx': nx, 'ny': ny, 'nz': nz,
            'h': h,
            'ndofs': ndofs,
            'L2': L2_err,
            'H1': H1_err,
            'iters': result.iterations,
            'time': t_solve
        })
        
        if rank == 0:
            print(f"Level {level}: {nx}×{ny}×{nz} mesh, h={h:.6f}, DOFs={ndofs}")
            print(f"  L2 error    : {L2_err:.6e}")
            print(f"  H1 seminorm : {H1_err:.6e}")
            print(f"  Iterations  : {result.iterations}")
            print(f"  Solve time  : {t_solve:.3f} s")
            print()
    
    # Compute convergence rates
    if rank == 0:
        print("\n" + "="*70)
        print("CONVERGENCE RATES")
        print("="*70)
        print(f"{'Level':<8} {'h':<12} {'L2 error':<14} {'L2 rate':<10} {'H1 error':<14} {'H1 rate':<10}")
        print("-"*70)
        
        for i, r in enumerate(results):
            if i == 0:
                print(f"{r['level']:<8} {r['h']:<12.6e} {r['L2']:<14.6e} {'---':<10} {r['H1']:<14.6e} {'---':<10}")
            else:
                r_prev = results[i-1]
                h_ratio = r_prev['h'] / r['h']
                L2_rate = np.log(r_prev['L2'] / r['L2']) / np.log(h_ratio)
                H1_rate = np.log(r_prev['H1'] / r['H1']) / np.log(h_ratio)
                print(f"{r['level']:<8} {r['h']:<12.6e} {r['L2']:<14.6e} {L2_rate:<10.4f} {r['H1']:<14.6e} {H1_rate:<10.4f}")
        
        print("="*70)
        
        # Check final convergence rate (average of last 2 refinements)
        if n_levels >= 3:
            L2_rates = []
            H1_rates = []
            for i in range(1, n_levels):
                r_prev = results[i-1]
                r = results[i]
                h_ratio = r_prev['h'] / r['h']
                L2_rate = np.log(r_prev['L2'] / r['L2']) / np.log(h_ratio)
                H1_rate = np.log(r_prev['H1'] / r['H1']) / np.log(h_ratio)
                L2_rates.append(L2_rate)
                H1_rates.append(H1_rate)
            
            avg_L2_rate = np.mean(L2_rates[-2:])
            avg_H1_rate = np.mean(H1_rates[-2:])
            
            print(f"\nAverage convergence rates (last 2 refinements):")
            print(f"  L2 rate: {avg_L2_rate:.4f}  (expected ≈ 2.0 for P1)")
            print(f"  H1 rate: {avg_H1_rate:.4f}  (expected ≈ 1.0 for P1)")
            print()
            
            # Tolerance check
            L2_ok = 1.8 <= avg_L2_rate <= 2.5
            H1_ok = 0.8 <= avg_H1_rate <= 1.5
            
            if L2_ok and H1_ok:
                print("✓ MMS TEST PASSED: Convergence rates within tolerance")
                return 0
            else:
                print("✗ MMS TEST FAILED: Convergence rates outside tolerance")
                if not L2_ok:
                    print(f"  L2 rate {avg_L2_rate:.4f} not in [1.8, 2.5]")
                if not H1_ok:
                    print(f"  H1 rate {avg_H1_rate:.4f} not in [0.8, 1.5]")
                return 1
        else:
            print("Insufficient refinement levels for rate validation")
            return 0
    
    return 0


# ════════════════════════════════════════════════════════════════════════
#  7.  BIAS SWEEP VALIDATION
# ════════════════════════════════════════════════════════════════════════

def run_bias_sweep(V0, V1, nV, nx=20, ny=60, nz=40):
    """
    Sweep bias voltage and verify linear response.
    
    For Laplace (ρ=0), φ_max should scale exactly as V_bias/V_t.
    KSP iterations should remain stable.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=nx, ny=ny, nz=nz)
    
    if rank == 0:
        print("\n" + "="*70)
        print(f"BIAS SWEEP: {V0} V → {V1} V ({nV} steps)")
        print("="*70)
        print(scaling.summary())
        print()
    
    # Create mesh once
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)
    
    V_array = np.linspace(V0, V1, nV)
    results = []
    
    for i, V_bias in enumerate(V_array):
        # Build system
        V, A, b, bcs, phi_hat = build_poisson_system(msh, scaling, dims, V_bias)
        
        # Solve
        t0 = time.perf_counter()
        result = solve_poisson(A, b, phi_hat, SolverConfig(), comm)
        t_solve = time.perf_counter() - t0
        
        # Measure solution
        phi_min = comm.allreduce(phi_hat.x.array.min(), op=MPI.MIN)
        phi_max = comm.allreduce(phi_hat.x.array.max(), op=MPI.MAX)
        phi_expected = scaling.phi_to_dimless(V_bias)
        
        deviation = abs(phi_max - phi_expected) / max(abs(phi_expected), 1e-10)
        
        results.append({
            'V': V_bias,
            'phi_hat_max': phi_max,
            'phi_hat_expected': phi_expected,
            'deviation': deviation,
            'iters': result.iterations,
            'time': t_solve
        })
        
        if rank == 0:
            print(f"V = {V_bias:6.3f} V | φ̂_max = {phi_max:10.6f} (expect {phi_expected:10.6f}) | "
                  f"dev = {deviation*100:6.4f}% | its = {result.iterations:4d}")
    
    if rank == 0:
        print("\n" + "="*70)
        print("BIAS SWEEP SUMMARY")
        print("="*70)
        
        max_deviation = max(r['deviation'] for r in results)
        iter_range = (min(r['iters'] for r in results), max(r['iters'] for r in results))
        
        print(f"Max deviation from linearity: {max_deviation*100:.6f}%")
        print(f"Iteration count range: {iter_range[0]} - {iter_range[1]}")
        print()
        
        if max_deviation < 1e-3 and iter_range[1] < 300:
            print("✓ BIAS SWEEP PASSED: Linear response, stable iterations")
            return 0
        else:
            print("✗ BIAS SWEEP FAILED")
            if max_deviation >= 1e-3:
                print(f"  Deviation {max_deviation*100:.4f}% exceeds 0.1%")
            if iter_range[1] >= 300:
                print(f"  Iteration count {iter_range[1]} too high")
            return 1
    
    return 0


# ════════════════════════════════════════════════════════════════════════
#  8.  FIXED DOPING PROFILES
# ════════════════════════════════════════════════════════════════════════

def uniform_doping_profile(Nd, Na, n_ref):
    """Uniform net doping: ρ̂ = (Nd − Na) / n_ref."""
    rho_hat_val = (Nd - Na) / n_ref
    return lambda x: np.full(x.shape[1], rho_hat_val)


def step_y_doping_profile(Nd, Na, n_ref, y_step, L_norm):
    """
    Step doping in y-direction:
        y < y_step:  ρ̂ = Nd / n_ref    (donor-rich region)
        y ≥ y_step:  ρ̂ = −Na / n_ref   (acceptor-rich region)
    Net charge: q(Nd − Na), split spatially.
    """
    y_step_hat = y_step / L_norm
    nd_hat = Nd / n_ref
    na_hat = Na / n_ref
    def profile(x):
        return np.where(x[1] < y_step_hat, nd_hat, -na_hat)
    return profile


def run_doping_sanity(args, comm, rank):
    """
    Doping sanity suite: V_bias=0, sweep Nd/n_ref in {0, 1e-3, 1e-2, 1e-1}.
    Prints KSP iterations, residual, phi_hat min/max for each.
    """
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    Nd_ratios = [0.0, 1e-3, 1e-2, 1e-1]

    if rank == 0:
        print("\n" + "="*70)
        print("DOPING SANITY SUITE: V_bias = 0 V, uniform doping sweep")
        print("="*70)
        print(f"{'Nd/n_ref':>12s}  {'KSP its':>8s}  {'rel resid':>12s}  "
              f"{'φ̂_min':>12s}  {'φ̂_max':>12s}")
        print("-" * 70)

    for ratio in Nd_ratios:
        Nd = ratio * scaling.n_ref
        rho_expr = uniform_doping_profile(Nd, 0.0, scaling.n_ref)
        V, A, b, bcs, phi_hat = build_poisson_system(
            msh, scaling, dims, V_bias=0.0, rho_hat_expr=rho_expr
        )
        result = solve_poisson(A, b, phi_hat, SolverConfig(), comm)

        phi_min = comm.allreduce(phi_hat.x.array.min(), op=MPI.MIN)
        phi_max = comm.allreduce(phi_hat.x.array.max(), op=MPI.MAX)

        # True residual
        r = b.duplicate()
        A.mult(phi_hat.x.petsc_vec, r)
        r.axpy(-1.0, b)
        true_rnorm = r.norm()
        b_norm = b.norm()
        r.destroy()
        rel_resid = true_rnorm / max(b_norm, 1e-30)

        if rank == 0:
            print(f"{ratio:>12.1e}  {result.iterations:>8d}  {rel_resid:>12.4e}  "
                  f"{phi_min:>12.6e}  {phi_max:>12.6e}")

    if rank == 0:
        print("="*70)
        print("Done. Inspect for blow-ups.\n")


# ════════════════════════════════════════════════════════════════════════
#  10.  BRICK 3: SEMI-LINEAR POISSON (Boltzmann carrier statistics)
# ════════════════════════════════════════════════════════════════════════
#
#  Nonlinear Poisson with exponential carrier coupling:
#
#    ∇̂²φ̂ = −α_eff [ (Nd − Na)/n_ref + (n_i/n_ref)(p̂ − n̂) ]
#
#  Boltzmann statistics (non-degenerate limit):
#    n̂(φ̂) = exp(+(φ̂ − φ̂_ref))
#    p̂(φ̂) = exp(−(φ̂ − φ̂_ref))
#
#  φ̂_ref is a reference potential (mean of BC values) to prevent exp overflow.
#  The residual form (F=0) is:
#    F = ∫ ∇̂φ̂·∇̂v dx̂ + α_eff ∫ [ ρ̂_doping + (n_i/n_ref)(p̂ − n̂) ] v dx̂ = 0
#
#  Solved with PETSc SNES (Newton + line search).

# MCT intrinsic carrier density at 77 K (MWIR, x~0.22)
# n_i ≈ 2e13 cm⁻³ = 2e19 m⁻³  (strongly temperature/composition dependent)
N_I_DEFAULT = 2e19  # m⁻³

# Overflow guard: clamp exp argument to [-EXP_CLIP, +EXP_CLIP]
EXP_CLIP = 50.0  # exp(50) ≈ 5e21, safe in float64


def _safe_exp(arg, clip=EXP_CLIP):
    """UFL-compatible clamped exponential to prevent overflow."""
    clamped = ufl.max_value(ufl.min_value(arg, clip), -clip)
    return ufl.exp(clamped)


def build_nonlinear_poisson(msh, scaling, dims, V_bias, Nd, Na, n_i,
                            phi_ref_hat=None):
    """
    Build the semi-linear Poisson residual F(φ̂; v) = 0.

    Uses Boltzmann statistics with overflow-protected exponentials.
    Returns (V, F_form, J_form, bcs, phi_hat, phi_ref_hat_value).
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi_hat = fem.Function(V, name="phi_hat")
    v = ufl.TestFunction(V)

    # --- Scaling ---
    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2

    # Reference potential: midpoint of BCs to center the exponentials
    if phi_ref_hat is None:
        phi_ref_hat_value = scaling.phi_to_dimless(V_bias) / 2.0
    else:
        phi_ref_hat_value = phi_ref_hat

    phi_ref = fem.Constant(msh, PETSc.ScalarType(phi_ref_hat_value))
    alpha_c = fem.Constant(msh, PETSc.ScalarType(alpha_eff))
    ni_ratio = fem.Constant(msh, PETSc.ScalarType(n_i / scaling.n_ref))
    rho_dop = fem.Constant(msh, PETSc.ScalarType((Nd - Na) / scaling.n_ref))

    # Boltzmann carriers (overflow-protected)
    n_hat = _safe_exp(+(phi_hat - phi_ref))    # electrons
    p_hat = _safe_exp(-(phi_hat - phi_ref))    # holes

    # Total dimensionless charge: ρ̂_total = ρ̂_doping + (n_i/n_ref)(p̂ − n̂)
    rho_total = rho_dop + ni_ratio * (p_hat - n_hat)

    # Residual: F(φ̂; v) = ∫ ∇φ̂·∇v dx̂ + α_eff ∫ ρ̂_total v dx̂
    # (Note: from −∇²φ̂ = −α_eff ρ̂, multiply by −v, integrate by parts →
    #  ∫ ∇φ̂·∇v dx̂ + α_eff ∫ ρ̂ v dx̂ = 0)
    F_form = (ufl.inner(ufl.grad(phi_hat), ufl.grad(v)) * ufl.dx
              + alpha_c * rho_total * v * ufl.dx)

    # Jacobian (automatic differentiation)
    J_form = ufl.derivative(F_form, phi_hat)

    # --- BCs (same as linear case) ---
    Lx_hat = dims["Lx_hat"]
    Ly_hat = dims["Ly_hat"]
    y_left_max = 1000e-6 / L_norm
    y_right_min = 2000e-6 / L_norm

    dofs_L = _locate_contact_dofs(V, msh, 0.0, 0.0, y_left_max)
    dofs_R = _locate_contact_dofs(V, msh, Lx_hat, y_right_min, Ly_hat)

    bc_L = fem.dirichletbc(PETSc.ScalarType(0.0), dofs_L, V)
    bc_R = fem.dirichletbc(PETSc.ScalarType(scaling.phi_to_dimless(V_bias)),
                           dofs_R, V)
    bcs = [bc_L, bc_R]

    # Initial guess: linear interpolation between BCs (much better than zero)
    phi_bias_hat = scaling.phi_to_dimless(V_bias)
    x_coord = ufl.SpatialCoordinate(msh)
    # Can't easily interpolate SpatialCoordinate, so do it via expression
    def linear_guess(x):
        # Linear ramp in x from 0 to phi_bias_hat
        return phi_bias_hat * (x[0] / Lx_hat)
    phi_hat.interpolate(linear_guess)

    return V, F_form, J_form, bcs, phi_hat, phi_ref_hat_value


def solve_nonlinear_poisson(F_form, J_form, phi_hat, bcs, comm,
                            max_newton=50, snes_rtol=1e-8, snes_atol=1e-10,
                            verbose=True):
    """
    Solve F(phi_hat) = 0 using DOLFINx 0.10 NewtonSolver (PETSc SNES).

    Returns dict with convergence info.
    """
    # DOLFINx 0.10 API: NonlinearProblem stores forms, NewtonSolver drives SNES
    problem = fem_petsc.NonlinearProblem(F_form, phi_hat, bcs=bcs, J=J_form)
    solver = dolfinx.nls.petsc.NewtonSolver(comm, problem)

    # Newton parameters
    solver.atol = snes_atol
    solver.rtol = snes_rtol
    solver.max_it = max_newton
    solver.convergence_criterion = "incremental"

    # KSP for linear sub-problems (accessed via the underlying SNES)
    ksp = solver.krylov_solver
    opts = PETSc.Options()
    opts["ksp_type"] = "cg"
    opts["ksp_rtol"] = 1e-9
    opts["ksp_atol"] = 1e-12
    opts["ksp_max_it"] = 1000
    opts["pc_type"] = "hypre"
    opts["pc_hypre_type"] = "boomeramg"
    ksp.setFromOptions()

    # Logging callback
    solver.report = verbose

    t0 = time.perf_counter()
    n_iters, converged = solver.solve(phi_hat)
    t_solve = time.perf_counter() - t0

    # Check exp argument bounds
    phi_arr = phi_hat.x.array
    phi_min_local = phi_arr.min()
    phi_max_local = phi_arr.max()
    phi_min = comm.allreduce(phi_min_local, op=MPI.MIN)
    phi_max = comm.allreduce(phi_max_local, op=MPI.MAX)

    info = {
        "converged": converged,
        "newton_iters": n_iters,
        "solve_time": t_solve,
        "phi_hat_min": phi_min,
        "phi_hat_max": phi_max,
    }

    if comm.rank == 0:
        tag = "CONVERGED" if converged else "FAILED"
        print(f"\n=== SNES {tag} ===")
        print(f"  Newton iterations : {n_iters}")
        print(f"  Converged         : {converged}")
        print(f"  Solve time        : {t_solve:.3f} s")
        print(f"  phi_hat range     : [{phi_min:.4f}, {phi_max:.4f}]")
        print(f"====================\n")

    if not converged:
        raise PoissonSolveError(
            f"SNES diverged: {n_iters} iters, converged={converged}"
        )

    return info


def run_nonlinear_poisson(args, comm, rank):
    """Run Brick 3: semi-linear Poisson with Boltzmann carriers."""
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)

    if rank == 0:
        print("\n" + "="*70)
        print("BRICK 3: Semi-linear Poisson (Boltzmann carriers)")
        print("="*70)
        print(scaling.summary())

    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2
    ni_ratio = args.n_i / scaling.n_ref

    if rank == 0:
        print(f"  V_bias    = {args.V_bias} V")
        print(f"  Nd        = {args.Nd:.3e} m⁻³")
        print(f"  Na        = {args.Na:.3e} m⁻³")
        print(f"  n_i       = {args.n_i:.3e} m⁻³")
        print(f"  n_i/n_ref = {ni_ratio:.6e}")
        print(f"  alpha_eff = {alpha_eff:.6e}")
        print()

    V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
        msh, scaling, dims, args.V_bias, args.Nd, args.Na, args.n_i
    )

    if rank == 0:
        print(f"  φ̂_ref (overflow shift) = {phi_ref_val:.4f}")
        print(f"  DOFs = {V.dofmap.index_map.size_global}")

    info = solve_nonlinear_poisson(
        F_form, J_form, phi_hat, bcs, comm, verbose=True
    )

    # Physical output
    if rank == 0:
        print(f"  φ_min = {scaling.phi_to_physical(info['phi_hat_min'])*1e3:.4f} mV")
        print(f"  φ_max = {scaling.phi_to_physical(info['phi_hat_max'])*1e3:.4f} mV")

    # XDMF export
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    phi_phys = fem.Function(V, name="phi_V")
    phi_phys.x.array[:] = phi_hat.x.array * scaling.V_t
    try:
        fname = out_dir / "phi_nonlinear.xdmf"
        with dolfinx.io.XDMFFile(comm, str(fname), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(phi_phys)
        if rank == 0:
            print(f"  Written to {fname}")
    except Exception as e:
        if rank == 0:
            print(f"  XDMF export skipped: {e}")

    return info


def run_nonlinear_bias_sweep(args, comm, rank):
    """Sweep V_bias for semi-linear Poisson. Report Newton iters vs bias."""
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)

    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    V_array = np.linspace(args.V0, args.V1, args.nV)

    if rank == 0:
        print("\n" + "="*70)
        print(f"BRICK 3 BIAS SWEEP: {args.V0} V -> {args.V1} V ({args.nV} pts)")
        print(f"  Nd={args.Nd:.3e}  Na={args.Na:.3e}  n_i={args.n_i:.3e}")
        print("="*70)
        print(f"{'V_bias':>8s}  {'Newton':>7s}  {'time':>7s}  "
              f"{'phi_min':>12s}  {'phi_max':>12s}  {'status':>10s}")
        print("-" * 70)

    all_ok = True
    for V_bias in V_array:
        try:
            V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
                msh, scaling, dims, V_bias, args.Nd, args.Na, args.n_i
            )
            info = solve_nonlinear_poisson(
                F_form, J_form, phi_hat, bcs, comm, verbose=False
            )
            if rank == 0:
                print(f"{V_bias:>8.3f}  {info['newton_iters']:>7d}  "
                      f"{info['solve_time']:>7.2f}  "
                      f"{info['phi_hat_min']:>12.4f}  {info['phi_hat_max']:>12.4f}  "
                      f"{'OK' if info['converged'] else 'FAIL':>10s}")
            if not info["converged"]:
                all_ok = False
        except PoissonSolveError as e:
            if rank == 0:
                print(f"{V_bias:>8.3f}  {'---':>7s}  {'---':>7s}  "
                      f"{'---':>12s}  {'---':>12s}  {'DIVERGED':>10s}")
            all_ok = False

    if rank == 0:
        print("="*70)
        if all_ok:
            print("BRICK 3 BIAS SWEEP PASSED")
        else:
            print("BRICK 3 BIAS SWEEP FAILED (some points diverged)")
        print()

    return 0 if all_ok else 1


# ════════════════════════════════════════════════════════════════════════
#  11.  MAIN DRIVER
# ════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MCT Photoconductor Poisson (Bricks 1-3)",
        epilog="PETSc options can be appended directly, e.g.: -pc_type gamg",
    )
    parser.add_argument("--V_bias", type=float, default=0.1, help="Bias voltage [V]")
    parser.add_argument("--nx", type=int, default=20)
    parser.add_argument("--ny", type=int, default=60)
    parser.add_argument("--nz", type=int, default=40)

    # Validation modes
    parser.add_argument("--test_mms", action="store_true", help="Run MMS convergence test")
    parser.add_argument("--sweep_bias", action="store_true", help="Run bias sweep test")
    parser.add_argument("--V0", type=float, default=0.0, help="Sweep start [V]")
    parser.add_argument("--V1", type=float, default=2.0, help="Sweep end [V]")
    parser.add_argument("--nV", type=int, default=11, help="Sweep points")

    # Doping
    parser.add_argument("--Nd", type=float, default=0.0,
                        help="Donor concentration [m⁻³]")
    parser.add_argument("--Na", type=float, default=0.0,
                        help="Acceptor concentration [m⁻³]")
    parser.add_argument("--doping_profile", type=str, default=None,
                        choices=["uniform", "step_y"],
                        help="Doping profile type")
    parser.add_argument("--y_step", type=float, default=1500e-6,
                        help="Step location for step_y doping [m]")
    parser.add_argument("--doping_sanity", action="store_true",
                        help="Run doping sanity sweep (V_bias=0, Nd/n_ref sweep)")

    # Brick 3: Nonlinear Poisson
    parser.add_argument("--nonlinear", action="store_true",
                        help="Brick 3: Semi-linear Poisson with Boltzmann carriers")
    parser.add_argument("--nl_sweep", action="store_true",
                        help="Brick 3: Bias sweep for nonlinear Poisson")
    parser.add_argument("--n_i", type=float, default=N_I_DEFAULT,
                        help="Intrinsic carrier density [m^-3]")

    args, _ = parser.parse_known_args()  # allow PETSc pass-through

    comm = MPI.COMM_WORLD
    rank = comm.rank

    # ── MMS Test Mode ──
    if args.test_mms:
        exit_code = run_mms_convergence_test()
        sys.exit(exit_code)

    # ── Bias Sweep Mode (linear) ──
    if args.sweep_bias:
        exit_code = run_bias_sweep(args.V0, args.V1, args.nV, args.nx, args.ny, args.nz)
        sys.exit(exit_code)

    # ── Doping Sanity Mode ──
    if args.doping_sanity:
        run_doping_sanity(args, comm, rank)
        sys.exit(0)

    # ── Brick 3: Nonlinear single point ──
    if args.nonlinear:
        run_nonlinear_poisson(args, comm, rank)
        sys.exit(0)

    # ── Brick 3: Nonlinear bias sweep ──
    if args.nl_sweep:
        exit_code = run_nonlinear_bias_sweep(args, comm, rank)
        sys.exit(exit_code)

    # ── Normal Production Mode ──
    scaling = make_mct_scaling_77K()
    if rank == 0:
        print(scaling.summary())
        print(f"  V_bias  = {args.V_bias} V  →  φ̂_bias = {scaling.phi_to_dimless(args.V_bias):.4f}\n")

    # ── Mesh ──
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)

    t0 = time.perf_counter()
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)
    t_mesh = time.perf_counter() - t0

    if rank == 0:
        n_elem = args.nx * args.ny * args.nz
        print(f"Mesh: {args.nx}×{args.ny}×{args.nz} = {n_elem} hex elements  ({t_mesh:.3f} s)")
        print(f"  Dimensionless domain: x̂∈[0,{dims['Lx_hat']:.6f}]  "
              f"ŷ∈[0,{dims['Ly_hat']:.6f}]  ẑ∈[0,{dims['Lz_hat']:.6f}]")
        print(f"  Normalization: L_norm = {dims['L_norm']*1e6:.1f} µm")

    # ── Doping Profile ──
    rho_hat_expr = None
    if args.doping_profile == "uniform":
        rho_hat_expr = uniform_doping_profile(args.Nd, args.Na, scaling.n_ref)
        rho_hat_val = (args.Nd - args.Na) / scaling.n_ref
        if rank == 0:
            print(f"  Doping: Uniform  Nd={args.Nd:.3e}  Na={args.Na:.3e}  "
                  f"ρ̂=(Nd−Na)/n_ref = {rho_hat_val:.6e}")
    elif args.doping_profile == "step_y":
        rho_hat_expr = step_y_doping_profile(
            args.Nd, args.Na, scaling.n_ref, args.y_step, dims["L_norm"]
        )
        if rank == 0:
            print(f"  Doping: step_y at y={args.y_step*1e6:.1f}µm  "
                  f"Nd={args.Nd:.3e}  Na={args.Na:.3e}")

    # ── Assemble ──
    t0 = time.perf_counter()
    V, A, b, bcs, phi_hat = build_poisson_system(
        msh, scaling, dims, args.V_bias, rho_hat_expr
    )
    t_asm = time.perf_counter() - t0

    ndofs = V.dofmap.index_map.size_global
    if rank == 0:
        print(f"  DOFs: {ndofs}  |  Assembly: {t_asm:.3f} s")

    # ── Solve ──
    t0 = time.perf_counter()
    result = solve_poisson(A, b, phi_hat, SolverConfig(), comm)
    t_solve = time.perf_counter() - t0

    if rank == 0:
        print(f"Solve time: {t_solve:.3f} s")

    # ── Sanity checks ──
    phi_min = comm.allreduce(phi_hat.x.array.min(), op=MPI.MIN)
    phi_max = comm.allreduce(phi_hat.x.array.max(), op=MPI.MAX)
    phi_bias_hat = scaling.phi_to_dimless(args.V_bias)

    if rank == 0:
        print(f"\n=== Solution Sanity Check ===")
        print(f"  φ̂_min = {phi_min:.6f}  (expect ≈ 0)")
        print(f"  φ̂_max = {phi_max:.6f}  (expect ≈ {phi_bias_hat:.6f})")
        print(f"  φ_min  = {scaling.phi_to_physical(phi_min)*1e3:.4f} mV")
        print(f"  φ_max  = {scaling.phi_to_physical(phi_max)*1e3:.4f} mV")

        if args.doping_profile is None:
            # Only check maximum principle for Laplace
            overshoot_rel = max(abs(phi_min), abs(phi_max - phi_bias_hat)) / max(abs(phi_bias_hat), 1e-10)
            if overshoot_rel < 1e-3:
                print(f"  Maximum principle: {overshoot_rel*100:.4f}% overshoot (acceptable) ✓")
            else:
                print(f"  *** WARNING: {overshoot_rel*100:.2f}% overshoot — check BCs! ***")
        print()

    # ── XDMF export ──
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    phi_phys = fem.Function(V, name="phi_V")
    phi_phys.x.array[:] = phi_hat.x.array * scaling.V_t
    try:
        with dolfinx.io.XDMFFile(comm, str(out_dir / "phi_solution.xdmf"), "w") as xdmf:
            xdmf.write_mesh(msh)
            xdmf.write_function(phi_phys)
        if rank == 0:
            print(f"Solution written to {out_dir / 'phi_solution.xdmf'}")
    except Exception as e:
        if rank == 0:
            print(f"XDMF export skipped: {e}")

    return result


if __name__ == "__main__":
    main()
