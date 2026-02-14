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

# Split sys.argv: argparse flags start with --, PETSc flags start with single -
# We must init petsc4py with PETSc flags BEFORE importing PETSc.
_petsc_argv = [sys.argv[0]]
_app_argv = [sys.argv[0]]
_i = 1
while _i < len(sys.argv):
    arg = sys.argv[_i]
    if arg.startswith("--"):
        # Application flag (argparse)
        _app_argv.append(arg)
        # Grab its value if present
        if _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("-"):
            _app_argv.append(sys.argv[_i + 1])
            _i += 2
        else:
            _i += 1
    else:
        # PETSc flag (single dash)
        _petsc_argv.append(arg)
        # Grab its value if present
        if _i + 1 < len(sys.argv) and not sys.argv[_i + 1].startswith("-"):
            _petsc_argv.append(sys.argv[_i + 1])
            _i += 2
        else:
            _i += 1

import petsc4py
petsc4py.init(_petsc_argv)
from petsc4py import PETSc

# Override sys.argv so argparse only sees application flags
sys.argv = _app_argv

import dolfinx
import dolfinx.fem as fem
import dolfinx.fem.petsc as fem_petsc
import dolfinx.mesh as dmesh
import dolfinx.io
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
#  4.  MMS VALIDATION  — manufactured solution test (trigonometric)
# ════════════════════════════════════════════════════════════════════════
#
#  Exact solution (3D product of sines, NOT representable in P1):
#    phi_exact(x,y,z) = sin(pi x/Lx) * sin(pi y/Ly) * sin(pi z/Lz)
#
#  Laplacian:
#    lap = -(pi/Lx)^2 * phi - (pi/Ly)^2 * phi - (pi/Lz)^2 * phi
#        = -[(pi/Lx)^2 + (pi/Ly)^2 + (pi/Lz)^2] * phi
#
#  So rho_hat = -lap / alpha_eff  =  [(pi/Lx)^2+...] * phi / alpha_eff
#
#  BCs: phi_exact = 0 on ALL boundaries (sin vanishes at 0 and L).

MMS_QUAD_DEGREE = 6  # quadrature degree for trig MMS (avoid quadrature error)


def build_mms_system(msh, scaling, dims):
    """
    Assemble MMS test system with full Dirichlet BCs.

    Exact solution: phi_exact = sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz)
    BCs: phi = 0 on all boundaries (homogeneous Dirichlet).
    RHS derived from strong form so that the weak form is exactly consistent.

    Returns (V, A, b, bcs, phi_hat, phi_exact_fn).
    """
    V = fem.functionspace(msh, ("Lagrange", 1))
    phi_hat = fem.Function(V, name="phi_hat")

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # Domain extents in dimensionless coordinates
    Lx = dims["Lx_hat"]
    Ly = dims["Ly_hat"]
    Lz = dims["Lz_hat"]

    # Physics scaling
    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2

    # --- Exact solution as UFL expression ---
    x = ufl.SpatialCoordinate(msh)
    pi = np.pi
    phi_exact_ufl = (ufl.sin(pi * x[0] / Lx)
                     * ufl.sin(pi * x[1] / Ly)
                     * ufl.sin(pi * x[2] / Lz))

    # --- RHS from strong form ---
    # lap(phi) = -[(pi/Lx)^2 + (pi/Ly)^2 + (pi/Lz)^2] * phi_exact
    # We need: a(u,v) = L(v)  =>  int grad(u).grad(v) dx = alpha_eff * int rho_hat * v dx
    # Strong form: -lap(phi) = -alpha_eff * rho_hat  =>  rho_hat = lap(phi) / alpha_eff
    # Actually: div(grad(phi_exact_ufl)) gives the Laplacian in UFL.
    # The weak form is: int grad(phi).grad(v) dx = int f * v dx
    #   where f = -lap(phi_exact) = -div(grad(phi_exact))
    # And we assemble RHS as: int alpha_eff * rho_hat * v dx
    #   so rho_hat = f / alpha_eff = -div(grad(phi_exact)) / alpha_eff
    #
    # For cleanliness: f = -div(grad(phi_exact_ufl)), then L = f * v * dx
    f_ufl = -ufl.div(ufl.grad(phi_exact_ufl))

    # --- Forms with elevated quadrature ---
    dx_mms = ufl.dx(metadata={"quadrature_degree": MMS_QUAD_DEGREE})
    a_form = ufl.inner(ufl.grad(u), ufl.grad(v)) * dx_mms
    L_form = f_ufl * v * dx_mms

    # --- BCs: homogeneous Dirichlet (phi_exact = 0 on all boundaries) ---
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_connectivity(fdim, tdim)
    boundary_facets = dmesh.exterior_facet_indices(msh.topology)
    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(PETSc.ScalarType(0.0), boundary_dofs, V)
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

    # --- Interpolate exact solution into V for error computation ---
    phi_exact_fn = fem.Function(V, name="phi_exact")

    def _phi_exact_numpy(x):
        return (np.sin(pi * x[0] / Lx)
                * np.sin(pi * x[1] / Ly)
                * np.sin(pi * x[2] / Lz))

    phi_exact_fn.interpolate(_phi_exact_numpy)

    return V, A, b, bcs, phi_hat, phi_exact_fn


def compute_errors(phi_h, phi_exact_fn, msh, dims, comm):
    """
    Compute L2 and H1 seminorm errors using elevated quadrature.

    Uses the UFL exact expression (not the interpolated function) for
    high-accuracy error integration via SpatialCoordinate.

    Returns (L2_error, H1_seminorm_error).
    """
    Lx = dims["Lx_hat"]
    Ly = dims["Ly_hat"]
    Lz = dims["Lz_hat"]
    pi = np.pi

    x = ufl.SpatialCoordinate(msh)
    phi_exact_ufl = (ufl.sin(pi * x[0] / Lx)
                     * ufl.sin(pi * x[1] / Ly)
                     * ufl.sin(pi * x[2] / Lz))

    dx_err = ufl.dx(metadata={"quadrature_degree": MMS_QUAD_DEGREE})

    # L2 error: ||phi_h - phi_exact||_L2
    e = phi_h - phi_exact_ufl
    L2_form = fem.form(e**2 * dx_err)
    L2_local = fem.assemble_scalar(L2_form)
    L2_global = comm.allreduce(L2_local, op=MPI.SUM)
    L2_error = np.sqrt(abs(L2_global))

    # H1 seminorm: ||grad(phi_h - phi_exact)||_L2
    grad_e = ufl.grad(phi_h) - ufl.grad(phi_exact_ufl)
    H1_form = fem.form(ufl.inner(grad_e, grad_e) * dx_err)
    H1_local = fem.assemble_scalar(H1_form)
    H1_global = comm.allreduce(H1_local, op=MPI.SUM)
    H1_seminorm = np.sqrt(abs(H1_global))

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


def solve_poisson(A, b, phi, cfg=SolverConfig(), comm=MPI.COMM_WORLD,
                  prefix=None) -> SolveResult:
    """CG + AMG solve with convergence check.  Raises on divergence."""
    ksp = PETSc.KSP().create(comm)
    if prefix:
        ksp.setOptionsPrefix(prefix)
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
    Run MMS convergence test with trigonometric exact solution.

    phi_exact = sin(pi x/Lx) sin(pi y/Ly) sin(pi z/Lz)

    Expected convergence rates for P1 elements:
    - L2 error: O(h^2) -> rate ~ 2.0
    - H1 seminorm: O(h) -> rate ~ 1.0

    Exits with code 1 if convergence rates are outside tolerance.
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()

    if rank == 0:
        print("\n" + "="*70)
        print("MMS CONVERGENCE TEST: phi_exact = sin(pi x/Lx)*sin(pi y/Ly)*sin(pi z/Lz)")
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
        V, A, b, bcs, phi_hat, phi_exact_fn = build_mms_system(msh, scaling, dims)

        ndofs = V.dofmap.index_map.size_global

        # Solve
        t0 = time.perf_counter()
        result = solve_poisson(A, b, phi_hat, SolverConfig(), comm)
        t_solve = time.perf_counter() - t0

        # Compute errors (using UFL exact expression, elevated quadrature)
        L2_err, H1_err = compute_errors(phi_hat, phi_exact_fn, msh, dims, comm)
        
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
            # Roundoff guard: if finest error is < 1e-12, rates are meaningless
            finest_L2 = results[-1]['L2']
            finest_H1 = results[-1]['H1']
            if finest_L2 < 1e-12 or finest_H1 < 1e-12:
                print("\n*** ROUNDOFF DOMINATED: finest L2={:.2e}, H1={:.2e} ***".format(
                    finest_L2, finest_H1))
                print("*** MMS not informative at this resolution. ***")
                print("*** Reporting PASS (solver is exact to roundoff). ***")
                return 0

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
    # Physical: n/n_ref = (n_i/n_ref) * exp(phi_hat)
    #           p/n_ref = (n_i/n_ref) * exp(-phi_hat)
    # Shifted:  n/n_ref = (n_i/n_ref) * exp(phi_ref) * exp(phi_hat - phi_ref)
    #           p/n_ref = (n_i/n_ref) * exp(-phi_ref) * exp(-(phi_hat - phi_ref))
    # We absorb exp(+-phi_ref) into the prefactors so physics is preserved.
    exp_phi_ref_pos = ufl.exp(phi_ref)    # = exp(phi_ref_hat_value)
    exp_phi_ref_neg = ufl.exp(-phi_ref)   # = exp(-phi_ref_hat_value)
    n_hat = exp_phi_ref_pos * _safe_exp(+(phi_hat - phi_ref))    # = exp(phi_hat)
    p_hat = exp_phi_ref_neg * _safe_exp(-(phi_hat - phi_ref))    # = exp(-phi_hat)

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

    # Initial guess: solve LINEAR Poisson with doping as RHS.
    # This gives a much better starting point than a naive ramp,
    # because it already accounts for the doping-induced potential.
    rho_dop_val = (Nd - Na) / scaling.n_ref
    if abs(rho_dop_val) > 1e-30:
        rho_dop_expr = lambda x: np.full(x.shape[1], rho_dop_val)
    else:
        rho_dop_expr = None

    if msh.comm.rank == 0:
        print(f"  Computing linear Poisson initial guess (rho_dop={rho_dop_val:.3e})...",
              flush=True)

    # Solve linear Poisson for initial guess
    V_lin, A_lin, b_lin, bcs_lin, phi_lin = build_poisson_system(
        msh, scaling, dims, V_bias, rho_hat_expr=rho_dop_expr
    )
    solve_poisson(A_lin, b_lin, phi_lin, SolverConfig(), msh.comm,
                  prefix="init_")

    # Copy linear solution into phi_hat as initial guess
    phi_hat.x.array[:] = phi_lin.x.array[:]
    phi_hat.x.scatter_forward()

    # Global min/max via MPI reduction (owned DOFs only)
    n_own = V.dofmap.index_map.size_local
    phi_local_min = phi_hat.x.array[:n_own].min() if n_own > 0 else 1e30
    phi_local_max = phi_hat.x.array[:n_own].max() if n_own > 0 else -1e30
    phi_global_min = msh.comm.allreduce(phi_local_min, op=MPI.MIN)
    phi_global_max = msh.comm.allreduce(phi_local_max, op=MPI.MAX)

    if msh.comm.rank == 0:
        print(f"  Initial guess: phi_hat in [{phi_global_min:.4f}, "
              f"{phi_global_max:.4f}] (global MPI min/max)", flush=True)

    return V, F_form, J_form, bcs, phi_hat, phi_ref_hat_value


def solve_nonlinear_poisson(F_form, J_form, phi_hat, bcs, comm,
                            phi_ref_hat_value=0.0,
                            max_newton=50, snes_rtol=1e-8, snes_atol=1e-10,
                            verbose=True):
    """
    Solve F(phi_hat) = 0 using DOLFINx NonlinearProblem (DOLFINx 0.10).

    Returns dict with convergence info + clamp diagnostics.
    """
    # Create NonlinearProblem — handles form compilation, vector/matrix
    # creation, and SNES callback wiring with correct ghost structure.
    problem = fem_petsc.NonlinearProblem(
        F_form, phi_hat, bcs=bcs, J=J_form,
        petsc_options_prefix="nl_"
    )

    # Configure SNES directly (not via PETSc options, which don't get consumed)
    snes = problem._snes
    snes.setType("newtonls")
    snes.setTolerances(rtol=snes_rtol, atol=snes_atol, max_it=max_newton)

    ls = snes.getLineSearch()
    ls.setType("bt")

    # KSP: GMRES + AMG
    ksp = snes.getKSP()
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-9, atol=1e-12, max_it=1000)
    pc = ksp.getPC()
    pc.setType("hypre")
    pc.setHYPREType("boomeramg")

    if verbose:
        snes.setMonitor(
            lambda s, it, fnorm: print(
                f"  SNES iter {it:3d} | ||F|| = {fnorm:.6e}", flush=True
            ) if comm.rank == 0 else None
        )

    # Allow CLI overrides via -nl_snes_*, -nl_ksp_*, -nl_pc_*
    snes.setFromOptions()

    if comm.rank == 0:
        print(f"  [SNES config] type={snes.getType()} max_it={max_newton} "
              f"rtol={snes_rtol:.1e} atol={snes_atol:.1e} "
              f"ls={ls.getType()}", flush=True)
        print(f"  [KSP  config] type={ksp.getType()} pc={pc.getType()}", flush=True)

    # --- Solve ---
    if comm.rank == 0:
        print(f"  Starting SNES solve...", flush=True)
    t0 = time.perf_counter()
    problem.solve()
    phi_hat.x.scatter_forward()
    t_solve = time.perf_counter() - t0

    n_iters = snes.getIterationNumber()
    fnorm = snes.getFunctionNorm()
    reason = snes.getConvergedReason()
    converged = reason > 0

    _SNES_REASONS = {
        2: "RTOL", 3: "ATOL", 4: "STOL", 5: "ITS",
        -1: "FNORM_NAN", -2: "FUNCTION_COUNT", -3: "LINEAR_SOLVE",
        -4: "FUNCTION_RANGE", -5: "DTOL",
        -6: "DIVERGED_LINE_SEARCH", -7: "DIVERGED_LOCAL_MIN",
        -8: "MAX_IT",
    }
    reason_str = _SNES_REASONS.get(reason, f"CODE_{reason}")

    # --- Exp-argument clamp diagnostics (owned DOFs only, no ghosts) ---
    n_owned = phi_hat.function_space.dofmap.index_map.size_local
    phi_arr = phi_hat.x.array[:n_owned]
    phi_min_local = phi_arr.min() if n_owned > 0 else 1e30
    phi_max_local = phi_arr.max() if n_owned > 0 else -1e30
    phi_min = comm.allreduce(phi_min_local, op=MPI.MIN)
    phi_max = comm.allreduce(phi_max_local, op=MPI.MAX)

    arg_arr = phi_arr - phi_ref_hat_value
    arg_min_local = arg_arr.min() if n_owned > 0 else 1e30
    arg_max_local = arg_arr.max() if n_owned > 0 else -1e30
    arg_min = comm.allreduce(arg_min_local, op=MPI.MIN)
    arg_max = comm.allreduce(arg_max_local, op=MPI.MAX)

    n_clamp_hi_local = int(np.sum(arg_arr > EXP_CLIP))
    n_clamp_lo_local = int(np.sum(arg_arr < -EXP_CLIP))
    n_total = comm.allreduce(n_owned, op=MPI.SUM)
    n_clamp_hi = comm.allreduce(n_clamp_hi_local, op=MPI.SUM)
    n_clamp_lo = comm.allreduce(n_clamp_lo_local, op=MPI.SUM)
    pct_clamp_hi = 100.0 * n_clamp_hi / max(n_total, 1)
    pct_clamp_lo = 100.0 * n_clamp_lo / max(n_total, 1)

    info = {
        "converged": converged,
        "reason": reason,
        "reason_str": reason_str,
        "newton_iters": n_iters,
        "fnorm": fnorm,
        "solve_time": t_solve,
        "phi_hat_min": phi_min,
        "phi_hat_max": phi_max,
        "exp_arg_min": arg_min,
        "exp_arg_max": arg_max,
        "pct_clamp_hi": pct_clamp_hi,
        "pct_clamp_lo": pct_clamp_lo,
    }

    if comm.rank == 0:
        tag = "CONVERGED" if converged else "FAILED"
        print(f"\n=== SNES {tag} ===")
        print(f"  SNES iterations   : {n_iters}")
        print(f"  Final ||F||       : {fnorm:.6e}")
        print(f"  Reason            : {reason_str} ({reason})")
        # KSP diagnostics if SNES failed due to linear solve
        if reason == -3:  # LINEAR_SOLVE
            ksp = snes.getKSP()
            ksp_reason = ksp.getConvergedReason()
            ksp_its = ksp.getIterationNumber()
            ksp_rnorm = ksp.getResidualNorm()
            print(f"  --- KSP failure details ---")
            print(f"  KSP iterations  : {ksp_its}")
            print(f"  KSP residual    : {ksp_rnorm:.6e}")
            print(f"  KSP reason      : {ksp_reason}")
        print(f"  Solve time        : {t_solve:.3f} s")
        print(f"  phi_hat range     : [{phi_min:.4f}, {phi_max:.4f}]")
        print(f"  --- Exp-argument diagnostics ---")
        print(f"  phi_ref_hat       : {phi_ref_hat_value:.4f}")
        print(f"  arg = phi-phi_ref : [{arg_min:.4f}, {arg_max:.4f}]")
        print(f"  EXP_CLIP          : +/-{EXP_CLIP}")
        print(f"  DOFs clamped high : {n_clamp_hi}/{n_total} ({pct_clamp_hi:.4f}%)")
        print(f"  DOFs clamped low  : {n_clamp_lo}/{n_total} ({pct_clamp_lo:.4f}%)")
        if pct_clamp_hi > 1.0 or pct_clamp_lo > 1.0:
            print(f"  *** WARNING: >1% DOFs clamped -- solution may be physically nonsense ***")
        print(f"====================\n")

    # NonlinearProblem owns the SNES — don't destroy it here

    if not converged:
        raise PoissonSolveError(
            f"SNES diverged: {n_iters} iters, reason={reason_str}, ||F||={fnorm:.3e}"
        )

    return info


def check_jacobian_consistency(F_form, J_form, phi_hat, bcs, comm, eps=1e-7):
    """
    Directional derivative test: verify J is the derivative of F.

    Computes: |F(u + eps*v) - F(u) - eps*J(u)*v| / (eps * |J(u)*v|)
    Should be << 1 (typically 1e-4 to 1e-6).

    Uses fem.Function for all vectors to ensure correct MPI-distributed layout.
    """
    V = phi_hat.function_space
    F_compiled = fem.form(F_form)
    J_compiled = fem.form(J_form)

    # --- Helper: assemble residual at current phi_hat state ---
    def assemble_F_vec():
        """Assemble F into a new PETSc Vec (owned by returned Function)."""
        f = fem.Function(V)
        fvec = f.x.petsc_vec
        with fvec.localForm() as f_local:
            f_local.set(0.0)
        fem_petsc.assemble_vector(fvec, F_compiled)
        fem_petsc.apply_lifting(fvec, [J_compiled], bcs=[bcs],
                                x0=[phi_hat.x.petsc_vec], alpha=-1.0)
        fvec.ghostUpdate(addv=PETSc.InsertMode.ADD,
                         mode=PETSc.ScatterMode.REVERSE)
        fem_petsc.set_bc(fvec, bcs, phi_hat.x.petsc_vec)
        return f

    # 1) Save current state (owned DOFs only)
    n_owned = V.dofmap.index_map.size_local
    u0 = phi_hat.x.array[:n_owned].copy()

    # 2) Assemble F(u) at current state
    F0_fun = assemble_F_vec()

    # 3) Assemble J(u)
    A = fem_petsc.create_matrix(J_compiled)
    A.zeroEntries()
    fem_petsc.assemble_matrix(A, J_compiled, bcs=bcs)
    A.assemble()

    # 4) Build random direction v (distributed, owned DOFs only)
    v_fun = fem.Function(V)
    rng = np.random.default_rng(seed=42 + comm.rank)
    v_fun.x.array[:n_owned] = rng.standard_normal(n_owned)
    # Zero v on BC DOFs so perturbation respects BCs
    for bc in bcs:
        bc_dofs = bc.dof_indices()[0]
        owned_bc = bc_dofs[bc_dofs < n_owned]
        v_fun.x.array[owned_bc] = 0.0
    v_fun.x.scatter_forward()

    # 5) Compute J(u)*v
    Jv_fun = fem.Function(V)
    A.mult(v_fun.x.petsc_vec, Jv_fun.x.petsc_vec)

    # 6) Perturb state: u_pert = u + eps*v
    phi_hat.x.array[:n_owned] = u0 + eps * v_fun.x.array[:n_owned]
    phi_hat.x.scatter_forward()

    # 7) Assemble F(u + eps*v)
    F1_fun = assemble_F_vec()

    # 8) Restore state immediately
    phi_hat.x.array[:n_owned] = u0
    phi_hat.x.scatter_forward()

    # 9) Form difference: dF = F1 - F0 - eps*Jv (using PETSc Vec ops)
    dF = F1_fun.x.petsc_vec.copy()
    dF.axpy(-1.0, F0_fun.x.petsc_vec)   # dF = F1 - F0
    dF.axpy(-eps, Jv_fun.x.petsc_vec)   # dF = (F1 - F0) - eps*Jv

    # 10) Norm ratio (PETSc norms are already global/MPI-reduced)
    err_norm = dF.norm()
    Jv_norm = Jv_fun.x.petsc_vec.norm()
    F0_norm = F0_fun.x.petsc_vec.norm()
    F1_norm = F1_fun.x.petsc_vec.norm()
    rel_err = err_norm / max(eps * Jv_norm, 1e-30)

    if comm.rank == 0:
        print(f"\n=== JACOBIAN CONSISTENCY TEST ===")
        print(f"  eps             = {eps:.1e}")
        print(f"  ||F(u)||        = {F0_norm:.6e}")
        print(f"  ||F(u+eps*v)||  = {F1_norm:.6e}")
        print(f"  ||J(u)*v||      = {Jv_norm:.6e}")
        print(f"  ||F1-F0-eps*Jv||= {err_norm:.6e}")
        print(f"  Relative error  = {rel_err:.6e}")
        if rel_err < 1e-3:
            print(f"  PASSED (Jacobian consistent with residual)")
        else:
            print(f"  *** FAILED *** (Jacobian inconsistent)")
        print(f"================================\n", flush=True)

    dF.destroy()
    A.destroy()

    return rel_err


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

    # Gate A: Jacobian consistency test (MANDATORY before trusting Newton)
    check_jacobian_consistency(F_form, J_form, phi_hat, bcs, comm)

    # Solve (catch divergence so we still get diagnostics)
    try:
        info = solve_nonlinear_poisson(
            F_form, J_form, phi_hat, bcs, comm,
            phi_ref_hat_value=phi_ref_val, verbose=True
        )
    except PoissonSolveError as e:
        if rank == 0:
            print(f"  SNES solve failed: {e}")
        return None

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


# ════════════════════════════════════════════════════════════════════════
#  BRICK 4a: Electron continuity with FIXED φ̂ (dark, G=R=0)
# ════════════════════════════════════════════════════════════════════════
#
#  Physics:
#    ∇·Jn = 0  (steady state, no generation, no recombination)
#    Jn = q μn (-n ∇φ + Vt ∇n)
#
#  With φ̂ fixed from Brick 3, this is LINEAR in n̂.
#
#  Dimensionless weak form (after IBP, n̂ = n/n_ref):
#    ∫ ∇̂n̂ · ∇̂w dx̂ - ∫ n̂ (∇̂φ̂ · ∇̂w) dx̂ = 0
#
#  This is: diffusion - drift = 0, tested against w.
#
#  BCs (ohmic contacts, dark): n̂ = N̂d at both contacts.
#
#  Acceptance tests:
#    1) Jn spatially constant (relative variation < 1e-3)
#    2) Jn_mean matches analytic J = q μn Nd V_bias / L within 5%
#    3) n ≈ Nd everywhere (quasi-neutrality, dark)
#    4) Jn = 0 when V_bias = 0

def build_continuity_linear(msh, scaling, dims, phi_hat, Nd, mu_n=1.0,
                            tau_n=0.0, G0=0.0, alpha_opt=0.0):
    """
    Build the SUPG-stabilized linear electron continuity system with φ̂ FIXED.

    PDE (dimensionless):
        ∇̂·Ĵn = Ŝ    where  Ĵn = -n̂ ∇̂φ̂ + ∇̂n̂
        Ŝ = Ĝ - R̂   (generation minus recombination, nondimensionalized)

    Nondimensionalization of source terms:
        The continuity equation ∇·Jn = q(G - R) in physical units becomes,
        after dividing by (q μn n_ref Vt / L_norm²):
            ∇̂·Ĵn = Ĝ - R̂
        where:
            R̂ = (n̂ - n̂₀) / τ̂    with τ̂ = τ_n × (μn Vt / L_norm²)
            Ĝ = G0 × (L_norm² / (n_ref μn Vt)) × exp(-α̂ x̂₀)
            α̂ = alpha_opt × L_norm

    Brick 4a: tau_n=0, G0=0 → pure dark resistor
    Brick 4b: tau_n>0, G0=0 → dark with recombination
    Brick 4c: tau_n>0, G0>0  → illuminated photoconductor

    With R linear in n̂, the problem remains LINEAR in n̂:
        a(n̂, w) = L(w) + source terms

    Returns (V_n, A, b, bcs_n, n_hat, source_info).
    """
    V_n = phi_hat.function_space
    n_hat = fem.Function(V_n, name="n_hat")
    u_n = ufl.TrialFunction(V_n)
    w = ufl.TestFunction(V_n)

    L_norm = dims["L_norm"]
    Nd_hat = Nd / scaling.n_ref

    # Advection velocity v = ∇̂φ̂
    v = ufl.grad(phi_hat)
    v_mag = ufl.sqrt(ufl.inner(v, v) + 1e-30)

    # --- Standard Galerkin ---
    a_galerkin = (ufl.inner(ufl.grad(u_n), ufl.grad(w)) * ufl.dx
                  - u_n * ufl.inner(v, ufl.grad(w)) * ufl.dx)

    # --- SUPG stabilization ---
    h = ufl.CellDiameter(msh)
    Pe_local = v_mag * h / 2.0
    xi = ufl.min_value(Pe_local / 3.0, 1.0)
    tau_supg = (h / (2.0 * v_mag)) * xi

    R_trial = ufl.inner(v, ufl.grad(u_n))
    w_supg = tau_supg * ufl.inner(v, ufl.grad(w))
    a_supg = w_supg * R_trial * ufl.dx

    # --- Recombination: R = (n - n0) / τ_n ---
    # In weak form: ∇̂·Ĵn = Ĝ - R̂
    # After IBP:  a(n,w) = ∫ Ŝ w dx̂
    # R̂ = (n̂ - n̂₀)/τ̂  is linear in n̂ → moves to LHS:
    #   a(n,w) += (1/τ̂) ∫ n̂ w dx̂    (adds "reaction" to bilinear form)
    #   L(w)   += (n̂₀/τ̂) ∫ w dx̂     (adds constant to RHS)

    source_info = {"tau_hat": 0.0, "G0_hat": 0.0, "alpha_hat": 0.0,
                   "has_recomb": False, "has_gen": False}

    a_recomb = fem.Constant(msh, PETSc.ScalarType(0.0)) * u_n * w * ufl.dx
    L_form = fem.Constant(msh, PETSc.ScalarType(0.0)) * w * ufl.dx

    if tau_n > 0:
        tau_hat = tau_n * mu_n * scaling.V_t / L_norm**2
        n0_hat = Nd_hat  # dark equilibrium for n-type
        source_info["tau_hat"] = tau_hat
        source_info["has_recomb"] = True

        # Reaction term: (1/τ̂) ∫ n̂ w dx̂ added to bilinear form
        inv_tau = fem.Constant(msh, PETSc.ScalarType(1.0 / tau_hat))
        a_recomb = inv_tau * u_n * w * ufl.dx

        # RHS contribution: (n̂₀/τ̂) ∫ w dx̂
        src_recomb = fem.Constant(msh, PETSc.ScalarType(n0_hat / tau_hat))
        L_form = src_recomb * w * ufl.dx

        # SUPG consistent source: add τ(v·∇w) × (source) for consistency
        # For recombination: strong-form source = -R̂ = -(n̂ - n̂₀)/τ̂
        # Trial part: -(1/τ̂) u_n  → goes to LHS as SUPG reaction
        a_recomb += inv_tau * w_supg * u_n * ufl.dx
        # Constant part: n̂₀/τ̂ → goes to RHS SUPG
        L_form += src_recomb * w_supg * ufl.dx

    # --- Generation: G(x) = G0 exp(-α x) ---
    if G0 > 0:
        G0_hat = G0 * L_norm**2 / (scaling.n_ref * mu_n * scaling.V_t)
        alpha_hat_opt = alpha_opt * L_norm
        source_info["G0_hat"] = G0_hat
        source_info["alpha_hat"] = alpha_hat_opt
        source_info["has_gen"] = True

        x = ufl.SpatialCoordinate(msh)
        # Generation depends on x[0] (thickness direction)
        G_hat_expr = fem.Constant(msh, PETSc.ScalarType(G0_hat)) * ufl.exp(
            fem.Constant(msh, PETSc.ScalarType(-alpha_hat_opt)) * x[0])

        # Adds to RHS: ∫ Ĝ w dx̂
        L_form += G_hat_expr * w * ufl.dx
        # SUPG consistent: ∫ Ĝ × τ(v·∇w) dx̂
        L_form += G_hat_expr * w_supg * ufl.dx

    # Total bilinear form (including recombination reaction term)
    a_form = a_galerkin + a_supg + a_recomb

    # BCs: n̂ = Nd/n_ref at ohmic contacts
    Nd_hat = Nd / scaling.n_ref
    L_norm = dims["L_norm"]
    Lx_hat = dims["Lx_hat"]
    Ly_hat = dims["Ly_hat"]
    y_left_max = 1000e-6 / L_norm
    y_right_min = 2000e-6 / L_norm
    tol = 1e-8

    dofs_left = _locate_contact_dofs(V_n, msh, 0.0, 0.0, y_left_max, tol)
    bc_left = fem.dirichletbc(PETSc.ScalarType(Nd_hat), dofs_left, V_n)

    dofs_right = _locate_contact_dofs(V_n, msh, Lx_hat, y_right_min, Ly_hat, tol)
    bc_right = fem.dirichletbc(PETSc.ScalarType(Nd_hat), dofs_right, V_n)

    bcs_n = [bc_left, bc_right]

    # Compile and assemble
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs_n)
    A.assemble()

    b = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b, [a_compiled], bcs=[bcs_n])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b, bcs_n)

    # Initial guess: n̂ = Nd_hat everywhere
    n_hat.x.array[:] = Nd_hat
    n_hat.x.scatter_forward()

    return V_n, A, b, bcs_n, n_hat, source_info


def solve_continuity(A, b, n_hat, comm, cfg=None):
    """Solve the linear continuity system An = b.

    Uses GMRES + ILU (nonsymmetric convection-diffusion operator).
    NOT CG — the operator is nonsymmetric due to the drift term.
    """
    ksp = PETSc.KSP().create(comm)
    ksp.setOptionsPrefix("cont_")
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
    # GMRES restart (default 30 may be too small for nonsymmetric problems)
    ksp.setGMRESRestart(100)

    pc = ksp.getPC()
    # For parallel: use ASM (additive Schwarz) with ILU sub-blocks
    # For serial: use ILU directly
    if comm.size > 1:
        pc.setType("asm")
        # Sub-solver: ILU on each subdomain
        ksp.setUp()
        try:
            for sub_ksp in pc.getASMSubKSP():
                sub_ksp.setType("preonly")
                sub_pc = sub_ksp.getPC()
                sub_pc.setType("ilu")
        except Exception:
            pass  # fallback: default sub-solvers
    else:
        pc.setType("ilu")

    ksp.setFromOptions()
    ksp.solve(b, n_hat.x.petsc_vec)
    n_hat.x.scatter_forward()

    its = ksp.getIterationNumber()
    rnorm = ksp.getResidualNorm()
    reason = ksp.getConvergedReason()
    converged = reason > 0

    if comm.rank == 0:
        tag = "CONVERGED" if converged else "FAILED"
        print(f"\n=== Continuity KSP {tag} ===")
        print(f"  Iterations : {its}")
        print(f"  Residual   : {rnorm:.6e}")
        print(f"  Reason     : {reason}")
        print(f"====================\n")

    ksp.destroy()

    if not converged:
        raise PoissonSolveError(
            f"Continuity solve diverged: {its} iters, reason={reason}, rnorm={rnorm:.3e}"
        )

    return SolveResult(converged, its, rnorm, reason,
                       {2: "RTOL", 3: "ATOL"}.get(reason, f"CODE_{reason}"))


def compute_current_density_3d(phi_hat, n_hat, scaling, dims, mu_n, comm):
    """Compute Jn from converged φ̂ and n̂ on the 3D mesh.

    Jn = q μn (-n ∇φ + Vt ∇n)  [A/m²]

    In dimensionless form:
        Ĵ = (-n̂ ∇̂φ̂ + ∇̂n̂)
    Physical:
        Jn = (q μn n_ref Vt / L_norm) × Ĵ

    Returns: J_scale (physical prefactor), Jn_hat (DG0 vector Function)
    """
    V = phi_hat.function_space
    msh = V.mesh

    # Compute Jn_hat = -n̂ ∇̂φ̂ + ∇̂n̂ as a vector DG0 field
    gdim = msh.geometry.dim
    V_dg = fem.functionspace(msh, ("DG", 0, (gdim,)))
    Jn_hat = fem.Function(V_dg, name="Jn_hat")

    # Project: Jn_hat = -n_hat * grad(phi_hat) + grad(n_hat)
    Jn_expr = -n_hat * ufl.grad(phi_hat) + ufl.grad(n_hat)
    # DOLFINx version compat: interpolation_points may be property or method
    ip = V_dg.element.interpolation_points
    if callable(ip):
        ip = ip()
    expr = fem.Expression(Jn_expr, ip)
    Jn_hat.interpolate(expr)
    Jn_hat.x.scatter_forward()

    # Physical scale factor
    L_norm = dims["L_norm"]
    J_scale = Q_E * mu_n * scaling.n_ref * scaling.V_t / L_norm

    return J_scale, Jn_hat


def run_brick4a_tests(phi_hat, n_hat, scaling, dims, Nd, mu_n, V_bias, comm,
                      tau_n=0.0, G0=0.0, alpha_opt=0.0, source_info=None):
    """
    Brick 4a acceptance tests for dark biased resistor.

    Physics-correct tests for 3D with partial contacts:
    1) n > 0 everywhere (positivity)
    2) n ≈ Nd (quasi-neutrality, dark)
    3) ∇·Jn ≈ 0 (current conservation, L2 norm)
    4) Contact current balance: |I_left + I_right| / max(|I|) < tol
    5) Scaling cross-checks (J_scale, Pe, dimensional consistency)
    """
    rank = comm.rank
    msh = phi_hat.function_space.mesh
    gdim = msh.geometry.dim

    # --- Compute dimensionless Jn_hat = -n̂ ∇̂φ̂ + ∇̂n̂ ---
    J_scale, Jn_hat = compute_current_density_3d(
        phi_hat, n_hat, scaling, dims, mu_n, comm
    )

    # --- n̂ statistics ---
    n_own = n_hat.function_space.dofmap.index_map.size_local
    n_arr = n_hat.x.array[:n_own]
    n_hat_min = comm.allreduce(n_arr.min() if n_own > 0 else 1e30, op=MPI.MIN)
    n_hat_max = comm.allreduce(n_arr.max() if n_own > 0 else -1e30, op=MPI.MAX)
    n_phys_min = n_hat_min * scaling.n_ref
    n_phys_max = n_hat_max * scaling.n_ref

    # --- φ̂ statistics ---
    phi_own = phi_hat.function_space.dofmap.index_map.size_local
    phi_arr = phi_hat.x.array[:phi_own]
    phi_hat_min = comm.allreduce(phi_arr.min() if phi_own > 0 else 1e30, op=MPI.MIN)
    phi_hat_max = comm.allreduce(phi_arr.max() if phi_own > 0 else -1e30, op=MPI.MAX)

    # --- Divergence of Jn (conservation test) ---
    # ∇·Jn_hat should be 0 for G=R=0
    # Compute in DG0 scalar space
    V_dg0 = fem.functionspace(msh, ("DG", 0))
    Jn_expr_ufl = -n_hat * ufl.grad(phi_hat) + ufl.grad(n_hat)
    div_Jn = ufl.div(Jn_expr_ufl)

    # L2 norm of div(Jn) via integration
    div_Jn_sq_form = fem.form(div_Jn**2 * ufl.dx)
    div_Jn_sq_local = fem.assemble_scalar(div_Jn_sq_form)
    div_Jn_sq_global = comm.allreduce(div_Jn_sq_local, op=MPI.SUM)
    div_Jn_L2 = np.sqrt(max(div_Jn_sq_global, 0.0))

    # L2 norm of Jn for normalization
    Jn_sq_form = fem.form(ufl.inner(Jn_expr_ufl, Jn_expr_ufl) * ufl.dx)
    Jn_sq_local = fem.assemble_scalar(Jn_sq_form)
    Jn_sq_global = comm.allreduce(Jn_sq_local, op=MPI.SUM)
    Jn_L2 = np.sqrt(max(Jn_sq_global, 0.0))

    # Domain length scale for normalization
    L_norm = dims["L_norm"]
    Lx_hat = dims["Lx_hat"]

    # Relative divergence metric: η_div = ||∇·Jn||_L2 / (||Jn||_L2 / L)
    eta_div = div_Jn_L2 / max(Jn_L2 / Lx_hat, 1e-30)

    # --- Contact current balance ---
    # I_k = ∫_Γk Jn · n̂ dA  (dimensionless, multiply by J_scale for physical)
    # Left contact: x=0 (outward normal = -x̂), so Jn·n̂ = -Jn_x
    # Right contact: x=Lx_hat (outward normal = +x̂), so Jn·n̂ = +Jn_x
    Ly_hat = dims["Ly_hat"]
    y_left_max = 1000e-6 / L_norm
    y_right_min = 2000e-6 / L_norm
    tol_bc = 1e-8

    # Create boundary measure with marked facets
    tdim = msh.topology.dim
    fdim = tdim - 1

    # Left contact facets
    def left_marker(x):
        return (np.abs(x[0]) < tol_bc) & (x[1] >= -tol_bc) & (x[1] <= y_left_max + tol_bc)
    left_facets = dmesh.locate_entities_boundary(msh, fdim, left_marker)

    # Right contact facets
    def right_marker(x):
        return (np.abs(x[0] - Lx_hat) < tol_bc) & (x[1] >= y_right_min - tol_bc) & (x[1] <= Ly_hat + tol_bc)
    right_facets = dmesh.locate_entities_boundary(msh, fdim, right_marker)

    # Create facet tags for boundary integration
    # Tag left=1, right=2
    all_facets = np.concatenate([left_facets, right_facets])
    all_tags = np.concatenate([np.full_like(left_facets, 1),
                               np.full_like(right_facets, 2)])
    sort_idx = np.argsort(all_facets)
    facet_tags = dmesh.meshtags(msh, fdim, all_facets[sort_idx], all_tags[sort_idx])

    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    n_vec = ufl.FacetNormal(msh)

    # Flux through left contact (outward = -x̂)
    I_left_form = fem.form(ufl.inner(Jn_expr_ufl, n_vec) * ds(1))
    I_left_local = fem.assemble_scalar(I_left_form)
    I_left_hat = comm.allreduce(I_left_local, op=MPI.SUM)

    # Flux through right contact (outward = +x̂)
    I_right_form = fem.form(ufl.inner(Jn_expr_ufl, n_vec) * ds(2))
    I_right_local = fem.assemble_scalar(I_right_form)
    I_right_hat = comm.allreduce(I_right_local, op=MPI.SUM)

    # Physical currents
    I_left_phys = I_left_hat * J_scale   # A/m² × (dimensionless area already in integral)
    I_right_phys = I_right_hat * J_scale

    # Current balance
    I_max = max(abs(I_left_hat), abs(I_right_hat), 1e-30)
    eta_I = abs(I_left_hat + I_right_hat) / I_max

    # --- Volume balance for Bricks 4b/4c ---
    # Global identity: ∫_∂Ω Jn·n̂ dA = ∫_Ω (G - R) dV
    # i.e. I_left + I_right = ∫(Ĝ - R̂) dV̂
    vol_G_hat = 0.0   # ∫ Ĝ dV̂
    vol_R_hat = 0.0   # ∫ R̂ dV̂
    vol_balance_valid = False

    if source_info is not None and (source_info.get("has_recomb") or source_info.get("has_gen")):
        vol_balance_valid = True

        if source_info.get("has_recomb") and source_info["tau_hat"] > 0:
            # R̂ = (n̂ - n̂₀) / τ̂
            n0_hat = Nd / scaling.n_ref
            tau_hat = source_info["tau_hat"]
            R_hat_expr = (n_hat - n0_hat) / tau_hat
            R_form = fem.form(R_hat_expr * ufl.dx)
            vol_R_local = fem.assemble_scalar(R_form)
            vol_R_hat = comm.allreduce(vol_R_local, op=MPI.SUM)

        if source_info.get("has_gen") and source_info["G0_hat"] > 0:
            G0_hat = source_info["G0_hat"]
            alpha_hat_opt = source_info["alpha_hat"]
            x = ufl.SpatialCoordinate(msh)
            G_hat_expr = G0_hat * ufl.exp(-alpha_hat_opt * x[0])
            G_form = fem.form(G_hat_expr * ufl.dx)
            vol_G_local = fem.assemble_scalar(G_form)
            vol_G_hat = comm.allreduce(vol_G_local, op=MPI.SUM)

    # Volume balance metric
    I_net_hat = I_left_hat + I_right_hat
    S_net_hat = vol_G_hat - vol_R_hat
    if vol_balance_valid:
        eta_vol = abs(I_net_hat - S_net_hat) / max(abs(I_net_hat), abs(S_net_hat), 1e-30)
    else:
        eta_vol = 0.0

    # --- Peclet numbers ---
    Pe_device = abs(V_bias) / scaling.V_t
    Pe_cell = Pe_device / (2.0 * 20)  # V_hat / (2*nx), approximate

    # --- Scaling cross-check ---
    # J_scale = q μn n_ref Vt / L_norm
    # Expected Jn_hat ≈ (Nd/n_ref) × (V̂/Lx_hat) = Nd_hat × grad_phi_hat (drift-dominated)
    Nd_hat = Nd / scaling.n_ref
    V_hat = V_bias / scaling.V_t
    Jn_hat_expected = Nd_hat * V_hat / Lx_hat  # expected |Ĵn| if drift-only, 1D
    J_phys_expected = Jn_hat_expected * J_scale  # = q μn Nd V_bias / L_norm

    # Direct dimensional analytic
    J_dim_analytic = Q_E * mu_n * Nd * V_bias / (Lx_hat * L_norm)
    # Note: Lx_hat * L_norm = Lx (physical device thickness)

    if rank == 0:
        print("\n" + "=" * 70)
        print("BRICK 4a ACCEPTANCE TESTS: Dark biased resistor (G=R=0)")
        print("=" * 70)

        # Field diagnostics
        print(f"  --- Field diagnostics ---")
        print(f"  n̂:  [{n_hat_min:.6f}, {n_hat_max:.6f}]  "
              f"(physical: [{n_phys_min:.3e}, {n_phys_max:.3e}] m⁻³)")
        print(f"  φ̂:  [{phi_hat_min:.6f}, {phi_hat_max:.6f}]  "
              f"(physical: [{phi_hat_min*scaling.V_t*1e3:.3f}, "
              f"{phi_hat_max*scaling.V_t*1e3:.3f}] mV)")

        # Test 1: Positivity
        t1_pass = n_hat_min > 0
        print(f"\n  1. POSITIVITY: min(n̂) = {n_hat_min:.6e} "
              f"({'PASS' if t1_pass else 'FAIL *** n < 0!'})")

        # Test 2: Quasi-neutrality
        n_err = max(abs(n_hat_min - Nd_hat), abs(n_hat_max - Nd_hat)) / Nd_hat
        t2_pass = n_err < 0.05  # 5% for 3D with partial contacts
        print(f"  2. NEUTRALITY: |n̂-N̂d|/N̂d = {n_err:.6e} "
              f"({'PASS' if t2_pass else 'FAIL'} < 5%)")

        # Test 3: Current conservation (divergence-free)
        t3_pass = eta_div < 1e-2
        print(f"  3. CONSERVATION: ||∇·Ĵn||_L2 = {div_Jn_L2:.6e}, "
              f"||Ĵn||_L2 = {Jn_L2:.6e}")
        print(f"     η_div = {eta_div:.6e} "
              f"({'PASS' if t3_pass else 'FAIL'} < 1e-2)")

        # Test 4: Contact current balance
        t4_pass = eta_I < 1e-2
        print(f"  4. CONTACT BALANCE:")
        print(f"     Î_left  = {I_left_hat:.6e}  "
              f"(I = {I_left_phys:.6e} A·m/m²)")
        print(f"     Î_right = {I_right_hat:.6e}  "
              f"(I = {I_right_phys:.6e} A·m/m²)")
        print(f"     η_I = |I_L+I_R|/max = {eta_I:.6e} "
              f"({'PASS' if t4_pass else 'FAIL'} < 1e-2)")

        # Test 5: Scaling cross-check
        print(f"  5. SCALING:")
        print(f"     J_scale = q μn n_ref Vt / L_norm = {J_scale:.6e} A/m²")
        print(f"     Expected |Ĵn| (1D drift-only) = "
              f"N̂d × V̂/L̂x = {Nd_hat:.1f}×{V_hat:.1f}/{Lx_hat:.4e} "
              f"= {Jn_hat_expected:.2e}")
        print(f"     → J_phys = {J_phys_expected:.6e} A/m²  "
              f"(= q μn Nd V_bias / Lx)")
        print(f"     Dimensional check: q μn Nd V/L = {J_dim_analytic:.6e} A/m²")
        print(f"     (3D partial contacts → total I differs from J×A)")

        # Test 6: Pe
        print(f"  6. Pe_device = {Pe_device:.2f}, Pe_cell ≈ {Pe_cell:.4f}")

        # Test 7: Volume balance (Bricks 4b/4c only)
        t7_pass = True
        if vol_balance_valid:
            t7_pass = eta_vol < 1e-2
            print(f"  7. VOLUME BALANCE (photoconductor identity):")
            print(f"     ∫Ĝ dV̂ = {vol_G_hat:.6e}")
            print(f"     ∫R̂ dV̂ = {vol_R_hat:.6e}")
            print(f"     I_net (contacts) = {I_net_hat:.6e}")
            print(f"     S_net (∫(G-R)) = {S_net_hat:.6e}")
            print(f"     η_vol = |I_net - S_net|/max = {eta_vol:.6e} "
                  f"({'PASS' if t7_pass else 'FAIL'} < 1e-2)")

        all_pass = t1_pass and t2_pass and t3_pass and t4_pass and t7_pass
        print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
        print("=" * 70 + "\n")

    return {
        "J_scale": J_scale,
        "I_left_hat": I_left_hat, "I_right_hat": I_right_hat,
        "eta_div": eta_div, "eta_I": eta_I,
        "n_hat_min": n_hat_min, "n_hat_max": n_hat_max,
        "Pe_device": Pe_device, "Pe_cell": Pe_cell,
    }


def run_brick4a(args, comm, rank):
    """
    Brick 4a: Solve continuity with fixed φ̂ from Brick 3, dark case.

    1. Build mesh and Poisson system
    2. Solve Brick 3 for φ̂(x)
    3. Build linear continuity system with φ̂ fixed
    4. Solve for n̂(x)
    5. Run acceptance tests
    """
    if rank == 0:
        # Determine which brick level
        tau_n = getattr(args, 'tau_n', 0.0)
        G0 = getattr(args, 'G0', 0.0)
        if G0 > 0:
            brick_label = "4c: Illuminated photoconductor (G>0, R>0)"
        elif tau_n > 0:
            brick_label = "4b: Dark with recombination (R>0)"
        else:
            brick_label = "4a: Dark biased resistor (G=R=0)"
        print("\n" + "=" * 70)
        print(f"BRICK {brick_label}")
        print("=" * 70)

    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    L_norm = dims["L_norm"]
    L_ref = dims["L_ref"]
    alpha_eff = scaling.alpha * (L_ref / L_norm)**2
    ni_ratio = args.n_i / scaling.n_ref

    if rank == 0:
        print(f"  n_i = {args.n_i:.3e} m⁻³ (n_i/n_ref = {ni_ratio:.6e})")
        print(f"  Nd  = {args.Nd:.3e} m⁻³ (Nd/n_ref = {args.Nd/scaling.n_ref:.6e})")
        print(f"  V_bias = {args.V_bias} V")
        print(f"  mu_n = {args.mu_n} m²/Vs")
        tau_n_val = getattr(args, 'tau_n', 0.0)
        G0_val = getattr(args, 'G0', 0.0)
        alpha_val = getattr(args, 'alpha_opt', 0.0)
        if tau_n_val > 0:
            print(f"  tau_n = {tau_n_val:.3e} s")
        if G0_val > 0:
            print(f"  G0 = {G0_val:.3e} m⁻³s⁻¹")
            print(f"  alpha_opt = {alpha_val:.3e} m⁻¹")

    # Step 1: Solve Brick 3 (Poisson) for φ̂
    if rank == 0:
        print(f"\n  Step 1: Solving Brick 3 (Poisson) for φ̂...")

    V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
        msh, scaling, dims, args.V_bias, args.Nd, args.Na, args.n_i
    )

    try:
        info = solve_nonlinear_poisson(
            F_form, J_form, phi_hat, bcs, comm,
            phi_ref_hat_value=phi_ref_val, verbose=True
        )
    except PoissonSolveError as e:
        if rank == 0:
            print(f"  Brick 3 FAILED: {e}")
            print(f"  TIP: Try --n_i 1e17 for easier convergence")
        return None

    # Report φ̂ range (all ranks participate)
    n_own = V.dofmap.index_map.size_local
    phi_min = comm.allreduce(
        phi_hat.x.array[:n_own].min() if n_own > 0 else 1e30, op=MPI.MIN)
    phi_max = comm.allreduce(
        phi_hat.x.array[:n_own].max() if n_own > 0 else -1e30, op=MPI.MAX)
    if rank == 0:
        print(f"  φ̂ converged, range [{phi_min:.4f}, {phi_max:.4f}]")

    # Step 2: Build and solve continuity
    if rank == 0:
        # Peclet diagnostics (dimensional and dimensionless)
        V_hat = args.V_bias / scaling.V_t
        Lx_hat = dims["Lx_hat"]
        L_norm = dims["L_norm"]
        geom = DeviceGeometry()

        # DIMENSIONAL Pe: Pe_dim = |E| * L / V_t  (global drift/diffusion ratio)
        E_est = args.V_bias / geom.Lx  # V/m
        Pe_dim = E_est * geom.Lx / scaling.V_t
        # This equals V_bias / V_t = V_hat
        print(f"\n  Step 2: Solving continuity for n̂...")
        print(f"    V̂_bias = {V_hat:.2f}")
        print(f"    |E| ≈ {E_est:.2e} V/m  (= V_bias/L)")
        print(f"    Pe_device = V_bias/V_t = {Pe_dim:.2f} "
              f"({'drift-dominated' if Pe_dim > 1 else 'diffusion-dominated'})")

        # CELL-LEVEL Pe: Pe_cell = |∇̂φ̂| * h_hat / 2
        # |∇̂φ̂| ≈ V̂/(Lx/L_norm) in the x-direction
        grad_phi_hat = V_hat / Lx_hat  # dimensionless gradient
        h_hat = Lx_hat / args.nx       # dimensionless cell size in x
        Pe_cell = grad_phi_hat * h_hat / 2.0
        # Note: Pe_cell = (V_hat / Lx_hat) * (Lx_hat / nx) / 2 = V_hat / (2*nx)
        print(f"    |∇̂φ̂| ≈ {grad_phi_hat:.1f}, h_hat = {h_hat:.2e}")
        print(f"    Pe_cell = {Pe_cell:.4f} "
              f"(SUPG needed if >0.5, current: {'YES' if Pe_cell > 0.5 else 'no'})")

    mu_n = args.mu_n
    tau_n = getattr(args, 'tau_n', 0.0)
    G0 = getattr(args, 'G0', 0.0)
    alpha_opt = getattr(args, 'alpha_opt', 0.0)
    V_n, A, b, bcs_n, n_hat, source_info = build_continuity_linear(
        msh, scaling, dims, phi_hat, args.Nd, mu_n,
        tau_n=tau_n, G0=G0, alpha_opt=alpha_opt
    )

    if rank == 0 and (source_info["has_recomb"] or source_info["has_gen"]):
        print(f"    --- Source terms ---")
        if source_info["has_recomb"]:
            print(f"    τ̂ = {source_info['tau_hat']:.6e}  "
                  f"(τ_n = {tau_n:.3e} s)")
        if source_info["has_gen"]:
            print(f"    Ĝ₀ = {source_info['G0_hat']:.6e}  "
                  f"(G0 = {G0:.3e} m⁻³s⁻¹)")
            print(f"    α̂ = {source_info['alpha_hat']:.4f}  "
                  f"(α_opt = {alpha_opt:.3e} m⁻¹)")

    solve_continuity(A, b, n_hat, comm)

    # n-positivity check (all ranks)
    n_own_v = V_n.dofmap.index_map.size_local
    n_arr = n_hat.x.array[:n_own_v]
    n_min_local = n_arr.min() if n_own_v > 0 else 1e30
    n_min_global = comm.allreduce(n_min_local, op=MPI.MIN)
    if rank == 0:
        if n_min_global < 0:
            print(f"  *** WARNING: n̂ < 0 detected! min(n̂) = {n_min_global:.6e} ***")
            print(f"  *** This indicates numerical oscillation — need stronger stabilization ***")
        else:
            print(f"  n̂ > 0 everywhere: min(n̂) = {n_min_global:.6e} ✓")

    # Step 3: Acceptance tests
    if rank == 0:
        print(f"  Step 3: Running acceptance tests...")

    results = run_brick4a_tests(
        phi_hat, n_hat, scaling, dims, args.Nd, mu_n, args.V_bias, comm,
        tau_n=tau_n, G0=G0, alpha_opt=alpha_opt, source_info=source_info
    )

    return results


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
              f"{'phi_min':>10s}  {'phi_max':>10s}  "
              f"{'arg_min':>8s}  {'arg_max':>8s}  {'clamp%':>7s}  {'status':>6s}")
        print("-" * 85)

    all_ok = True
    for V_bias in V_array:
        try:
            V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
                msh, scaling, dims, V_bias, args.Nd, args.Na, args.n_i
            )
            info = solve_nonlinear_poisson(
                F_form, J_form, phi_hat, bcs, comm,
                phi_ref_hat_value=phi_ref_val, verbose=False
            )
            total_clamp = info['pct_clamp_hi'] + info['pct_clamp_lo']
            if rank == 0:
                print(f"{V_bias:>8.3f}  {info['newton_iters']:>7d}  "
                      f"{info['solve_time']:>7.2f}  "
                      f"{info['phi_hat_min']:>10.4f}  {info['phi_hat_max']:>10.4f}  "
                      f"{info['exp_arg_min']:>8.2f}  {info['exp_arg_max']:>8.2f}  "
                      f"{total_clamp:>6.2f}%  "
                      f"{'OK' if info['converged'] else 'FAIL':>6s}")
            if not info["converged"]:
                all_ok = False
        except PoissonSolveError as e:
            if rank == 0:
                print(f"{V_bias:>8.3f}  {'---':>7s}  {'---':>7s}  "
                      f"{'---':>10s}  {'---':>10s}  "
                      f"{'---':>8s}  {'---':>8s}  {'---':>7s}  {'DIV':>6s}")
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

def run_nd_continuation(args, comm, rank):
    """
    Brick 3 continuation: ramp Nd from 1e18 to target in log steps.

    Each step rebuilds the nonlinear system at the new Nd but uses the
    previous converged solution as the initial guess, giving Newton a
    much better starting point for stiff regimes.
    """
    import math

    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)

    Nd_target = args.Nd
    if Nd_target <= 0:
        if rank == 0:
            print("ERROR: --Nd must be > 0 for continuation")
        return

    # Build continuation schedule: log-spaced from 1e18 to Nd_target
    Nd_start = 1e18
    if Nd_target <= Nd_start:
        Nd_schedule = [Nd_target]
    else:
        n_steps = max(2, int(math.log10(Nd_target / Nd_start) * 3))  # ~3 steps per decade
        Nd_schedule = np.logspace(math.log10(Nd_start), math.log10(Nd_target), n_steps)
    Nd_schedule = list(Nd_schedule)

    if rank == 0:
        print("\n" + "="*70)
        print("BRICK 3: Nd CONTINUATION")
        print("="*70)
        print(scaling.summary())
        print(f"  V_bias = {args.V_bias} V")
        print(f"  Na     = {args.Na:.3e} m^-3")
        print(f"  n_i    = {args.n_i:.3e} m^-3")
        print(f"  Nd schedule ({len(Nd_schedule)} steps):")
        for i, Nd in enumerate(Nd_schedule):
            print(f"    [{i}] Nd = {Nd:.3e} m^-3")
        print()

    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    # Initial guess: start from linear Poisson solution at first Nd
    prev_phi_array = None

    if rank == 0:
        print(f"{'Step':<6} {'Nd':>12} {'Nd/n_ref':>12} {'||F0||':>12} "
              f"{'Iters':>6} {'||F_final||':>12} {'Reason':>12}")
        print("-" * 78)

    for step_idx, Nd_val in enumerate(Nd_schedule):
        # Build system at this Nd
        V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
            msh, scaling, dims, args.V_bias, Nd_val, args.Na, args.n_i
        )

        # Override initial guess with previous converged solution
        if prev_phi_array is not None:
            n_own = V.dofmap.index_map.size_local
            phi_hat.x.array[:n_own] = prev_phi_array[:n_own]
            phi_hat.x.scatter_forward()

        # Skip Jacobian test for continuation (already validated)
        try:
            info = solve_nonlinear_poisson(
                F_form, J_form, phi_hat, bcs, comm,
                phi_ref_hat_value=phi_ref_val, verbose=(rank == 0),
            )
            n_own = V.dofmap.index_map.size_local
            prev_phi_array = phi_hat.x.array[:n_own].copy()

            if rank == 0:
                print(f"{step_idx:<6} {Nd_val:>12.3e} {Nd_val/scaling.n_ref:>12.3e} "
                      f"{'--':>12} {info['newton_iters']:>6} "
                      f"{info['fnorm']:>12.3e} {info['reason_str']:>12}")

        except PoissonSolveError as e:
            if rank == 0:
                print(f"{step_idx:<6} {Nd_val:>12.3e} {Nd_val/scaling.n_ref:>12.3e} "
                      f"{'--':>12} {'FAIL':>6} {'--':>12} {'DIVERGED':>12}")
                print(f"  Continuation stopped at Nd = {Nd_val:.3e}: {e}")
            return

    if rank == 0:
        print("\n  Continuation COMPLETE — all Nd steps converged.")


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
    parser.add_argument("--nd_continuation", action="store_true",
                        help="Brick 3: Ramp Nd from 1e18 to --Nd with continuation")
    parser.add_argument("--n_i", type=float, default=N_I_DEFAULT,
                        help="Intrinsic carrier density [m^-3]")

    # Brick 4a: Continuity with fixed phi
    parser.add_argument("--brick4a", action="store_true",
                        help="Brick 4a: Electron continuity (phi fixed, dark)")
    parser.add_argument("--mu_n", type=float, default=1.0,
                        help="Electron mobility [m²/Vs]")
    # Brick 4b/4c: Recombination and generation
    parser.add_argument("--tau_n", type=float, default=0.0,
                        help="Electron lifetime [s] (0=off, Brick 4b)")
    parser.add_argument("--G0", type=float, default=0.0,
                        help="Peak generation rate [m⁻³s⁻¹] (0=off, Brick 4c)")
    parser.add_argument("--alpha_opt", type=float, default=0.0,
                        help="Optical absorption coefficient [m⁻¹] (Brick 4c)")
    # Brick 5: Two-carrier photoconductor
    parser.add_argument("--brick5", action="store_true",
                        help="Brick 5: Two-carrier photoconductor (phi fixed, Gummel)")
    parser.add_argument("--mu_p", type=float, default=0.01,
                        help="Hole mobility [m²/Vs]")
    parser.add_argument("--gummel_max", type=int, default=30,
                        help="Max Gummel iterations")
    parser.add_argument("--gummel_tol", type=float, default=1e-6,
                        help="Gummel convergence tolerance")

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

    # ── Brick 5: Two-carrier photoconductor ──
    if args.brick5:
        run_brick5(args, comm, rank)
        sys.exit(0)

    # ── Brick 4a: Continuity with fixed phi ──
    if args.brick4a:
        run_brick4a(args, comm, rank)
        sys.exit(0)

    # ── Brick 3: Nonlinear single point ──
    if args.nonlinear:
        run_nonlinear_poisson(args, comm, rank)
        sys.exit(0)

    # ── Brick 3: Nd continuation ──
    if args.nd_continuation:
        run_nd_continuation(args, comm, rank)
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


# ════════════════════════════════════════════════════════════════════════
#  BRICK 5: Two-carrier photoconductor (φ fixed, n + p, Gummel)
# ════════════════════════════════════════════════════════════════════════
#
#  Physics:
#    ∇·Jn = q(G - R)          electron continuity
#    ∇·Jp = -q(G - R)         hole continuity
#    Jn = -qμn n ∇φ + qDn ∇n
#    Jp = -qμp p ∇φ - qDp ∇p
#    R = (np - ni²) / (τ(n + p + 2ni))   simplified SRH
#    G(x) = G0 exp(-α x)
#
#  φ is FIXED from Brick 3.
#
#  Nondimensionalization (n̂=n/nref, p̂=p/nref, φ̂=φ/Vt):
#    Ĵn = -n̂ ∇̂φ̂ + ∇̂n̂
#    Ĵp = -p̂ ∇̂φ̂ - ∇̂p̂
#    R̂ = (n̂ p̂ - n̂i²) / (τ̂ (n̂ + p̂ + 2n̂i))
#    Ĝ = G0_hat exp(-α̂ x̂)
#
#  Solver: Gummel iteration (segregated):
#    1. Solve continuity for n̂ with p̂ fixed → compute R̂(n̂, p̂_old)
#    2. Solve continuity for p̂ with n̂ fixed → compute R̂(n̂_new, p̂)
#    3. Repeat until convergence
#
#  BCs (ohmic contacts):
#    n̂ = Nd/nref  at contacts (majority carrier)
#    p̂ = ni²/(Nd × nref) at contacts (minority carrier, equilibrium)
#

def _build_carrier_system(msh, scaling, dims, phi_hat, carrier_type,
                          n_hat_fn, p_hat_fn, Nd, ni,
                          mu, tau_n, G0, alpha_opt):
    """
    Build SUPG-stabilized continuity for one carrier (n or p) with the
    other carrier FIXED (for Gummel iteration).

    carrier_type: "electron" or "hole"

    For electrons:
        ∇̂·Ĵn = Ĝ - R̂
        Weak: ∫ ∇̂n̂·∇̂w - n̂(∇̂φ̂·∇̂w) dx̂ = ∫ (Ĝ - R̂) w dx̂
        R̂ linearized with p̂ fixed:
            R̂ ≈ (n̂ p̂_old - n̂i²) / (τ̂(n̂ + p̂_old + 2n̂i))
            ≈ p̂_old/(τ̂(n̂ + p̂_old + 2n̂i)) × n̂  - n̂i²/(τ̂(n̂ + p̂_old + 2n̂i))
            For linearization, evaluate τ̂(n̂_old + p̂_old + 2n̂i) at previous values.
            Then: R̂ ≈ r_coeff × n̂ - r_const
            where r_coeff = p̂_old / (τ̂ denom), r_const = n̂i² / (τ̂ denom)

    For holes:
        ∇̂·Ĵp = -(Ĝ - R̂)
        Ĵp = -p̂ ∇̂φ̂ - ∇̂p̂
        Weak: ∫ (p̂ ∇̂φ̂ + ∇̂p̂)·∇̂w dx̂ = ∫ (Ĝ - R̂) w dx̂

    Returns (A, b, bcs) ready for KSP solve into the carrier Function.
    """
    V = phi_hat.function_space
    u = ufl.TrialFunction(V)
    w = ufl.TestFunction(V)

    L_norm = dims["L_norm"]
    Nd_hat = Nd / scaling.n_ref
    ni_hat = ni / scaling.n_ref

    v = ufl.grad(phi_hat)
    v_mag = ufl.sqrt(ufl.inner(v, v) + 1e-30)

    # --- SUPG parameters ---
    h = ufl.CellDiameter(msh)
    Pe_local = v_mag * h / 2.0
    xi = ufl.min_value(Pe_local / 3.0, 1.0)
    tau_supg = (h / (2.0 * v_mag)) * xi
    w_supg = tau_supg * ufl.inner(v, ufl.grad(w))

    if carrier_type == "electron":
        # Ĵn = -n̂ ∇̂φ̂ + ∇̂n̂  → advection v = ∇̂φ̂
        a_gal = (ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
                 - u * ufl.inner(v, ufl.grad(w)) * ufl.dx)
        R_trial_adv = ufl.inner(v, ufl.grad(u))
    else:  # hole
        # Ĵp = -p̂ ∇̂φ̂ - ∇̂p̂
        # After IBP of ∇·Ĵp:
        # ∫ (p̂ ∇̂φ̂ + ∇̂p̂)·∇̂w dx̂
        # = ∫ ∇̂p̂·∇̂w dx̂ + ∫ p̂ (∇̂φ̂·∇̂w) dx̂
        # Advection velocity for holes: -∇̂φ̂ (holes drift opposite to electrons)
        # Actually let me re-derive:
        # ∇·Jp = -q(G-R)
        # Jp = -qμp p ∇φ - qDp ∇p
        # Dimensionless: Ĵp_hat = -p̂ ∇̂φ̂ - ∇̂p̂
        # ∇̂·Ĵp = -(Ĝ - R̂)
        # Weak: ∫ (∇̂·Ĵp) w dx̂ = -∫ (Ĝ - R̂) w dx̂
        # IBP: -∫ Ĵp · ∇̂w dx̂ = -∫ (Ĝ - R̂) w dx̂
        #       ∫ (p̂ ∇̂φ̂ + ∇̂p̂) · ∇̂w dx̂ = ∫ (Ĝ - R̂) w dx̂
        # So: a(p̂, w) = ∫ ∇̂p̂·∇̂w dx̂ + ∫ p̂ (∇̂φ̂·∇̂w) dx̂
        a_gal = (ufl.inner(ufl.grad(u), ufl.grad(w)) * ufl.dx
                 + u * ufl.inner(v, ufl.grad(w)) * ufl.dx)
        # Hole advection velocity = -∇̂φ̂ (opposite direction)
        v_hole = -v
        w_supg = tau_supg * ufl.inner(v_hole, ufl.grad(w))
        R_trial_adv = ufl.inner(v_hole, ufl.grad(u))

    a_supg = w_supg * R_trial_adv * ufl.dx
    a_form = a_gal + a_supg

    # --- Source terms: linearized R̂ and Ĝ ---
    L_form = fem.Constant(msh, PETSc.ScalarType(0.0)) * w * ufl.dx

    tau_hat = tau_n * mu * scaling.V_t / L_norm**2 if tau_n > 0 else 1e30

    if tau_n > 0:
        # R̂ = (n̂ p̂ - n̂i²) / (τ̂ (n̂ + p̂ + 2n̂i))
        # Linearize: evaluate denominator at previous iteration values
        # For electron solve: n is unknown, p is fixed → p_hat_fn
        # R̂ ≈ [p̂_old × n̂ - n̂i²] / (τ̂ × denom_old)
        # where denom_old = n̂_old + p̂_old + 2n̂i
        # This gives: R̂ = r_coeff × n̂ - r_const
        #   r_coeff = p̂_old / (τ̂ × denom_old)
        #   r_const = n̂i² / (τ̂ × denom_old)

        if carrier_type == "electron":
            other = p_hat_fn
            self_old = n_hat_fn
        else:
            other = n_hat_fn
            self_old = p_hat_fn

        # denom is evaluated at old values (both carriers)
        denom = self_old + other + 2.0 * ni_hat
        # Avoid division by zero
        denom_safe = ufl.max_value(denom, 1e-30)

        r_coeff = other / (tau_hat * denom_safe)
        r_const = ni_hat**2 / (tau_hat * denom_safe)

        # For electron: ∇·Jn = G - R → source = G - R
        # R = r_coeff × n̂ - r_const
        # source = G - r_coeff × n̂ + r_const
        # LHS gets: + r_coeff × ∫ u w dx  (reaction term)
        # RHS gets: + r_const × ∫ w dx + ∫ Ĝ w dx

        a_form += r_coeff * u * w * ufl.dx
        # SUPG consistent reaction
        a_form += r_coeff * w_supg * u * ufl.dx

        L_form += r_const * w * ufl.dx
        L_form += r_const * w_supg * ufl.dx

    if G0 > 0:
        G0_hat = G0 * L_norm**2 / (scaling.n_ref * mu * scaling.V_t)
        alpha_hat_opt = alpha_opt * L_norm
        x = ufl.SpatialCoordinate(msh)
        G_hat = G0_hat * ufl.exp(-alpha_hat_opt * x[0])

        L_form += G_hat * w * ufl.dx
        L_form += G_hat * w_supg * ufl.dx

    # --- BCs ---
    Lx_hat = dims["Lx_hat"]
    Ly_hat = dims["Ly_hat"]
    y_left_max = 1000e-6 / L_norm
    y_right_min = 2000e-6 / L_norm
    tol_bc = 1e-8

    if carrier_type == "electron":
        bc_val = Nd_hat  # majority carrier
    else:
        # Minority carrier: p_eq = ni²/Nd (mass action law)
        bc_val = ni_hat**2 / Nd_hat if Nd_hat > 0 else ni_hat

    dofs_left = _locate_contact_dofs(V, msh, 0.0, 0.0, y_left_max, tol_bc)
    bc_left = fem.dirichletbc(PETSc.ScalarType(bc_val), dofs_left, V)

    dofs_right = _locate_contact_dofs(V, msh, Lx_hat, y_right_min, Ly_hat, tol_bc)
    bc_right = fem.dirichletbc(PETSc.ScalarType(bc_val), dofs_right, V)

    bcs = [bc_left, bc_right]

    # Compile and assemble
    a_compiled = fem.form(a_form)
    L_compiled = fem.form(L_form)

    A = fem_petsc.assemble_matrix(a_compiled, bcs=bcs)
    A.assemble()

    b_vec = fem_petsc.assemble_vector(L_compiled)
    fem_petsc.apply_lifting(b_vec, [a_compiled], bcs=[bcs])
    b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem_petsc.set_bc(b_vec, bcs)

    return A, b_vec, bcs


def run_brick5(args, comm, rank):
    """
    Brick 5: Two-carrier photoconductor (φ fixed, Gummel iteration).

    Solves electron + hole continuity with fixed φ from Brick 3.
    Uses Gummel iteration: alternate n and p solves until self-consistent.
    """
    if rank == 0:
        print("\n" + "=" * 70)
        print("BRICK 5: Two-carrier photoconductor (φ fixed, Gummel)")
        print("=" * 70)

    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)

    L_norm = dims["L_norm"]
    Nd_hat = args.Nd / scaling.n_ref
    ni_hat = args.n_i / scaling.n_ref
    mu_n = args.mu_n
    mu_p = getattr(args, 'mu_p', 0.01)  # hole mobility, typically << mu_n
    tau_n = getattr(args, 'tau_n', 1e-6)
    G0 = getattr(args, 'G0', 0.0)
    alpha_opt = getattr(args, 'alpha_opt', 0.0)
    gummel_max = getattr(args, 'gummel_max', 30)
    gummel_tol = getattr(args, 'gummel_tol', 1e-6)

    if rank == 0:
        print(f"  Nd = {args.Nd:.3e}, ni = {args.n_i:.3e}")
        print(f"  mu_n = {mu_n}, mu_p = {mu_p}")
        print(f"  tau_n = {tau_n:.3e} s")
        print(f"  G0 = {G0:.3e} m⁻³s⁻¹, alpha = {alpha_opt:.3e} m⁻¹")
        print(f"  V_bias = {args.V_bias} V")
        p_eq = args.n_i**2 / args.Nd
        print(f"  p_eq = ni²/Nd = {p_eq:.3e} m⁻³")
        print(f"  p̂_eq = {ni_hat**2/Nd_hat:.3e}")

    # Step 1: Solve Brick 3 (Poisson) for φ̂
    if rank == 0:
        print(f"\n  Step 1: Solving Poisson for φ̂...")

    V, F_form, J_form, bcs_phi, phi_hat, phi_ref_val = build_nonlinear_poisson(
        msh, scaling, dims, args.V_bias, args.Nd, args.Na, args.n_i
    )
    try:
        solve_nonlinear_poisson(
            F_form, J_form, phi_hat, bcs_phi, comm,
            phi_ref_hat_value=phi_ref_val, verbose=True
        )
    except PoissonSolveError as e:
        if rank == 0:
            print(f"  Poisson FAILED: {e}")
        return None

    # Step 2: Initialize carrier densities
    n_hat = fem.Function(V, name="n_hat")
    p_hat = fem.Function(V, name="p_hat")
    n_hat.x.array[:] = Nd_hat
    p_hat.x.array[:] = ni_hat**2 / Nd_hat  # minority equilibrium
    n_hat.x.scatter_forward()
    p_hat.x.scatter_forward()

    if rank == 0:
        print(f"\n  Step 2: Gummel iteration (max {gummel_max} iters, tol {gummel_tol})...")

    # Step 3: Gummel iteration
    for gummel_iter in range(gummel_max):
        # Save previous for convergence check
        n_prev = n_hat.x.array.copy()
        p_prev = p_hat.x.array.copy()

        # Solve electron continuity (n) with p fixed
        A_n, b_n, bcs_n = _build_carrier_system(
            msh, scaling, dims, phi_hat, "electron",
            n_hat, p_hat, args.Nd, args.n_i,
            mu_n, tau_n, G0, alpha_opt
        )
        ksp_n = PETSc.KSP().create(comm)
        ksp_n.setOptionsPrefix("gum_n_")
        ksp_n.setOperators(A_n)
        ksp_n.setType("gmres")
        ksp_n.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
        ksp_n.setGMRESRestart(100)
        pc_n = ksp_n.getPC()
        if comm.size > 1:
            pc_n.setType("asm")
            ksp_n.setUp()
            try:
                for sub_ksp in pc_n.getASMSubKSP():
                    sub_ksp.setType("preonly")
                    sub_ksp.getPC().setType("ilu")
            except Exception:
                pass
        else:
            pc_n.setType("ilu")
        ksp_n.setFromOptions()
        ksp_n.solve(b_n, n_hat.x.petsc_vec)
        n_hat.x.scatter_forward()
        n_its = ksp_n.getIterationNumber()
        ksp_n.destroy()
        A_n.destroy()
        b_n.destroy()

        # Solve hole continuity (p) with n fixed
        A_p, b_p, bcs_p = _build_carrier_system(
            msh, scaling, dims, phi_hat, "hole",
            n_hat, p_hat, args.Nd, args.n_i,
            mu_p, tau_n, G0, alpha_opt
        )
        ksp_p = PETSc.KSP().create(comm)
        ksp_p.setOptionsPrefix("gum_p_")
        ksp_p.setOperators(A_p)
        ksp_p.setType("gmres")
        ksp_p.setTolerances(rtol=1e-8, atol=1e-12, max_it=2000)
        ksp_p.setGMRESRestart(100)
        pc_p = ksp_p.getPC()
        if comm.size > 1:
            pc_p.setType("asm")
            ksp_p.setUp()
            try:
                for sub_ksp in pc_p.getASMSubKSP():
                    sub_ksp.setType("preonly")
                    sub_ksp.getPC().setType("ilu")
            except Exception:
                pass
        else:
            pc_p.setType("ilu")
        ksp_p.setFromOptions()
        ksp_p.solve(b_p, p_hat.x.petsc_vec)
        p_hat.x.scatter_forward()
        p_its = ksp_p.getIterationNumber()
        ksp_p.destroy()
        A_p.destroy()
        b_p.destroy()

        # Convergence check
        n_own = V.dofmap.index_map.size_local
        dn = np.linalg.norm(n_hat.x.array[:n_own] - n_prev[:n_own])
        dp = np.linalg.norm(p_hat.x.array[:n_own] - p_prev[:n_own])
        nn = np.linalg.norm(n_hat.x.array[:n_own])
        pn = np.linalg.norm(p_hat.x.array[:n_own])
        dn_rel = comm.allreduce(dn, op=MPI.SUM) / max(comm.allreduce(nn, op=MPI.SUM), 1e-30)
        dp_rel = comm.allreduce(dp, op=MPI.SUM) / max(comm.allreduce(pn, op=MPI.SUM), 1e-30)
        update = max(dn_rel, dp_rel)

        if rank == 0:
            print(f"    Gummel {gummel_iter:3d}: KSP(n)={n_its:4d} KSP(p)={p_its:4d} "
                  f"Δn/n={dn_rel:.3e} Δp/p={dp_rel:.3e}", flush=True)

        if update < gummel_tol:
            if rank == 0:
                print(f"  Gummel CONVERGED at iter {gummel_iter} (update={update:.3e})")
            break
    else:
        if rank == 0:
            print(f"  Gummel did NOT converge after {gummel_max} iters (update={update:.3e})")

    # Step 4: Diagnostics
    n_min = comm.allreduce(
        n_hat.x.array[:n_own].min() if n_own > 0 else 1e30, op=MPI.MIN)
    n_max = comm.allreduce(
        n_hat.x.array[:n_own].max() if n_own > 0 else -1e30, op=MPI.MAX)
    p_min = comm.allreduce(
        p_hat.x.array[:n_own].min() if n_own > 0 else 1e30, op=MPI.MIN)
    p_max = comm.allreduce(
        p_hat.x.array[:n_own].max() if n_own > 0 else -1e30, op=MPI.MAX)

    if rank == 0:
        print(f"\n  --- Carrier diagnostics ---")
        print(f"  n̂: [{n_min:.6e}, {n_max:.6e}]  (phys: [{n_min*scaling.n_ref:.3e}, {n_max*scaling.n_ref:.3e}] m⁻³)")
        print(f"  p̂: [{p_min:.6e}, {p_max:.6e}]  (phys: [{p_min*scaling.n_ref:.3e}, {p_max*scaling.n_ref:.3e}] m⁻³)")
        t_pos = n_min > 0 and p_min > 0
        print(f"  Positivity: {'PASS' if t_pos else 'FAIL'}")

    # Step 5: Photoconductor identity
    # Total current = ∫_contacts (Jn + Jp)·n̂ dA = q ∫(G - R) dV
    Jn_expr = -n_hat * ufl.grad(phi_hat) + ufl.grad(n_hat)
    Jp_expr = -p_hat * ufl.grad(phi_hat) - ufl.grad(p_hat)
    J_total = Jn_expr + Jp_expr

    # Contact integration (reuse facet marking)
    tdim = msh.topology.dim
    fdim = tdim - 1
    Lx_hat = dims["Lx_hat"]
    Ly_hat = dims["Ly_hat"]
    y_left_max = 1000e-6 / L_norm
    y_right_min = 2000e-6 / L_norm
    tol_bc = 1e-8

    def left_m(x):
        return (np.abs(x[0]) < tol_bc) & (x[1] >= -tol_bc) & (x[1] <= y_left_max + tol_bc)
    def right_m(x):
        return (np.abs(x[0] - Lx_hat) < tol_bc) & (x[1] >= y_right_min - tol_bc) & (x[1] <= Ly_hat + tol_bc)

    lf = dmesh.locate_entities_boundary(msh, fdim, left_m)
    rf = dmesh.locate_entities_boundary(msh, fdim, right_m)
    af = np.concatenate([lf, rf])
    at = np.concatenate([np.full_like(lf, 1), np.full_like(rf, 2)])
    si = np.argsort(af)
    ft = dmesh.meshtags(msh, fdim, af[si], at[si])
    ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
    nv = ufl.FacetNormal(msh)

    I_left_form = fem.form(ufl.inner(J_total, nv) * ds(1))
    I_right_form = fem.form(ufl.inner(J_total, nv) * ds(2))
    I_left = comm.allreduce(fem.assemble_scalar(I_left_form), op=MPI.SUM)
    I_right = comm.allreduce(fem.assemble_scalar(I_right_form), op=MPI.SUM)

    # Volume integrals
    x = ufl.SpatialCoordinate(msh)
    vol_G = 0.0
    vol_R = 0.0

    if G0 > 0:
        G0_hat = G0 * L_norm**2 / (scaling.n_ref * mu_n * scaling.V_t)
        alpha_hat_opt = alpha_opt * L_norm
        G_hat = G0_hat * ufl.exp(-alpha_hat_opt * x[0])
        vol_G = comm.allreduce(fem.assemble_scalar(fem.form(G_hat * ufl.dx)), op=MPI.SUM)

    if tau_n > 0:
        tau_hat_n = tau_n * mu_n * scaling.V_t / L_norm**2
        R_hat = (n_hat * p_hat - ni_hat**2) / (tau_hat_n * (n_hat + p_hat + 2*ni_hat))
        vol_R = comm.allreduce(fem.assemble_scalar(fem.form(R_hat * ufl.dx)), op=MPI.SUM)

    I_net = I_left + I_right
    S_net = vol_G - vol_R
    eta_vol = abs(I_net - S_net) / max(abs(I_net), abs(S_net), 1e-30)

    # Physical scale
    J_scale = Q_E * mu_n * scaling.n_ref * scaling.V_t / L_norm

    if rank == 0:
        print(f"\n  --- Photoconductor identity ---")
        print(f"  Î_left  = {I_left:.6e}")
        print(f"  Î_right = {I_right:.6e}")
        print(f"  I_net   = {I_net:.6e}")
        print(f"  ∫Ĝ dV̂  = {vol_G:.6e}")
        print(f"  ∫R̂ dV̂  = {vol_R:.6e}")
        print(f"  S_net   = {S_net:.6e}")
        print(f"  η_vol   = {eta_vol:.6e} ({'PASS' if eta_vol < 0.05 else 'FAIL'} < 5%)")
        print(f"  J_scale = {J_scale:.6e} A/m²")
        print("=" * 70 + "\n")

    return {
        "I_left": I_left, "I_right": I_right,
        "vol_G": vol_G, "vol_R": vol_R, "eta_vol": eta_vol,
        "n_min": n_min, "n_max": n_max,
        "p_min": p_min, "p_max": p_max,
    }


if __name__ == "__main__":
    main()
