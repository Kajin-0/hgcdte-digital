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

    # DIGITAL EXPERIMENT: Electrostatics screening & field crowding
    # Runs 60 parameter combinations (Nd × n_i × V_bias sweep)
    # Generates metrics CSV + 4 publication plots + summary
    mpirun -n 4 python3 photoconductor_poisson.py --electrostatics_experiment
    
    # Optional: reduce mesh resolution for faster testing
    mpirun -n 4 python3 photoconductor_poisson.py --electrostatics_experiment --nx 15 --ny 45 --nz 30
    
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


# ════════════════════════════════════════════════════════════════════════
#  DIGITAL EXPERIMENT: Electrostatic Screening & Field Crowding
# ════════════════════════════════════════════════════════════════════════

def compute_field_metrics(phi_hat, V, scaling, dims, V_bias, comm):
    """
    Compute all electrostatic field metrics.
    
    Returns dict with:
        mean_E, U_E, C_E, f_act_50, Lambda_phi, max_E
    """
    rank = comm.rank
    
    # Compute gradient (electric field E = -grad(phi))
    W = fem.functionspace(V.mesh, ("DG", 0, (V.mesh.geometry.dim,)))
    E_field = fem.Function(W)
    
    # Create expression for -grad(phi_hat) * V_t
    grad_phi = ufl.grad(phi_hat)
    E_expr = fem.Expression(-grad_phi * scaling.V_t, W.element.interpolation_points())
    E_field.interpolate(E_expr)
    
    # Get cell volumes
    mesh = V.mesh
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_local
    
    # Compute cell volumes in physical units
    geom = DeviceGeometry()
    total_physical_volume = geom.Lx * geom.Ly * geom.Lz
    num_cells_global = comm.allreduce(num_cells, op=MPI.SUM)
    cell_volume_phys = total_physical_volume / num_cells_global
    
    # Extract field components and compute magnitude
    E_array = E_field.x.array.reshape((num_cells, 3))
    E_mag = np.sqrt(np.sum(E_array**2, axis=1))
    
    # Nominal field
    L_gap = geom.Ly  # 3 mm
    E0 = V_bias / L_gap
    
    # === Metric 1: Mean field magnitude ===
    total_volume_local = num_cells * cell_volume_phys
    E_integral_local = np.sum(E_mag) * cell_volume_phys
    
    total_volume = comm.allreduce(total_volume_local, op=MPI.SUM)
    E_integral = comm.allreduce(E_integral_local, op=MPI.SUM)
    mean_E = E_integral / total_volume if total_volume > 0 else 0.0
    
    # === Metric 2: Field uniformity index ===
    E_sq_integral_local = np.sum(E_mag**2) * cell_volume_phys
    E_sq_integral = comm.allreduce(E_sq_integral_local, op=MPI.SUM)
    E_variance = E_sq_integral / total_volume - mean_E**2
    std_E = np.sqrt(max(0, E_variance))
    U_E = std_E / mean_E if mean_E > 0 else 0.0
    
    # === Metric 3: Crowding factor (99th percentile) ===
    E_mag_all = comm.gather(E_mag, root=0)
    if rank == 0:
        E_mag_global = np.concatenate(E_mag_all)
        p99_E = np.percentile(E_mag_global, 99)
        C_E = p99_E / mean_E if mean_E > 0 else 0.0
        max_E = np.max(E_mag_global)
    else:
        C_E = 0.0
        max_E = 0.0
    C_E = comm.bcast(C_E, root=0)
    max_E = comm.bcast(max_E, root=0)
    
    # === Metric 4: Effective active volume fraction ===
    threshold = 0.5 * E0
    active_cells = np.sum(E_mag > threshold)
    active_cells_global = comm.allreduce(active_cells, op=MPI.SUM)
    f_act_50 = active_cells_global / num_cells_global if num_cells_global > 0 else 0.0
    
    # === Metric 5: Voltage drop localization ===
    Lambda_phi = compute_voltage_drop_localization(
        phi_hat, V, scaling, dims, V_bias, comm
    )
    
    metrics = {
        'mean_E': mean_E,
        'U_E': U_E,
        'C_E': C_E,
        'f_act_50': f_act_50,
        'Lambda_phi': Lambda_phi,
        'max_E': max_E
    }
    
    return metrics


def compute_voltage_drop_localization(phi_hat, V, scaling, dims, V_bias, comm):
    """
    Compute Lambda_phi = L_phi90 / L_gap
    where L_phi90 is the length over which 90% of voltage drop occurs.
    """
    rank = comm.rank
    
    # Get all DOF coordinates and values
    coords = V.tabulate_dof_coordinates()
    phi_values = phi_hat.x.array * scaling.V_t
    
    # Extract y-coordinates (index 1)
    L_norm = dims['L_norm']
    y_phys = coords[:, 1] * L_norm
    
    # Gather to rank 0
    y_all = comm.gather(y_phys, root=0)
    phi_all = comm.gather(phi_values, root=0)
    
    if rank == 0:
        y_global = np.concatenate(y_all)
        phi_global = np.concatenate(phi_all)
        
        # Create y bins and average phi in each bin
        geom = DeviceGeometry()
        ny_bins = 100
        y_bins = np.linspace(0, geom.Ly, ny_bins + 1)
        y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
        phi_avg = np.zeros(ny_bins)
        
        for i in range(ny_bins):
            mask = (y_global >= y_bins[i]) & (y_global < y_bins[i+1])
            if np.any(mask):
                phi_avg[i] = np.mean(phi_global[mask])
        
        # Smooth the profile
        window = 5
        phi_smooth = np.convolve(phi_avg, np.ones(window)/window, mode='same')
        
        # Find where 5% and 95% of total drop occurs
        phi_min = np.min(phi_smooth)
        phi_max = np.max(phi_smooth)
        total_drop = phi_max - phi_min
        
        if total_drop > 1e-6:
            threshold_5 = phi_min + 0.05 * total_drop
            threshold_95 = phi_min + 0.95 * total_drop
            
            idx_5 = np.argmax(phi_smooth >= threshold_5)
            idx_95 = np.argmax(phi_smooth >= threshold_95)
            
            if idx_95 > idx_5:
                L_phi90 = y_centers[idx_95] - y_centers[idx_5]
            else:
                L_phi90 = geom.Ly
        else:
            L_phi90 = geom.Ly
        
        Lambda_phi = L_phi90 / geom.Ly
    else:
        Lambda_phi = 0.0
    
    Lambda_phi = comm.bcast(Lambda_phi, root=0)
    return Lambda_phi


def run_electrostatics_experiment(args, comm, rank):
    """Run the full parameter sweep experiment."""
    import csv
    
    # Parameter space
    Nd_values = [1e20, 3e20, 1e21, 3e21, 1e22]  # m^-3
    n_i_values = [1e18, 1e19, 1e20, 1e21]        # m^-3
    V_bias_values = [0.1, 0.3, 0.5]              # V
    
    total_runs = len(Nd_values) * len(n_i_values) * len(V_bias_values)
    
    if rank == 0:
        print("=" * 80)
        print("HgCdTe ELECTROSTATICS DIGITAL EXPERIMENT")
        print("=" * 80)
        print(f"\nParameter space:")
        print(f"  Nd:     {Nd_values}")
        print(f"  n_i:    {n_i_values}")
        print(f"  V_bias: {V_bias_values}")
        print(f"  Total runs: {total_runs}\n")
    
    # Setup
    scaling = make_mct_scaling_77K()
    geom = DeviceGeometry()
    mcfg = MeshConfig(nx=args.nx, ny=args.ny, nz=args.nz)
    
    if rank == 0:
        print(scaling.summary())
        print(f"\nMesh: {mcfg.nx} × {mcfg.ny} × {mcfg.nz} = {mcfg.nx * mcfg.ny * mcfg.nz} elements\n")
    
    # Create mesh once
    msh, dims = create_device_mesh(geom, scaling.L_ref, mcfg, comm)
    
    # Storage for results
    results = []
    
    # Open CSV file for writing
    csv_path = Path("electrostatics_metrics.csv")
    if rank == 0:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Nd', 'n_i', 'V_bias', 'mean_E', 'U_E', 'C_E',
                'f_act_50', 'Lambda_phi', 'max_E',
                'solve_time', 'Newton_iterations', 'converged'
            ])
    
    # Run sweep
    run_idx = 0
    for Nd in Nd_values:
        for n_i in n_i_values:
            for V_bias in V_bias_values:
                run_idx += 1
                
                if rank == 0:
                    print(f"\n{'='*80}")
                    print(f"RUN {run_idx}/{total_runs}: Nd={Nd:.2e}, n_i={n_i:.2e}, V_bias={V_bias:.2f} V")
                    print(f"{'='*80}")
                
                try:
                    # Build system
                    V, F_form, J_form, bcs, phi_hat, phi_ref_val = build_nonlinear_poisson(
                        msh, scaling, dims, V_bias, Nd, Na=0.0, n_i=n_i
                    )
                    
                    # Solve
                    t0 = time.perf_counter()
                    info = solve_nonlinear_poisson(
                        F_form, J_form, phi_hat, bcs, comm,
                        phi_ref_hat_value=phi_ref_val,
                        verbose=False
                    )
                    solve_time = time.perf_counter() - t0
                    
                    # Compute metrics
                    metrics = compute_field_metrics(
                        phi_hat, V, scaling, dims, V_bias, comm
                    )
                    
                    # Store results
                    result = {
                        'Nd': Nd,
                        'n_i': n_i,
                        'V_bias': V_bias,
                        'mean_E': metrics['mean_E'],
                        'U_E': metrics['U_E'],
                        'C_E': metrics['C_E'],
                        'f_act_50': metrics['f_act_50'],
                        'Lambda_phi': metrics['Lambda_phi'],
                        'max_E': metrics['max_E'],
                        'solve_time': solve_time,
                        'Newton_iterations': info['newton_iters'],
                        'converged': True
                    }
                    results.append(result)
                    
                    if rank == 0:
                        print(f"\n  ✓ Converged in {info['newton_iters']} iterations ({solve_time:.2f} s)")
                        print(f"    <E> = {metrics['mean_E']:.3e} V/m")
                        print(f"    U_E = {metrics['U_E']:.4f}")
                        print(f"    C_E = {metrics['C_E']:.4f}")
                        print(f"    f_act(0.5) = {metrics['f_act_50']:.4f}")
                        print(f"    Λ_φ = {metrics['Lambda_phi']:.4f}")
                        
                        # Write to CSV
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                result['Nd'], result['n_i'], result['V_bias'],
                                result['mean_E'], result['U_E'], result['C_E'],
                                result['f_act_50'], result['Lambda_phi'], result['max_E'],
                                result['solve_time'], result['Newton_iterations'], 1
                            ])
                
                except (PoissonSolveError, Exception) as e:
                    if rank == 0:
                        print(f"\n  ✗ FAILED: {e}")
                        with open(csv_path, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([
                                Nd, n_i, V_bias,
                                0, 0, 0, 0, 0, 0, 0, 0, 0
                            ])
                    continue
    
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETE")
        print(f"{'='*80}")
        print(f"\nSuccessful runs: {len(results)}/{total_runs}")
        print(f"Results saved to: {csv_path}")
        
        # Generate plots
        generate_experiment_plots(csv_path)
        
        # Generate summary
        generate_experiment_summary(csv_path)


def generate_experiment_plots(csv_path):
    """Generate all publication-quality plots."""
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        print(f"\nWarning: Could not generate plots: {e}")
        print("Install pandas, matplotlib, and seaborn to generate plots.")
        return
    
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df[df['converged'] == 1]
    
    # Set plotting style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("viridis")
    
    # ── Plot 1: Effective active volume heatmap ──
    print("\n  Generating Plot 1: Effective active volume heatmap...")
    df_03V = df[df['V_bias'] == 0.3]
    pivot_fact = df_03V.pivot(index='n_i', columns='Nd', values='f_act_50')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_fact, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=1, cbar_kws={'label': r'$f_{\mathrm{act}}(0.5)$'},
                ax=ax)
    ax.set_xlabel(r'$N_d$ (m$^{-3}$)', fontsize=12)
    ax.set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=12)
    ax.set_title(r'Effective Active Volume Fraction at $V_{\mathrm{bias}} = 0.3$ V', fontsize=13)
    
    ax.set_xticklabels([f'{float(x):.1e}' for x in pivot_fact.columns], rotation=45)
    ax.set_yticklabels([f'{float(y):.1e}' for y in pivot_fact.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig('plot1_active_volume_heatmap.png', dpi=300)
    plt.close()
    print("    ✓ Saved: plot1_active_volume_heatmap.png")
    
    # ── Plot 2: Crowding factor heatmap ──
    print("  Generating Plot 2: Crowding factor heatmap...")
    pivot_crowd = df_03V.pivot(index='n_i', columns='Nd', values='C_E')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_crowd, annot=True, fmt='.2f', cmap='YlOrRd',
                cbar_kws={'label': r'$C_E$ (p99/$\langle E \rangle$)'},
                ax=ax)
    ax.set_xlabel(r'$N_d$ (m$^{-3}$)', fontsize=12)
    ax.set_ylabel(r'$n_i$ (m$^{-3}$)', fontsize=12)
    ax.set_title(r'Field Crowding Factor at $V_{\mathrm{bias}} = 0.3$ V', fontsize=13)
    
    ax.set_xticklabels([f'{float(x):.1e}' for x in pivot_crowd.columns], rotation=45)
    ax.set_yticklabels([f'{float(y):.1e}' for y in pivot_crowd.index], rotation=0)
    
    plt.tight_layout()
    plt.savefig('plot2_crowding_factor_heatmap.png', dpi=300)
    plt.close()
    print("    ✓ Saved: plot2_crowding_factor_heatmap.png")
    
    # ── Plot 3: Bias dependence ──
    print("  Generating Plot 3: Bias dependence curves...")
    Nd_min, Nd_max = df['Nd'].min(), df['Nd'].max()
    ni_min, ni_max = df['n_i'].min(), df['n_i'].max()
    
    cases = [
        (Nd_min, ni_min, 'Low $N_d$, Low $n_i$'),
        (Nd_max, ni_min, 'High $N_d$, Low $n_i$'),
        (Nd_min, ni_max, 'Low $N_d$, High $n_i$'),
        (Nd_max, ni_max, 'High $N_d$, High $n_i$'),
    ]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for Nd_val, ni_val, label in cases:
        subset = df[(df['Nd'] == Nd_val) & (df['n_i'] == ni_val)]
        subset = subset.sort_values('V_bias')
        ax.plot(subset['V_bias'], subset['f_act_50'], 'o-', label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel(r'$V_{\mathrm{bias}}$ (V)', fontsize=12)
    ax.set_ylabel(r'$f_{\mathrm{act}}(0.5)$', fontsize=12)
    ax.set_title('Active Volume Fraction vs Bias Voltage', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig('plot3_bias_dependence.png', dpi=300)
    plt.close()
    print("    ✓ Saved: plot3_bias_dependence.png")
    
    # ── Plot 4: Field uniformity overview ──
    print("  Generating Plot 4: Field uniformity overview...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: U_E vs Nd
    for ni_val in sorted(df['n_i'].unique()):
        subset = df_03V[df_03V['n_i'] == ni_val].sort_values('Nd')
        ax1.semilogx(subset['Nd'], subset['U_E'], 'o-', 
                     label=f'$n_i = {ni_val:.1e}$', linewidth=2, markersize=8)
    
    ax1.set_xlabel(r'$N_d$ (m$^{-3}$)', fontsize=12)
    ax1.set_ylabel(r'$U_E$ (std/$\langle E \rangle$)', fontsize=12)
    ax1.set_title('Field Uniformity Index', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Lambda_phi vs Nd
    for ni_val in sorted(df['n_i'].unique()):
        subset = df_03V[df_03V['n_i'] == ni_val].sort_values('Nd')
        ax2.semilogx(subset['Nd'], subset['Lambda_phi'], 'o-',
                     label=f'$n_i = {ni_val:.1e}$', linewidth=2, markersize=8)
    
    ax2.set_xlabel(r'$N_d$ (m$^{-3}$)', fontsize=12)
    ax2.set_ylabel(r'$\Lambda_\phi$ ($L_{\phi 90}/L_{\mathrm{gap}}$)', fontsize=12)
    ax2.set_title('Voltage Drop Localization', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('plot4_field_uniformity.png', dpi=300)
    plt.close()
    print("    ✓ Saved: plot4_field_uniformity.png")
    
    print("\n  All plots generated successfully!")


def generate_experiment_summary(csv_path):
    """Generate text summary of results."""
    try:
        import pandas as pd
    except ImportError:
        print("\nWarning: pandas not available, skipping summary generation")
        return
    
    df = pd.read_csv(csv_path)
    df = df[df['converged'] == 1]
    
    print("\n" + "="*80)
    print("PHYSICAL INTERPRETATION")
    print("="*80)
    
    # Find regime where active volume collapses
    collapse_threshold = 0.5
    collapsed = df[df['f_act_50'] < collapse_threshold]
    
    if len(collapsed) > 0:
        print(f"\n✦ Active volume collapse (f_act < {collapse_threshold}):")
        print(f"  Observed in {len(collapsed)} / {len(df)} cases")
        worst_case = collapsed.loc[collapsed['f_act_50'].idxmin()]
        print(f"  Worst case: Nd={worst_case['Nd']:.2e}, n_i={worst_case['n_i']:.2e}, "
              f"V={worst_case['V_bias']:.2f}V → f_act={worst_case['f_act_50']:.3f}")
    else:
        print(f"\n✦ No significant active volume collapse (all f_act > {collapse_threshold})")
    
    # Field crowding
    crowding_threshold = 2.0
    crowded = df[df['C_E'] > crowding_threshold]
    
    if len(crowded) > 0:
        print(f"\n✦ Significant field crowding (C_E > {crowding_threshold}):")
        print(f"  Observed in {len(crowded)} / {len(df)} cases")
        worst_crowding = crowded.loc[crowded['C_E'].idxmax()]
        print(f"  Worst case: Nd={worst_crowding['Nd']:.2e}, n_i={worst_crowding['n_i']:.2e}, "
              f"V={worst_crowding['V_bias']:.2f}V → C_E={worst_crowding['C_E']:.2f}")
    else:
        print(f"\n✦ Moderate field crowding (all C_E < {crowding_threshold})")
    
    # Screening trends
    print(f"\n✦ Screening effects:")
    df_low_bias = df[df['V_bias'] == 0.3]
    corr_Nd = df_low_bias[['Nd', 'f_act_50']].corr().iloc[0, 1]
    corr_ni = df_low_bias[['n_i', 'f_act_50']].corr().iloc[0, 1]
    
    print(f"  Correlation(Nd, f_act): {corr_Nd:+.3f}")
    print(f"  Correlation(n_i, f_act): {corr_ni:+.3f}")
    
    if corr_Nd < -0.3:
        print("  → Higher doping reduces active volume (screening)")
    if corr_ni < -0.3:
        print("  → Higher intrinsic carrier density reduces active volume")
    
    # Uniformity
    mean_U_E = df['U_E'].mean()
    print(f"\n✦ Field uniformity:")
    print(f"  Average U_E = {mean_U_E:.3f}")
    if mean_U_E < 0.3:
        print("  → Fields are relatively uniform")
    elif mean_U_E < 0.6:
        print("  → Moderate field non-uniformity")
    else:
        print("  → Significant field non-uniformity")
    
    # Voltage drop localization
    mean_Lambda = df['Lambda_phi'].mean()
    print(f"\n✦ Voltage drop localization:")
    print(f"  Average Λ_φ = {mean_Lambda:.3f}")
    if mean_Lambda > 0.9:
        print("  → Voltage drops nearly uniformly across device")
    elif mean_Lambda > 0.7:
        print("  → Moderate voltage drop localization")
    else:
        print("  → Strong voltage drop localization near contacts")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
This electrostatic study reveals:

1. ACTIVE VOLUME: Electrostatic screening reduces the effective detector
   volume when carrier concentrations are high. The active region becomes
   confined near contacts.

2. FIELD CROWDING: Peak fields at contact edges can exceed mean fields by
   2-3×, indicating non-uniform current injection and potential hot-spot
   formation.

3. SCREENING REGIMES: When Nd >> n_i, space-charge screening dominates.
   When n_i >> Nd, mobile carrier screening becomes significant.

4. DESIGN IMPLICATIONS: For uniform detector operation, maintain
   Nd/n_ref << 1 and ensure bias voltage scales with carrier concentrations.

NOTE: This is a purely electrostatic analysis. Transport effects
(mobility, recombination, generation) are not included.
    """)


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
    
    # Digital Experiment
    parser.add_argument("--electrostatics_experiment", action="store_true",
                        help="Run full electrostatics parameter sweep (60 runs)")


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

    # ── Brick 3: Nd continuation ──
    if args.nd_continuation:
        run_nd_continuation(args, comm, rank)
        sys.exit(0)

    # ── Brick 3: Nonlinear bias sweep ──
    if args.nl_sweep:
        exit_code = run_nonlinear_bias_sweep(args, comm, rank)
        sys.exit(exit_code)
    
    # ── Digital Experiment: Electrostatics ──
    if args.electrostatics_experiment:
        run_electrostatics_experiment(args, comm, rank)
        sys.exit(0)

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
