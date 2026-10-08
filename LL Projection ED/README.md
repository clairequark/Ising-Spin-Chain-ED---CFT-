# LL Projection ED

Exact diagonalization of the truncated Landau-orbital Hamiltonian of
Rahmani, Sung, Putterman, Roushan, Ghaemi & Jiang, *PRX Quantum* **1**, 020309 (2020), Eq. (3):

```
H = Σ_j [ V10 n_{j+1} n_{j+2} + V20 n_j n_{j+2} + V30 n_j n_{j+3}
          + √(V10 V30) ( c†_j c†_{j+3} c_{j+2} c_{j+1} + h.c. ) ]
```

Adapted from the Ising ED code in the parent folder.

## Files

| File | What it does | Ising counterpart |
|---|---|---|
| `basis_construction.jl` | Fermion signs, fixed-N_e enumeration, K_y, T³ Bloch basis | `basis_construction.jl` |
| `ConstructHamiltonian.jl` | Module `ConstructLLHamiltonian`: `ModelParams`, H in the bitstring basis, H in (K_y, kx) blocks, diagonalization | `ConstructHamiltonian.jl` |
| `test_hamiltonian.jl` | Checks: Hermiticity, [H,T]=0, block vs brute force, 3-fold degeneracy, E₀=0, amplitudes (−t)^P, OBC | none |
| `run_ed.jl` | Example: lowest levels in every inequivalent block | your runner |

Run from inside this folder: `julia test_hamiltonian.jl`, then `julia run_ed.jl`. Needs `Arpack` (as before); `Plots` is not used.

## Physics in one page

**Where the electrons live.** On a torus of circumferences L₁ × L₂, in a perpendicular field B,
with magnetic length ℓ = √(ħ/eB) set to 1. The number of flux quanta through the torus is
N = L₁L₂/2π. That's the paper's N, the number of orbitals, and the number of qubits.

**What an orbital is.** In Landau gauge (A = Bx ŷ), momentum along the L₁ direction is conserved,
so lowest-Landau-level states are
`φ_j(x, y) ∝ e^{i k_j y} e^{-(x - X_j)²/2}` with `k_j = 2πj/L₁` and `X_j = k_j ℓ² = 2πj/L₁`.
- Orbital j is a ring wrapped around the L₁ cycle: a plane wave along y, a Gaussian stripe of width ~ℓ in x, centered at X_j.
- Each stripe encloses exactly one flux quantum.
- Orbital j + N is orbital j shifted by L₂, so periodicity in x makes the chain a ring with c_{j+N} = c_j.

The chain index j **is** the x coordinate of the torus, coarse-grained into stripes. Momentum along y is locked to position along x, so a single integer labels a state.

**Why one integer is enough.** Inside a Landau level the guiding-center coordinates obey
[X, Y] = iℓ², just like (x, p) of a 1D particle. A Landau level therefore *is* the Hilbert space of a 1D
problem, with one state per flux quantum, the way phase space has one state per area 2πħ.

**Why the Hamiltonian is so simple.** Start from the full Hamiltonian, Σ (p + eA)²/2m + V_Coulomb.
1. The kinetic term gives Landau levels ħω_c(n + ½), each N-fold degenerate.
2. At ν = 1/3 the lowest level is partly filled. For ħω_c ≫ e²/εℓ we project onto it. Every state then has the same kinetic energy, so it drops out as a constant.
3. What remains is the interaction, projected into the Landau level: Σ V_{j1 j2 j3 j4} c† c† c c. It's the same "flat band + projected interaction" story as an FCI, and the LLL is the ideal C = 1 flat band.
4. The magnetic field survives in three places: the degenerate manifold itself, the Gaussian orbital shape that sets the matrix elements V_km, and the x ↔ k_y lock.
5. y-momentum conservation means j1 + j2 = j3 + j4 (mod N). Every term moves two electrons symmetrically and keeps their center of mass (Eq. 1).
6. The matrix elements fall off like exp[−2π²(k² + m²)/L₁²], the overlap of stripes a distance ~2π/L₁ apart. On a thin torus only the shortest-range terms survive, which gives Eq. (3).

**Reading Eq. (3) in real space.**
- V10, V20, V30: repulsion between electrons in stripes 1, 2, 3 apart.
- Pair hop: two electrons in neighboring stripes (j+1, j+2) move outward to (j, j+3), or the reverse ("squeezing"), with the center of mass fixed.
- At t = √(V30/V10) → 0 the ground state is the stripe crystal 100100… (thin-torus limit). Increasing t dresses it into a Laughlin-type liquid, Eq. (4).
- H is a sum of positive operators, so the ground state has E = 0 exactly.

**Fermions.** Electrons are spin polarized, so each orbital holds 0 or 1 electron: a bitstring.
Jordan-Wigner signs appear only when an electron moves around the ring past the others, giving (−1)^(N_e−1).
- N_e odd: no signs at all.
- N_e even: signs, as for the paper's N = 24 (N_e = 8).

## Quantum numbers (PBC)

| Label | Operator | Values | Notes |
|---|---|---|---|
| N_e | Σ n_j | fixed (N/3) | Only binomial(N, N_e) states are enumerated |
| K_y | Σ j n_j mod N | 0 … N−1 | Diagonal: a filter, like fixing S^z |
| kx | T³ eigenvalue e^{2πi kx/N_e} | 0 … N_e−1 | Translation by 3 orbitals |

Single-orbital translation T commutes with H but shifts K_y by N_e. This is the magnetic-translation algebra.
- Consequence: (K_y, kx), (K_y+N_e, kx) and (K_y+2N_e, kx) have identical spectra, which is the exact 3-fold center-of-mass degeneracy.
- Only K_y ∈ 0 … N_e−1 needs to be diagonalized: N_e² blocks of size ≈ binomial(N, N_e)/(N·N_e).
- With OBC (the paper's circuit geometry) only N_e and the plain sum K_y = Σ j n_j survive.

The ground state built from the root 100100… sits at K_y = Σ 3m and kx = 0 (N_e odd) or N_e/2 (N_e even).
`predicted_ground_sector` computes this, and the tests check it.

## Conventions

- Bit j of the integer s is n_j. Orbitals are j = 0 … N−1.
- |s⟩ = c†_{j1} c†_{j2} … |0⟩ with j1 < j2 < …. Acting with c_j or c†_j gives (−1)^(# occupied below j).
- Bloch states: |a, kx⟩ = R^{-1/2} Σ_{r<R} e^{-ikr} (T³)^r |a⟩, with k = 2πkx/N_e. Fermion signs are included in (T³)^r|a⟩.
