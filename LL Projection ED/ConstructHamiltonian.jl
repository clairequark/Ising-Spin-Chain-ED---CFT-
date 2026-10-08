module ConstructLLHamiltonian

using SparseArrays
using LinearAlgebra
using Arpack

#= Model parameters for the truncated Landau-orbital Hamiltonian, Eq. (3) of Rahmani et al.:

    H = Σ_j [ V10 n_{j+1} n_{j+2} + V20 n_j n_{j+2} + V30 n_j n_{j+3}
              + √(V10 V30) ( c†_j c†_{j+3} c_{j+2} c_{j+1} + h.c. ) ]

N   = number of lowest-Landau-level orbitals = number of flux quanta = the paper's N (qubits).
Ne  = number of electrons. ν = Ne/N = 1/3 for the Laughlin state.
PBC = 1: ring (the torus). PBC = 0: open chain (what the paper's circuit actually implements).

NOTE: ModelParams is defined BEFORE the include below, because the functions in
basis_construction.jl have `params::ModelParams` in their signatures and Julia resolves those
types when the method is defined. =#
Base.@kwdef mutable struct ModelParams
    N::Int64
    Ne::Int64
    V10::Float64
    V20::Float64
    V30::Float64
    PBC::Int64 = 1
end

include("basis_construction.jl")


# ---------------------------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------------------------

#= The ground state only depends on t = √(V30/V10) (Eq. 4), so it is natural to parametrize by t.
V20 does not enter the ground state but does shape the excitations. If not given, it is set to
the value implied by Eq. (2): V20/V10 = 4 (t/3)^(3/4). Needs t >= 0. =#
function params_from_t(N::Int; t::Real, V10::Real = 1.0,
                       V20::Union{Nothing,Real} = nothing, PBC::Int = 1, Ne::Int = N ÷ 3)
    if V20 === nothing
        V20 = 4 * V10 * (t / 3)^(3 / 4)
    end
    return ModelParams(N = N, Ne = Ne, V10 = V10, V20 = V20, V30 = V10 * t^2, PBC = PBC)
end

#= Parameters from the torus circumference L1 (in units of the magnetic length), via Eq. (2):
    V_km ∝ (k² - m²) exp[-2π²(k² + m²)/L1²].
(The PDF prints "/L1" in the exponent; dimensionally it has to be L1² with ℓ_B = 1.)
With x = exp(-2π²/L1²): V10 ∝ x, V20 ∝ 4x⁴, V30 ∝ 9x⁹, and the hopping V21 = 3x⁵ = √(V10 V30).
We divide everything by V10 so that V10 = 1. Then t = 3x⁴. =#
function params_from_L1(N::Int, L1::Real; PBC::Int = 1, Ne::Int = N ÷ 3)
    x = exp(-2π^2 / L1^2)
    return ModelParams(N = N, Ne = Ne, V10 = 1.0, V20 = 4 * x^3, V30 = 9 * x^8, PBC = PBC)
end


# ---------------------------------------------------------------------------------------------
# The two kinds of terms in H
# ---------------------------------------------------------------------------------------------

#= Diagonal (electrostatic) part, the analogue of construct_occupation_terms in the Ising code.
Σ_j V10 n_{j+1}n_{j+2} on a ring is the same as Σ_a V10 n_a n_{a+1}, and likewise for distances 2
and 3, so we loop over pairs (a, a+d) at distance d = 1, 2, 3.
PBC: a + d wraps around mod N.  OBC: pairs that would stick out past orbital N-1 are dropped.
Each of these terms is either a product of occupations (0 or 1), so no fermion signs appear. =#
function diagonal_energy(s::Int, params::ModelParams)
    N = params.N
    V = (params.V10, params.V20, params.V30)        # V[d] = V_{d0}, the distance-d repulsion
    E = 0.0
    for d in 1:3, a in 0:(N - 1)
        b = a + d
        if b >= N
            if params.PBC != 1
                continue
            end
            b -= N
        end
        E += V[d] * ((s >> a) & 1) * ((s >> b) & 1)
    end
    return E
end

#= Off-diagonal (pair-hopping) part, the analogue of construct_interaction_terms in the Ising code.
For each window of four orbitals (j0, j1, j2, j3) = (j, j+1, j+2, j+3):
    c†_{j0} c†_{j3} c_{j2} c_{j1} : 0110 -> 1001   (two neighbors move apart)
    c†_{j1} c†_{j2} c_{j3} c_{j0} : 1001 -> 0110   (h.c.: two electrons "squeeze" together)
Both moves keep j1 + j2 = j0 + j3, i.e. K_y is conserved.
Returns a list of (new_state, amplitude) pairs, with the fermion sign folded into the amplitude.
Away from the boundary the sign is always +1 (no other electron sits inside the window).
On a ring, the three windows that wrap around give (-1)^(Ne-1).
PBC: all N windows (indices mod N).  OBC: only windows that fit, j = 0, ..., N-4. =#
function offdiagonal_terms(s::Int, params::ModelParams)
    N = params.N
    hop = sqrt(params.V10 * params.V30)
    out = Tuple{Int,Float64}[]
    hop == 0.0 && return out

    jmax = params.PBC == 1 ? N - 1 : N - 4
    for j in 0:jmax
        j0, j1, j2, j3 = j, mod(j + 1, N), mod(j + 2, N), mod(j + 3, N)

        s2, sg = pair_hop(s, j0, j3, j2, j1)            # c†_{j0} c†_{j3} c_{j2} c_{j1}
        sg != 0 && push!(out, (s2, hop * sg))

        s2, sg = pair_hop(s, j1, j2, j3, j0)            # c†_{j1} c†_{j2} c_{j3} c_{j0}
        sg != 0 && push!(out, (s2, hop * sg))
    end
    return out
end


# ---------------------------------------------------------------------------------------------
# Hamiltonian in the plain bitstring basis (fixed N_e, optionally fixed K_y)
# ---------------------------------------------------------------------------------------------

#= Builds H as a sparse matrix in the bitstring basis with N_e electrons. If Ky is given, only that
K_y sector is kept (works for PBC and OBC). This is the analogue of generate_hamiltonian in the
Ising code; it is the brute-force reference that the momentum blocks are checked against.
Returns (H, states): states[i] is the bitstring that labels row/column i. =#
function generate_hamiltonian(params::ModelParams; Ky::Union{Nothing,Int} = nothing)
    states = fixed_N_states(params.N, params.Ne)
    if Ky !== nothing
        states = filter(s -> total_Ky(s, params) == Ky, states)
    end
    dim = length(states)
    rows, cols, vals = Int[], Int[], Float64[]

    for (i, s) in enumerate(states)
        push!(rows, i)
        push!(cols, i)
        push!(vals, diagonal_energy(s, params))

        for (s2, amp) in offdiagonal_terms(s, params)
            i2 = index_in(states, s2)
            i2 == 0 && error("H left the sector. This cannot happen if H conserves N_e and K_y.")
            push!(rows, i2)             # new state -> row, old state -> column (same as Ising)
            push!(cols, i)
            push!(vals, amp)
        end
    end

    return sparse(rows, cols, vals, dim, dim), states
end

#= The fermionic translation operator T (c_j -> c_{j+1}) in the fixed-N_e bitstring basis.
`states` must be closed under T, i.e. the full fixed_N_states(N, Ne) list and not a single K_y
sector (T shifts K_y by N_e). Used to check [H, T] = 0. =#
function construct_translation_operator(params::ModelParams, states::Vector{Int})
    params.PBC == 1 || error("Translation is not a symmetry with OBC (PBC == 0).")
    N, Ne = params.N, params.Ne
    mask = (1 << N) - 1
    rows, cols, vals = Int[], Int[], Float64[]

    for (i, s) in enumerate(states)
        s2, sg = translate(s, N, Ne, mask)
        i2 = index_in(states, s2)
        i2 == 0 && error("`states` is not closed under T; pass fixed_N_states(N, Ne).")
        push!(rows, i2)
        push!(cols, i)
        push!(vals, sg)
    end
    return sparse(rows, cols, vals, length(states), length(states))
end


# ---------------------------------------------------------------------------------------------
# Hamiltonian in a momentum block (K_y, kx)
# ---------------------------------------------------------------------------------------------

#= Matrix of H in the Bloch basis of one block. For each kernel |a⟩ (period R_a):

    H|a⟩ = E_diag(a)|a⟩ + Σ_ℓ h_ℓ |s_ℓ⟩                   (h_ℓ includes the hopping fermion sign)
    find_rep_and_shift: (T^q)^l |s_ℓ⟩ = χ |b⟩              (χ = translation fermion sign)
    ⟨b, kx| H |a, kx⟩ += h_ℓ χ exp(-i k l) √(R_a / R_b),   k = 2π kx / N0.

Because [H, T^q] = 0 it is enough to act with H on the kernel |a⟩ alone. The Ising
block_interaction_terms also looped over all R translates of |a⟩, which gives R identical
contributions of size 1/√(R_a R_b); that reproduces the same √(R_a/R_b) with R times the work.
If |b⟩ is not in the basis (its Bloch sum vanishes at this kx), the term drops out. =#
function block_hamiltonian(basis::MomentumBasis, params::ModelParams)
    N0 = num_kx(params)
    mask = (1 << params.N) - 1
    k = 2π * basis.kx / N0
    nb = length(basis.states)
    rows, cols, vals = Int[], Int[], ComplexF64[]

    for i in 1:nb
        a = basis.states[i].rep
        Ra = basis.states[i].R

        # Diagonal energies are translation invariant, so ⟨a,kx|H_diag|a,kx⟩ = E_diag(a).
        push!(rows, i)
        push!(cols, i)
        push!(vals, diagonal_energy(a, params))

        for (s2, amp) in offdiagonal_terms(a, params)
            rep, l, chi = find_rep_and_shift(s2, params, mask)
            j = get(basis.indexmap, rep, 0)
            j == 0 && continue
            Rb = basis.states[j].R
            push!(rows, j)
            push!(cols, i)
            push!(vals, amp * chi * exp(-im * k * l) * sqrt(Ra / Rb))
        end
    end

    H = sparse(rows, cols, vals, nb, nb)            # sparse() sums repeated (row, col) entries
    if nb > 0
        herr = maximum(abs, H - H')
        herr > 1e-10 && @warn "Block (Ky=$(basis.Ky), kx=$(basis.kx)) is not Hermitian: max|H - H'| = $herr"
    end
    # Remove round-off asymmetry from the phases so that Hermitian solvers can be used.
    return (H + H') / 2
end

#= Builds every block (K_y, kx) for K_y in Ky_list and all kx ∈ {0, ..., N0-1}.
Default Ky_list = 0:(N0-1). K_y and K_y + N_e are related by T and have identical spectra (the
exact 3-fold center-of-mass degeneracy at ν = 1/3), so these N0 values are the inequivalent ones.
Pass Ky_list = 0:(N-1) to build every block (useful for checking against the full spectrum).
Returns Dict (Ky, kx) => (basis = MomentumBasis, H = sparse block matrix). =#
function construct_block_hamiltonian(params::ModelParams; Ky_list = 0:(num_kx(params) - 1))
    params.PBC == 1 || error("Momentum blocks need PBC.")
    blocks = Dict{Tuple{Int,Int},Any}()

    for Ky in Ky_list
        candidates = ky_sector_states(Ky, params)       # enumerate each K_y sector once
        for kx in 0:(num_kx(params) - 1)
            basis = build_momentum_basis(Ky, kx, params; candidates = candidates)
            blocks[(Ky, kx)] = (basis = basis, H = block_hamiltonian(basis, params))
        end
    end
    return blocks
end


# ---------------------------------------------------------------------------------------------
# Where the ν = 1/3 ground states should be
# ---------------------------------------------------------------------------------------------

# The thin-torus "root" configuration 100100...100 of Eq. (4) (orbitals 0, 3, 6, ... occupied).
root_state(params::ModelParams) = sum(1 << (3 * m) for m in 0:(params.Ne - 1))

#= Predicted block of the root's ground state (sector "a" in the paper). K_y = Σ 3m. kx is the
unique T^q momentum compatible with the root's orbit: the root is T^3-invariant (R = 1) with sign
(-1)^(Ne-1), so kx = 0 for N_e odd and kx = N_e/2 for N_e even. Its partners b, c sit at
K_y + N_e and K_y + 2N_e with the same kx. =#
function predicted_ground_sector(params::ModelParams)
    s = root_state(params)
    Ky = total_Ky(s, params)
    kx = findfirst(k -> checkstate(s, k, params) != -1, 0:(num_kx(params) - 1)) - 1
    return Ky, kx
end


# ---------------------------------------------------------------------------------------------
# Diagonalization
# ---------------------------------------------------------------------------------------------

#= Partial, lowest-eigenvalue computation for H stored as a sparse matrix (Arpack).
Complex Hermitian blocks go through Arpack's general complex solver, so the eigenvalues come back
with ~1e-15 imaginary parts; we keep the real part. Results are sorted. =#
function sparse_diagonalization(A::SparseMatrixCSC; numeigs::Int)
    vals, vecs = eigs(A; nev = numeigs, which = :SR)
    p = sortperm(real.(vals))
    return real.(vals[p]), vecs[:, p]
end

# Full diagonalization for H stored as a sparse matrix.
function dense_diagonalization(A::SparseMatrixCSC)
    eig = eigen(Hermitian(Matrix(A)))
    return eig.values, eig.vectors
end

# Lowest `numeigs` eigenpairs: dense below `dense_cutoff`, Arpack above.
function lowest_eigs(A::SparseMatrixCSC; numeigs::Int = 6, dense_cutoff::Int = 2000)
    n = size(A, 1)
    n == 0 && return Float64[], zeros(eltype(A), 0, 0)
    if n <= dense_cutoff || numeigs >= n - 1
        vals, vecs = dense_diagonalization(A)
        m = min(numeigs, n)
        return vals[1:m], vecs[:, 1:m]
    end
    return sparse_diagonalization(A; numeigs = numeigs)
end

export ModelParams, params_from_t, params_from_L1,
       fixed_N_states, ky_sector_states, index_in, total_Ky, translate, translate_r,
       translation_step, num_kx, checkstate, find_rep_and_shift,
       MomentumState, MomentumBasis, build_momentum_basis, unfold_to_fock,
       diagonal_energy, offdiagonal_terms, generate_hamiltonian, construct_translation_operator,
       block_hamiltonian, construct_block_hamiltonian, root_state, predicted_ground_sector,
       sparse_diagonalization, dense_diagonalization, lowest_eigs

end
