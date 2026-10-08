#= Basis construction for the Landau-orbital chain (Rahmani et al., PRX Quantum 1, 020309, Eq. (3)).

This is the fermionic, fixed-particle-number, magnetic-translation version of the Ising
basis_construction.jl. The schematic procedure is the same as before:

    Act a translation on b until b returns to itself.
    IF b is the smallest element of its orbit, store b and the period of the orbit.

What changes relative to the Ising code:

    1. Bits are now ORBITAL OCCUPATIONS of spin-polarized electrons: bit j = n_j ∈ {0, 1}.
       (Ising: bit = spin. Here: bit 1 = an electron sits in Landau orbital j.)
    2. Electrons are fermions, so moving them around can produce minus signs (Jordan-Wigner).
    3. Particle number N_e is conserved, so we only ever enumerate bitstrings with N_e ones.
    4. K_y = Σ_j j n_j (mod N) is conserved. It is diagonal in the bitstring basis, so it is
       just a filter, exactly like fixing total S^z would be for a spin chain.
    5. The single-orbital translation T does NOT commute with K_y (it shifts K_y by N_e), so we
       build Bloch states of T^q instead, where q = N / gcd(N, N_e) (q = 3 at ν = 1/3).

Conventions used everywhere in this folder:

    - Orbitals are j = 0, ..., N-1 (zero indexed, like bits). N is the paper's N = number of qubits.
    - A bitstring s stands for the ordered fermion state
          |s⟩ = c†_{j1} c†_{j2} ... c†_{jn} |0⟩     with j1 < j2 < ... < jn.
    - With that ordering, acting with c_j or c†_j on |s⟩ gives a sign
          (-1)^(number of occupied orbitals with index < j).
      This is the Jordan-Wigner string of Eq. (5) of the paper.
=#


# ---------------------------------------------------------------------------------------------
# 1. Fermionic bit operations
# ---------------------------------------------------------------------------------------------

# Jordan-Wigner sign: (-1)^(number of occupied orbitals strictly below orbital j).
function fermion_sign(s::Int, j::Int)
    below = s & ((1 << j) - 1)              # keep only bits 0, ..., j-1
    return iseven(count_ones(below)) ? 1 : -1
end

# c_j |s⟩ = sign |s'⟩. Returns (s', sign). sign = 0 means c_j killed the state (orbital j was empty).
function annihilate(s::Int, j::Int)
    if ((s >> j) & 1) == 0
        return 0, 0
    end
    return s ⊻ (1 << j), fermion_sign(s, j)
end

# c†_j |s⟩ = sign |s'⟩. Returns (s', sign). sign = 0 means Pauli blocking (orbital j was full).
function create(s::Int, j::Int)
    if ((s >> j) & 1) == 1
        return 0, 0
    end
    return s ⊻ (1 << j), fermion_sign(s, j)
end

#= Two-body operator c†_a c†_b c_c c_d acting on |s⟩. Operators act right to left: c_d first, c†_a last.
Every term in the Hamiltonian of Eq. (1) has this form. Returns (s', sign), sign = 0 if the state is killed.
The sign is the product of the four Jordan-Wigner signs, each evaluated on the state at that moment. =#
function pair_hop(s::Int, a::Int, b::Int, c::Int, d::Int)
    s, sg1 = annihilate(s, d)
    sg1 == 0 && return 0, 0
    s, sg2 = annihilate(s, c)
    sg2 == 0 && return 0, 0
    s, sg3 = create(s, b)
    sg3 == 0 && return 0, 0
    s, sg4 = create(s, a)
    sg4 == 0 && return 0, 0
    return s, sg1 * sg2 * sg3 * sg4
end


# ---------------------------------------------------------------------------------------------
# 2. Translations (now with fermion signs)
# ---------------------------------------------------------------------------------------------

#= Fermionic translation T: c_j -> c_{j+1}, with c_N ≡ c_0 (PBC). On bits this is the same left
rotation as the Ising code: bit j -> bit j+1, and the top bit (orbital N-1) wraps to bit 0.

The new ingredient is the sign. T|s⟩ = c†_{j1+1} ... c†_{jn+1}|0⟩. If orbital N-1 was occupied,
its creation operator became c†_0, which is now the LOWEST orbital but still sits LAST in the
product. Moving it to the front passes it through the other N_e - 1 creation operators:
sign (-1)^(N_e - 1). If orbital N-1 was empty, nothing reorders and the sign is +1.

So for N_e odd the sign is always +1 (fermions behave like hard-core bosons here); for N_e even it
is -1 whenever an electron wraps around the ring.
!! MASK MUST BE OF THE FORM OF A BITSTRING WITH ALL ONES !! =#
function translate(s::Int, N::Int, Ne::Int, mask::Int)
    s_rot = ((s << 1) | (s >> (N - 1))) & mask
    wraps = (s >> (N - 1)) & 1                  # is the top orbital occupied?
    sign = (wraps == 1 && isodd(Ne - 1)) ? -1 : 1
    return s_rot, sign
end

# T^r |s⟩ = sign |s'⟩, applying T r times and accumulating the sign.
function translate_r(s::Int, r::Int, N::Int, Ne::Int, mask::Int)
    sign = 1
    for _ in 1:r
        s, sg = translate(s, N, Ne, mask)
        sign *= sg
    end
    return s, sign
end

#= The symmetry group we actually block-diagonalize with.
T shifts K_y by N_e (every electron moves one orbital, so Σ j n_j grows by N_e). Therefore T^q,
with q = N / gcd(N, N_e), shifts K_y by q*N_e, which is a multiple of N, i.e. by 0 (mod N).
So T^q commutes with K_y and we can label states by both.
(T^q)^(N0) = T^N = 1 with N0 = gcd(N, N_e), so the T^q momentum is kx ∈ {0, ..., N0 - 1}.
At ν = 1/3: q = 3 and N0 = N_e. =#
translation_step(params::ModelParams) = params.N ÷ gcd(params.N, params.Ne)
num_kx(params::ModelParams) = gcd(params.N, params.Ne)


# ---------------------------------------------------------------------------------------------
# 3. The conserved "y-momentum" K_y = Σ_j j n_j
# ---------------------------------------------------------------------------------------------

#= K_y is the total y-momentum (orbital j carries k_y = 2πj/L1), equivalently the center of mass of
the electrons along x (orbital j sits at x_j = 2πj ℓ²/L1). Every term in H moves two electrons
symmetrically, (j+1, j+2) <-> (j, j+3), so it never changes Σ j n_j.
PBC: defined mod N (the ring identifies orbital N with orbital 0).
OBC: there is no wrap-around, so the plain integer sum is conserved. =#
function total_Ky(s::Int, params::ModelParams)
    K = 0
    for j in 0:(params.N - 1)
        K += j * ((s >> j) & 1)
    end
    return params.PBC == 1 ? mod(K, params.N) : K
end


# ---------------------------------------------------------------------------------------------
# 4. Enumerating the fixed-N_e Hilbert space
# ---------------------------------------------------------------------------------------------

#= Gosper's hack: given a bitstring v with n ones, returns the next larger integer with n ones.
Starting at 0...0111 (N_e ones at the bottom), repeated calls walk through all binomial(N, N_e)
states in increasing order. This replaces the Ising loop `for s in 0:(2^N - 1)`, which would
visit ~10^9 states at N = 30 when only ~3*10^7 have the right particle number. =#
function next_bit_permutation(v::Int)
    t = v | (v - 1)                                         # set all bits below the lowest 1
    return (t + 1) | (((~t & -(~t)) - 1) >> (trailing_zeros(v) + 1))
end

# All bitstrings of length N with exactly Ne ones, sorted from least to greatest.
function fixed_N_states(N::Int, Ne::Int)
    Ne == 0 && return [0]
    states = Int[]
    s = (1 << Ne) - 1               # 0...0111: the smallest state with Ne electrons
    last = s << (N - Ne)            # 1110...0: the largest
    while true
        push!(states, s)
        s == last && break
        s = next_bit_permutation(s)
    end
    return states
end

# Position of bitstring s in a SORTED list of states (binary search). Returns 0 if absent.
function index_in(states::Vector{Int}, s::Int)
    i = searchsortedfirst(states, s)
    return (i <= length(states) && states[i] == s) ? i : 0
end

# All fixed-N_e bitstrings in a given K_y sector (still sorted).
function ky_sector_states(Ky::Int, params::ModelParams)
    return filter(s -> total_Ky(s, params) == Ky, fixed_N_states(params.N, params.Ne))
end


# ---------------------------------------------------------------------------------------------
# 5. Momentum (Bloch) basis for the pair (K_y, kx)
# ---------------------------------------------------------------------------------------------

#= Bloch states of T^q built on a kernel (representative) |a⟩ with period R:

    |a, kx⟩ = (1/√R) Σ_{r=0}^{R-1} exp(-i k r) (T^q)^r |a⟩,     k = 2π kx / N0.

This is your Ising |a, k⟩ with T -> T^q. Two differences:
    - (T^q)^r |a⟩ can carry a fermion sign, which is part of the state.
    - After one full period, (T^q)^R |a⟩ = σ |a⟩ with σ = ±1 (not always +1).
      The Bloch sum is nonzero only if exp(-i k R) σ = 1. For σ = +1 this is your old
      condition kx R / N0 ∈ Z; for σ = -1 it becomes kx R / N0 ∈ Z + 1/2. =#

#= Determines the orbital period of s within momentum sector kx. Rejects s if it is not a kernel state.
    INPUTS:  s - bitstring, kx - T^q momentum integer, params - model params
    OUTPUT:  R  - the orbit period (number of T^q steps to return to s)
             R = -1 if s is not the kernel of its orbit, or the Bloch sum vanishes at this kx. =#
function checkstate(s::Int, kx::Int, params::ModelParams)
    N, Ne = params.N, params.Ne
    q = translation_step(params)
    N0 = num_kx(params)
    mask = (1 << N) - 1
    t = s
    sign = 1

    for R in 1:N0
        t, sg = translate_r(t, q, N, Ne, mask)
        sign *= sg

        if t < s                        # s is NOT the smallest element of its orbit
            return -1
        elseif t == s                   # back to s after R steps: (T^q)^R |s⟩ = sign |s⟩
            # Need exp(-2πi kx R / N0) * sign == 1. Multiply the phase angle by N0/π:
            # 2 kx R + (N0 if sign == -1) must be a multiple of 2 N0.
            twice_phase = 2 * kx * R + (sign == -1 ? N0 : 0)
            if mod(twice_phase, 2 * N0) != 0
                return -1
            end
            return R
        end
    end

    return -1                           # unreachable, since (T^q)^N0 = 1
end

# Stores the kernel and period of a Bloch state.
struct MomentumState
    rep::Int        # kernel bitstring |a⟩ (smallest integer in its T^q orbit)
    R::Int          # orbit period under T^q
end

# Stores the Bloch basis of one symmetry block (K_y, kx).
struct MomentumBasis
    Ky::Int
    kx::Int
    states::Vector{MomentumState}
    indexmap::Dict{Int,Int}         # maps rep -> basis index
end

#= Generates the Bloch basis for the block (K_y, kx).
    INPUT:  Ky, kx, params. Optionally `candidates` = ky_sector_states(Ky, params), so that a loop
            over kx does not re-enumerate the K_y sector every time.
    OUTPUT: MomentumBasis, with states sorted from least to greatest rep.
T^q never changes K_y, so a whole orbit lives inside one K_y sector and filtering first is safe. =#
function build_momentum_basis(Ky::Int, kx::Int, params::ModelParams;
                              candidates::Vector{Int} = ky_sector_states(Ky, params))
    params.PBC == 1 || error("Momentum blocks need PBC; with OBC only N_e and K_y are good quantum numbers.")
    states = MomentumState[]
    indexmap = Dict{Int,Int}()

    for s in candidates
        R = checkstate(s, kx, params)
        if R != -1
            push!(states, MomentumState(s, R))
            indexmap[s] = length(states)
        end
    end

    return MomentumBasis(Ky, kx, states, indexmap)
end

#= I: bitstring s (in any K_y sector)
   O: (rep, shift, sign) such that (T^q)^shift |s⟩ = sign |rep⟩, where rep is the kernel of s's orbit.
The sign is new relative to the Ising code; it multiplies the Hamiltonian matrix element. =#
function find_rep_and_shift(s::Int, params::ModelParams, mask::Int)
    N, Ne = params.N, params.Ne
    q = translation_step(params)
    rep, shift, repsign = s, 0, 1
    t, sign = s, 1

    for r in 1:num_kx(params)
        t, sg = translate_r(t, q, N, Ne, mask)
        sign *= sg
        if t == s                       # completed the orbit
            break
        end
        if t < rep
            rep, shift, repsign = t, r, sign
        end
    end
    return rep, shift, repsign
end

#= Unfolds a vector written in a Bloch basis back into the plain bitstring basis `fock_states`
(e.g. fixed_N_states(N, Ne), or ky_sector_states(basis.Ky, params)). Used to check the block
Hamiltonian against the full one, and later for measuring operators in the bitstring basis. =#
function unfold_to_fock(psi::AbstractVector, basis::MomentumBasis, params::ModelParams,
                        fock_states::Vector{Int})
    N, Ne = params.N, params.Ne
    q = translation_step(params)
    mask = (1 << N) - 1
    k = 2π * basis.kx / num_kx(params)
    out = zeros(ComplexF64, length(fock_states))

    for (i, st) in enumerate(basis.states)
        t, sign = st.rep, 1
        for r in 0:(st.R - 1)
            idx = index_in(fock_states, t)
            idx == 0 && error("bitstring $t is missing from the supplied fock_states")
            out[idx] += psi[i] * sign * exp(-im * k * r) / sqrt(st.R)
            t, sg = translate_r(t, q, N, Ne, mask)
            sign *= sg
        end
    end
    return out
end
