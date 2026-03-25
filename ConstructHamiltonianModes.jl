module ConstructHamiltonian

using SparseArrays 
using LinearAlgebra
using Plots
using Arpack

# Stores model paramers: Note J, and g should be -J and -g for the PSI Start group project hamiltonian. 
# The interesting thing about ED is that it constructs the algebraic structure of the hamiltonian in a specific basis. 
# When writing ED, you never have some abstract operator H that you eventually put into a basis. The aptly named "computational
# basis" is always assumed. Isn't that interesting? 
Base.@kwdef mutable struct ModelParams
    J::Float64
    g::Float64
    N::Int64
end

#=Takes bitstrings inputs and constructs the diagonal terms corresponding to the pauli Z term in the Hamiltonian. Bit 0 = spin up , Bit 1 = spin down" 
NOTE: THIS FILE PROBABLY NEEDS HEAVILY TRIMMED; contains redundant info from ConstructHamiltonian.jl

This time, we are constructing H_n, the fourier decomposed hamiltonian: 

Hₙ = -N/2π [∑ⱼᴺ exp(ijn 2π/N)σᶻⱼ + exp(i(j+1/2)n 2π/N)σˣⱼ σˣⱼ₊₁]

First term is contained in 'construct_occupation_terms', second term in 'construct_interaction_terms_PBC'

Friendly reminder your comp basis is a set, B := { Perm ( ⊗ᵢᴺ|↑> ), ↑ = 0 , ↓ = 1 } 
as bitstrings, this is simply {n = 0, 1, 2,...,2^N-1}4

NOTE.1: Need to remove PBC capability since critical CFTs only exist for PBC hamiltonians

NOTE.2: PBC leads to parity preserving block diagonal subsectors, but computationally only changes
the last term in the interaction, σ_N * σ_1 as opposed to truncating the sum at j = N - 1 

NOTE.3: Bits are ZERO indexed, julia is ONE indexed. Thus, a sum over all bits in a state
begins at 0, the ZEROTH bit, and ends at bit value 2^N - 1, the FINAL (leftmost bit)
=#
function construct_modal_occupation_terms(params::ModelParams, mode_index::int)
    J = params.J
    N = params.N 
    n = mode_index

    dim = 2^N
    rows, cols, vals = Int[], Int[], Float64[]


    for s in 0:(dim-1) #loop over the hilbert space basis states
        zsum = 0.0              #zsum is the overall sum of occupation energies

        for j in 0:(N-1) # loop over all sites
            bit = (s >> j) & 1  # Checks the ith bit for occupation. ; outputs 0 or 1 

            phase = exp(2im * j * 2 * π * n /N) # from formula (28) of arxiv 1706.01436
            zsum += ( 1 - 2*bit ) * phase
        end
        zsum = J*zsum
        push!(rows, s+1)                 # Julia is 1 indexed. 
        push!(cols, s+1)                 # Expression of fact that occupation terms are diagonal. 
        push!(vals, zsum)
    end

    return sparse(rows, cols, vals, dim, dim)
end

function construct_modal_interaction_terms(params::ModelParams, mode_index::int)
    N = params.N
    g = params.g
    n = mode_index

    dim = 2^N 
    rows, cols, vals = Int[], Int[], Float64[]
    

    for s in 0:(dim-1)

        for j in 0:(N-1)
            
            phase = exp(2im * (j +(1/2)) * n * π / N)

            j_next = (j+1) % (N)                              # mod N gets PBC term, which in this index convention is: σ_N-1 * σ_N = σ_N-1 * σ_0 
            s_flipped = s ⊻ (1 << j) ⊻ (1 << j_next)           # Flips the bit at site j and site j+1 
            push!(rows, s_flipped + 1)
            push!(cols, s + 1)
            push!(vals, g*phase)
        end
    end
    
    return sparse(rows, cols, vals, dim, dim)
end


#=Generates the fourier modes of the hamiltonian
 as a sparse matrix. Minus sign for ising model is absorbed into couplings=#

function generate_H_n(params::ModelParams, mode_index::int) 

    N = params.N

    H_occ = construct_modal_occupation_terms(params, mode_index)

    H_int = construct_modal_interaction_terms(params, mode_index)

    return (N * /(2*π))*(H_occ + H_int)
end



end