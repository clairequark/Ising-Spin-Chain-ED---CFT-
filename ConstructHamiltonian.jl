module ConstructHamiltonian

include("basis_construction.jl")

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



#= Takes bitstrings inputs and constructs the diagonal terms corresponding to the pauli Z term in the Hamiltonian.
Bit 0 = spin up , Bit 1 = spin down
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

This file diagonalizes H, the original Ising hamitlonian. =#

function construct_occupation_terms(params::ModelParams)
    J = params.J
    N = params.N 

    dim = 2^N
    rows, cols, vals = Int[], Int[], Float64[]


    for s in 0:(dim-1) #loop over the hilbert space basis states
        zsum = 0.0              #zsum is the overall sum of occupation energies

        for j in 0:(N-1) # loop over all sites
            bit = (s >> j) & 1  # Checks the ith bit for occupation. ; outputs 0 or 1 
            zsum += ( 1 - 2*bit )
        end
        zsum = J*zsum
        push!(rows, s+1)                 # Julia is 1 indexed. 
        push!(cols, s+1)                 # Expression of fact that occupation terms are diagonal. 
        push!(vals, zsum)
    end

    return sparse(rows, cols, vals, dim, dim)
end

#= Constructs interaction term in ising hamiltonian. 
The interaction term is a sum of nearest neighbor flips. Thus, for any input state, stored as a column
there are N (out of 2^N) rows populated with new, flipped vectors. Its kinda like a transfer matrix! Telling you 
how one spin state gets flipped into a sortment of new ones. The j loop handles the fact that the interaction is a sum. 
Since the rows correspond to the new states (which, a new state is really just a new label in an eigenbasis)
they get the new index! Simple, really. 
=#
function construct_interaction_terms_PBC(params::ModelParams)
    N = params.N
    g = params.g

    dim = 2^N 
    rows, cols, vals = Int[], Int[], Float64[]
    

    for s in 0:(dim-1)

        for j in 0:(N-1)

            j_next = (j+1) % N
            s_flipped = s ⊻ (1 << j) ⊻ (1 << j_next)           # Flips the bit at site j and site j+1 
            push!(rows, s_flipped + 1)
            push!(cols, s + 1)
            push!(vals, g)
        end
    end
    
    return sparse(rows, cols, vals, dim, dim)
end

function construct_interaction_terms_OBC(params::ModelParams)
    N = params.N
    g = params.g
    dim = 2^N 
    rows, cols, vals = Int[], Int[], Float64[]

    for s in 0:(dim-1)
        for j in 0:(N-2)
            j_next = (j+1)
            s_flipped = s ⊻ (1 << j) ⊻ (1 << j_next)           # Flips the bit at site j and site j+1 
            push!(rows, s_flipped + 1)
            push!(cols, s + 1)
            push!(vals, g)
        end
    end
    
    return sparse(rows, cols, vals, dim, dim)
end

#= Generates the full hamiltonian as a sparse matrix consisting of the occupation matrix elements 
and the interaction matrix elements. =#

function generate_hamiltonian(params::ModelParams; PBC::Int = 1)
    H_occ = construct_occupation_terms(params)

    if PBC == 1 
        H_int = construct_interaction_terms_PBC(params)
    else
        H_int = construct_interaction_terms_OBC(params)
    end
    return H_occ + H_int
end

#Constructs the translation operator (in bitstring basis) which acts as the rule "shift all spins over by one." With the lattice as your
# reference frame, this is just the passive transform of the lattice labels getting shifted to the right by one.
#for a spin state, this corresponds to shifting all of the spins in the state (circularly) by one. Since bits index High < - Low
#this corresponds to a left-wise bit rotation, kay? :3  
function construct_translation_operator(params::ModelParams; PBC::Int = 1)

    N = params.N 
    dim = 2^N 
    mask = (1 << N) - 1
    rows, cols, vals = Int[], Int[], Float64[]

    if PBC  == 0 
        print("Error: Crystal Momentum is not a good quantum number. Please check boundary conditions (PBC ==0)")
    end

    if PBC == 1
        for s in 0:(dim-1)
            s_rot = translate(s, N, mask)

            push!(rows, s_rot+1) # indexinggggg ; s_rot is the output so naturally it goes into the row :3
            push!(cols, s+1) #bitstrings are 0 indexed, julia arrays are one indexed, kay? 
            push!(vals, 1) #bitstrings are treated as normalized eigenvectors, so their norm is 1.  

        end
    end
    return sparse(rows, cols, vals, dim, dim)
end
#=Does partial, lowest energy eigenvalue computation for H stored as a sparse matrix. 
numeigs = number of eigenvalues to compute
issymetric reflects hermiticity of our hamiltonian. Set to false if energy is not conserved=#

function sparse_diagonalization(A::SparseMatrixCSC;  numeigs::Int)
    vals, vecs = eigs(A; nev=numeigs, which =:SR)
    return vals, vecs
end

#Does full diagonalization for H stored as a sparse matrix.

function dense_diagonalization(A::SparseMatrixCSC)
    A_dense = Matrix(A)
    eig = eigen(A_dense)
    return eig.values, eig.vectors
end


#do it term by term, just like last time. (IDK if i want this jawn!)
function construct_block_hamiltonian(params::ModelParams)

    N = params.N 
    blocks = Dict{Int, AbstractMatrix}()

    for k in 0:(N-1)

        basis = build_momentum_basis(k, params)
        H_occ = block_occupation_terms(basis, params)
        H_int = block_interaction_terms(basis, params)

        blocks[k] = H_occ + H_int
    end


    return blocks
end

#Constructs Z_j term in translationally invariant basis (k basis) for a specific momentum block
#Algorithm is streamlined. Abuses fact that Z_j is still diagonal in k basis, with eigenvalue given by a simple formula. 
function block_occupation_terms(basis::MomentumBasis, params::ModelParams)

    N = params.N

    nb = length(basis.states)
    H = spzeros(Float64, nb, nb)

    for i in 1:nb

        b = basis.states[i].rep

        n_down = count_ones(b) #bit = 0 is SPIN UP 
        n_up = N - n_down
        zsum = 2n_up - N
        H[i, i] = params.J * zsum 

    end

    return H
end


function block_interaction_terms(basis::MomentumBasis, params::ModelParams)

    N = params.N
    nb = length(basis.states)
    k = basis.k
    mask = (1 << N) - 1
    
    I = Int[]
    J = Int[]
    V = ComplexF64[]

    for i in 1:nb

        b = basis.states[i].rep
        R = basis.states[i].R

        for r in 0:R-1 

            
            s = translate_r(b, r, N, mask)

            for site in 0:(N-1) 
                
                site_next = (site + 1) % N
                s_flipped = s ⊻ (1 << site) ⊻ (1 << site_next)


                rep, shift = find_rep_and_shift(s_flipped, params, mask)

                if !haskey(basis.indexmap, rep)
                    continue
                end


                Rb2 = basis.states[basis.indexmap[rep]].R   # grab the R value for the state b2. (aka, the state after X_j X_j+1 act on the rth term of |b, k >)
                val = params.g * exp(im * 2*π * k * (shift - r)/N) / sqrt(R * Rb2)

                push!(I, basis.indexmap[rep])
                push!(J, i)
                push!(V, val) #need to work out precise form of matrix element 
            end
        end
    end

    return sparse(I, J, V, nb, nb)
end




end
