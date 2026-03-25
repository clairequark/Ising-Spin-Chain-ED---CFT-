
module ConstructOperators


using LinearAlgebra
using SparseArrays


function net_magnetization_z_PBC(gstate::AbstractVector, N::Int)
    mz = 0.0 
    for s in 0:(length(gstate) - 1)
        zsum = 0.0 
        for j in 0:(N - 1)
            spin = ((s >> j) & 1 == 1) ? -1 : 1 

            zsum += (gstate[s+1]^2 * spin)
        end
        #@assert abs(zsum) <= 1 
        mz += abs(zsum) 
    end

    mz = mz/N

    return mz
end


function net_magnetization_z_Alt(gstate::AbstractVector, N::Int)
    mz = 0.0 
    for s in 0:(length(gstate) - 1)
        zsum = 0.0 
        prob = gstate[s+1]^2

        for j in 0:(N - 1)
            spin = ((s >> j) & 1 == 1) ? -1 : 1 

            zsum += (spin)
        end
        #@assert abs(zsum) <= 1 
        mz += (prob * zsum) 
    end

    mz = mz/N

    return mz
end


 
function transverse_spin_spin(gstate::AbstractVector, N::Int; PBC::Int = 1)
    spinsum = 0.0
    bc_complement = 1 - PBC
    for j in 0:(N - 1 - bc_complement)

        if PBC == 1
            jp1 = (j + 1) % N
        end

        if PBC == 0 
            jp1 = (j + 1)
        end

        sitespins = 0.0
        for s in 0:(length(gstate) - 1)

            s_flipped = s ⊻ (1<<j) ⊻ (1<<jp1)
            sitespins += gstate[s + 1] * gstate[s_flipped + 1]
        end

        spinsum += sitespins 
    end

    return spinsum/N
end

#these methods are NOT dependent on boundary conditions. r = subsystem size

function reduced_density_matrix(state::Vector{Float64}, r::Int, N::Int)
    dA = 2^r
    dB = 2^(N - r)
    state_matrix = reshape(state, (dA, dB))  # Reshape into matrix

    ρA = state_matrix * state_matrix'        # Trace over subsystem B
    return ρA                        # ρ_A is dA × dA Hermitian matrix
end

function von_neumann_entropy(ρ::Matrix{Float64})
    vals = eigen(ρ).values
    vals = vals[vals .> 0]          # Exclude zero eigenvalues
    return -sum(vals .* log.(vals))
end


end