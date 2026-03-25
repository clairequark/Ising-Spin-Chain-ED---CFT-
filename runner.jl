include("ConstructHamiltonian.jl")
include("ConstructOperators.jl")


#gs to 0 -> fixes b 
# diag mom. in energy basis. 
#fix constant "a" find lowest energy state with momentum 2 or minus 2 , rescale that lowest energy state to have energy 2, precisely. 
# this fixes a. 

using .ConstructHamiltonian
using .ConstructOperators
using LinearAlgebra
using Plots
using LaTeXStrings

#Gets ground state energy for spin chain of size N

params = ConstructHamiltonian.ModelParams(J = -1.0, g = -1.00, N = 15)

#construct the unnormalized hamiltonian in the bitstring basis 
H1 = ConstructHamiltonian.generate_hamiltonian(params)

#diagonalize hamiltonian in this eigenbasis ; h prefix stands for eigvals of h. 
heigvals, eigvecs = ConstructHamiltonian.sparse_diagonalization(H, numeigs = 40)

T = ConstructHamiltonian.construct_translation_operator(params)

# constructs partial unitary for numeigs subspace diagonalization of T
Vk = eigvecs 

Tsub = Vk' * T * Vk 
# t prefix stands for eigs of T. teigvecs is actually just the coefficient array. These coefficients correspond to 
# to a superposition of the original H eigenvectors (as expected) ; to get the physical simultaneous eigenvector of T and H 
# you simply construct | ϕ > = Vk * teigvecs. 
teigvals, teigvecs = eigen(T)

shifted_eigvals = eigvals .- minimum(eigvals)

plot(
    shifted_eigvals,
    seriestype = :scatter,
    marker = :circle,
    xlabel = "Eigenstate index",
    ylabel = "Energy",
    title = "Energy Spectrum",
    legend = false
)

savefig("energyspectrum_1_18_26_g=$(-params.g).png")
println(minimum(eigvals))

print((eigvals[2] - eigvals[3])/(-1*params.g))

