#= Example driver: low-lying spectrum of Eq. (3) on a ring, block by block.
Run from this folder with:   julia run_ed.jl
Edit N and t below. N must be a multiple of 3 (ν = 1/3). N = 24 (the paper's size) takes a
little while; N = 15 or 18 is instant. =#

include("ConstructHamiltonian.jl")
using .ConstructLLHamiltonian
using Printf

N = 18
t = 1.0
numeigs = 4

params = params_from_t(N; t = t)
println("N = $N orbitals, Ne = $(params.Ne) electrons, t = $t, V10 = $(params.V10), ",
        "V20 = $(round(params.V20, digits = 4)), V30 = $(params.V30)")
println("Fixed-Ne Hilbert space: $(binomial(N, params.Ne)) states.")
println("Blocks (Ky, kx) with Ky, kx ∈ 0:$(num_kx(params) - 1). Each level also appears at Ky + Ne and Ky + 2Ne.\n")

Ky_a, kx_a = predicted_ground_sector(params)
println("Predicted ground-state block (sector a): Ky = $Ky_a (≡ $(mod(Ky_a, num_kx(params))) mod Ne), kx = $kx_a\n")

blocks = construct_block_hamiltonian(params)
lowest = Float64[]
for Ky in 0:(num_kx(params) - 1), kx in 0:(num_kx(params) - 1)
    blk = blocks[(Ky, kx)]
    dim = length(blk.basis.states)
    dim == 0 && continue
    vals, _ = lowest_eigs(blk.H; numeigs = numeigs)
    append!(lowest, vals)
    @printf("Ky = %2d  kx = %2d  dim = %6d   E = %s\n", Ky, kx, dim,
            join([@sprintf("%9.6f", v) for v in vals], " "))
end

sort!(lowest)
@printf("\nGround energy E0 = %.2e, lowest excitation = %.6f\n", lowest[1], lowest[findfirst(>(1e-8), lowest)])
