#= Sanity checks for the Landau-orbital Hamiltonian, Eq. (3) of Rahmani et al.
Run from this folder with:   julia test_hamiltonian.jl
Every check is small (N = 12 and 15), so this takes seconds. Requires Arpack (as the Ising code did). =#

include("ConstructHamiltonian.jl")
using .ConstructLLHamiltonian
using LinearAlgebra
using SparseArrays
using Test

# Columns of P are the Bloch states of `basis`, written out in the bitstring basis `fock_states`.
function bloch_matrix(basis, params, fock_states)
    nb = length(basis.states)
    P = zeros(ComplexF64, length(fock_states), nb)
    for i in 1:nb
        e = zeros(ComplexF64, nb)
        e[i] = 1
        P[:, i] = unfold_to_fock(e, basis, params, fock_states)
    end
    return P
end

spectrum(H) = eigvals(Hermitian(Matrix(H)))

# The squeezed configuration of Eq. (4) with the first block squeezed: 1001 -> 0110 on orbitals 0..3.
first_squeeze(params) = root_state(params) ⊻ 15          # 15 = 0b1111 as an Int

t = 0.8     # generic value; V20 then follows Eq. (2)

@testset "N = $N (Ne = $(N ÷ 3), Ne $(isodd(N ÷ 3) ? "odd" : "even"))" for N in (12, 15)
    params = params_from_t(N; t = t)
    Ne, N0 = params.Ne, num_kx(params)
    Hfull, fock = generate_hamiltonian(params)
    Efull = spectrum(Hfull)

    @testset "full bitstring-basis H" begin
        @test length(fock) == binomial(N, Ne)
        @test maximum(abs, Hfull - Hfull') < 1e-12                       # Hermitian
        T = construct_translation_operator(params, fock)
        @test maximum(abs, Hfull * T - T * Hfull) < 1e-12                 # [H, T] = 0 with fermion signs
        @test maximum(abs, T' * T - I) < 1e-12                            # T is unitary
    end

    blocks = construct_block_hamiltonian(params; Ky_list = 0:(N - 1))     # every block

    @testset "Bloch states: P'P = 1 and P' H P = H_block" begin
        worst_iso, worst_H = 0.0, 0.0
        for ((Ky, kx), blk) in blocks
            length(blk.basis.states) == 0 && continue
            P = bloch_matrix(blk.basis, params, fock)
            worst_iso = max(worst_iso, maximum(abs, P' * P - I))
            worst_H = max(worst_H, maximum(abs, P' * Hfull * P - Matrix(blk.H)))
        end
        @test worst_iso < 1e-12
        @test worst_H < 1e-12
    end

    @testset "blocks reproduce the full spectrum" begin
        @test sum(length(blk.basis.states) for blk in values(blocks)) == binomial(N, Ne)
        Eblocks = sort(vcat([spectrum(blk.H) for blk in values(blocks) if length(blk.basis.states) > 0]...))
        @test maximum(abs, Eblocks - Efull) < 1e-10
    end

    @testset "exact 3-fold center-of-mass degeneracy" begin
        for Ky in 0:(N - 1), kx in 0:(N0 - 1)
            A, B = blocks[(Ky, kx)].H, blocks[(mod(Ky + Ne, N), kx)].H
            @test size(A) == size(B)
            size(A, 1) > 0 && @test maximum(abs, spectrum(A) - spectrum(B)) < 1e-10
        end
    end

    @testset "ν = 1/3 ground state (PBC)" begin
        @test maximum(abs, Efull[1:3]) < 1e-10                           # E = 0, frustration free
        @test Efull[4] > 1e-3                                             # ...and gapped
        Ky_a, kx_a = predicted_ground_sector(params)
        @test kx_a == (isodd(Ne) ? 0 : N0 ÷ 2)
        Ea = spectrum(blocks[(Ky_a, kx_a)].H)
        @test abs(Ea[1]) < 1e-10
        @test Ea[2] > 1e-3                                                # unique in its block

        # Amplitudes of Eq. (4): squeezing one block multiplies the amplitude by -t.
        Hs, sec = generate_hamiltonian(params; Ky = Ky_a)
        vals, vecs = lowest_eigs(Hs; numeigs = 2)
        psi = vecs[:, 1]
        @test abs(vals[1]) < 1e-10
        ratio = psi[index_in(sec, first_squeeze(params))] / psi[index_in(sec, root_state(params))]
        @test ratio ≈ -t atol = 1e-10
    end
end

@testset "OBC (paper's circuit geometry), N = 12" begin
    params = params_from_t(12; t = t, PBC = 0)
    Ky_a = total_Ky(root_state(params), params)                          # = 0 + 3 + 6 + 9 = 18
    @test Ky_a == 18
    H, sec = generate_hamiltonian(params; Ky = Ky_a)
    @test maximum(abs, H - H') < 1e-12
    vals, vecs = lowest_eigs(H; numeigs = 2)
    @test abs(vals[1]) < 1e-10                                            # unique zero mode
    @test vals[2] > 1e-3
    psi = vecs[:, 1]
    ratio = psi[index_in(sec, first_squeeze(params))] / psi[index_in(sec, root_state(params))]
    @test ratio ≈ -t atol = 1e-10
end
