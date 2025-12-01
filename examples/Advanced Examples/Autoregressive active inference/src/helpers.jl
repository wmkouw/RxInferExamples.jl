using LinearAlgebra
using Distributions
using SpecialFunctions

function backshift(x::AbstractVector, a::Number)
    "Shift elements down and add element"

    N = size(x,1)

    # Shift operator
    S = Tridiagonal(ones(N-1), zeros(N), zeros(N-1))

    # Basis vector
    e = [1.0; zeros(N-1)]

    return S*x + e*a
end

function backshift(M::AbstractMatrix, a::Number)
    return diagm(backshift(diag(M), a))
end

function backshift(x::AbstractMatrix, a::Vector)
    return [a x[:,1:end-1]]
end

function proj2psd(S::AbstractMatrix)
    L,V = eigen(S)
    S = V*diagm(max.(1e-8,L))*V'
    return (S+S')/2
end