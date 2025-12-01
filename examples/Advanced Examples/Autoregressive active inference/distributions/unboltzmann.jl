export unBoltzmann

import BayesBase
import Distributions
import RxInfer
using Optim
using DomainSets
using LinearAlgebra
using SpecialFunctions


struct unBoltzmann <: ContinuousMultivariateDistribution
 
    G::Function     # Energy function
    N::Integer      # Number of input arguments of energy function
    D::Rectangle    # Support of Boltzmann distribution

    function unBoltzmann(G::Function, N::Integer, D::Rectangle)
        return new(G,N,D)
    end
end

BayesBase.ndims(d::unBoltzmann) = d.N
BayesBase.support(d::unBoltzmann) = d.D

function BayesBase.mode(dist::unBoltzmann; time_limit=10., show_trace=false, iterations=1000)
    "Use optimization methods to find maximizer"

    opts = Optim.Options(time_limit=time_limit, 
                         show_trace=show_trace, 
                         allow_f_increases=true, 
                         outer_iterations=iterations,
                         iterations=10)
    @debug opts

    gradG(J,u) = ForwardDiff.gradient!(J,dist.G,u)
    results = optimize(dist.G, gradG, support(dist).a, support(dist).b, 1e-8*randn(dist.N), Fminbox(LBFGS()), opts)
    return Optim.minimizer(results)
    # end
end

BayesBase.cov(dist::unBoltzmann) = inv(precision(dist))

function BayesBase.precision(dist::unBoltzmann)
    "Laplace approximated precision matrix"
    P_laplace = ForwardDiff.hessian(dist.G, mode(dist))
    return proj2psd(P_laplace);
end

function pdf(dist::unBoltzmann, u::Vector)
    "Evaluate exponentiated energy function"
    return exp(-dist.G(u))
end

function Distributions.logpdf(dist::unBoltzmann, u::Vector)
    "Evaluate energy function"
    return -dist.G(u)
end

BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:unBoltzmann})      = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:AbstractMvNormal}, ::Type{<:unBoltzmann}) = BayesBase.ClosedProd()
BayesBase.default_prod_rule(::Type{<:unBoltzmann}, ::Type{<:AbstractMvNormal}) = BayesBase.ClosedProd()

function BayesBase.prod(::BayesBase.ClosedProd, left::unBoltzmann, right::unBoltzmann)    
    if left.N != right.N; error("Dimensionalities of energy functions do not match."); end
    G(u) = left.G(u) + right.G(u)
    return unBoltzmann(G,right.N, intersectdomain(left.D, right.D))
end

function BayesBase.prod(::BayesBase.ClosedProd, left::AbstractMvNormal, right::unBoltzmann)    
    if ndims(left) != right.N; error("Dimensionality of Gaussian and number of inputs of energy function do not match."); end
    G(u) = -BayesBase.logpdf(left,u) + right.G(u)
    return unBoltzmann(G,right.N,right.D)
end

BayesBase.prod(::BayesBase.ClosedProd, left::unBoltzmann, right::AbstractMvNormal) = BayesBase.prod(ClosedProd(), right, left)    

