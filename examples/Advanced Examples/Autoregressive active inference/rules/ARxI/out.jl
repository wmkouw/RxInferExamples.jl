@rule ARxI(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal},
                                   q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                   q_in::PointMass,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart) = begin

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
      
    return MvLocationScaleT(η,μ,Σ)
end

@rule ARxI(:out, Marginalisation) (q_outprev1::PointMass, 
                                   q_outprev2::PointMass, 
                                   m_in::unBoltzmann,
                                   q_inprev1::PointMass, 
                                   q_inprev2::PointMass,
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(m_in); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule ARxI(:out, Marginalisation) (m_outprev1::Union{AbstractMvNormal,MvLocationScaleT}, 
                                   q_outprev2::PointMass,
                                   m_in::Union{AbstractMvNormal,unBoltzmann},
                                   m_inprev1::Union{PointMass,unBoltzmann}, 
                                   q_inprev2::Union{PointMass,unBoltzmann},
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(q_outprev2); mode(m_in); mode(m_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule ARxI(:out, Marginalisation) (m_outprev1::Union{AbstractMvNormal,MvLocationScaleT}, 
                                   m_outprev2::Union{AbstractMvNormal,MvLocationScaleT},
                                   m_in::Union{AbstractMvNormal,MvLocationScaleT,unBoltzmann}, 
                                   m_inprev1::Union{AbstractMvNormal,MvLocationScaleT,unBoltzmann}, 
                                   m_inprev2::Union{AbstractMvNormal,MvLocationScaleT,unBoltzmann},
                                   m_Φ::MatrixNormalWishart,) = begin 

    # Extract parameters 
    M,Λ,Ω,ν = params(m_Φ)

    # Construct buffer vector
    x = [mode(m_outprev1); mode(m_outprev2); mode(m_in); mode(m_inprev1); mode(m_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
    
    return MvLocationScaleT(η,μ,Σ)
end

@rule ARxI(:out, Marginalisation) (q_outprev1::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                   q_outprev2::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                   q_in::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                   q_inprev1::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                   q_inprev2::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                   q_Φ::MatrixNormalWishart, ) = begin
    # Extract parameters 
    M,Λ,Ω,ν = params(q_Φ)

    # Construct buffer vector
    x = [mode(q_outprev1); mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

    # Parameters of multivariate location-scale T posterior predictive distribution
    η = ν - Dy + 1
    μ = M'*x
    Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)

    return MvLocationScaleT(η,μ,Σ)
end                                   

@rule ARxI(:outprev1, Marginalisation) (q_out::unBoltzmann, 
                                        q_outprev2::PointMass, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::AbstractMvNormal, 
                                        q_inprev2::PointMass, 
                                        q_Φ::MatrixNormalWishart, ) = begin

    M,Λ,Ω,ν = params(q_Φ)
    m_star  = mode(q_out)
    Du = length(mode(q_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))                                
end

@rule ARxI(:outprev1, Marginalisation) (q_out::unBoltzmann, 
                                        q_outprev2::PointMass, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::PointMass, 
                                        q_Φ::MatrixNormalWishart, ) = begin        
    
    M,Λ,Ω,ν = params(q_Φ)
    m_star  = mode(q_out)
    Du = length(mode(q_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end                                           

@rule ARxI(:outprev1, Marginalisation) (q_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::Union{PointMass,AbstractMvNormal},
                                        q_inprev1::Union{PointMass,AbstractMvNormal}, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal},
                                        m_Φ::MatrixNormalWishart) = begin
 
    M,Λ,Ω,ν = params(m_Φ)
    m_star  = mean(q_out)
    Du = length(mode(q_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev1, Marginalisation) (q_out::Union{PointMass,AbstractMvNormal,unBoltzmann}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_in::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_inprev1::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_inprev2::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(q_Φ)
    m_star  = mode(q_out)
    Du = length(mode(q_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(q_in); mode(q_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev1, Marginalisation) (m_out::Union{AbstractMvNormal,MvLocationScaleT}, 
                                        q_outprev2::Union{PointMass,AbstractMvNormal}, 
                                        m_in::Union{PointMass,AbstractMvNormal},
                                        m_inprev1::Union{PointMass,unBoltzmann}, 
                                        q_inprev2::Union{PointMass,unBoltzmann},
                                        m_Φ::MatrixNormalWishart) = begin

    M,Λ,Ω,ν = params(m_Φ)
    m_star  = mean(m_out)
    Du = length(mode(m_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(m_in); mode(m_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev1, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev2::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::AbstractMvNormal, 
                                        m_inprev2::AbstractMvNormal, 
                                        m_Φ::MatrixNormalWishart, ) = begin 
    
    M,Λ,Ω,ν = params(m_Φ)
    m_star  = mean(m_out)
    Du = length(mode(m_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(m_outprev2); mode(m_in); mode(q_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev1, Marginalisation) (m_out::MvNormalMeanCovariance, 
                                        q_outprev2::PointMass, 
                                        m_in::Union{PointMass,unBoltzmann}, 
                                        m_inprev1::Union{PointMass,unBoltzmann}, 
                                        q_inprev2::Union{PointMass,unBoltzmann},
                                        m_Φ::MatrixNormalWishart,) = begin 
    
    M,Λ,Ω,ν = params(m_Φ)
    m_star  = mean(m_out)
    Du = length(mode(m_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(q_outprev2); mode(m_in); mode(m_inprev1); mode(q_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev1, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev2::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::Union{PointMass,unBoltzmann}, 
                                        m_inprev2::Union{PointMass,unBoltzmann}, 
                                        m_Φ::MatrixNormalWishart, ) = begin 

    M,Λ,Ω,ν = params(m_Φ)
    m_star  = mean(m_out)
    Du = length(mode(m_in))
    
    function G(outprev1)

        # Construct buffer vector
        x = [outprev1; mode(m_outprev2); mode(m_in); mode(m_inprev1); mode(m_inprev2)]

        # Multivariate location-scale T posterior predictive distribution
        η = ν - Dy + 1
        μ = M'*x
        Σ = 1/(ν-Dy+1)*Ω*(1 + x'*inv(Λ)*x)
        return logpdf(MvLocationScaleT(η,μ,Σ), m_star)
    end
    return unBoltzmann(G,Dy,ProductDomain([(-Inf..Inf) for i in 1:Du]))
end

@rule ARxI(:outprev2, Marginalisation) (q_out::AbstractMvNormal, 
                                        q_outprev1::unBoltzmann, 
                                        q_in::Union{PointMass,unBoltzmann}, 
                                        q_inprev1::Union{PointMass,unBoltzmann}, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin
    return Uninformative()
end

@rule ARxI(:outprev2, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev1::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::AbstractMvNormal, 
                                        m_inprev2::AbstractMvNormal, 
                                        m_Φ::MatrixNormalWishart, ) = begin 

    return Uninformative()
end

@rule ARxI(:outprev2, Marginalisation) (m_out::AbstractMvNormal, 
                                        m_outprev1::AbstractMvNormal, 
                                        m_in::AbstractMvNormal, 
                                        m_inprev1::unBoltzmann, 
                                        m_inprev2::unBoltzmann, 
                                        m_Φ::MatrixNormalWishart, ) = begin 
    
    return Uninformative()
end

@rule ARxI(:outprev2, Marginalisation) (q_out::AbstractMvNormal, 
                                        q_outprev1::Union{PointMass,AbstractMvNormal}, 
                                        q_in::unBoltzmann, 
                                        q_inprev1::unBoltzmann, 
                                        q_inprev2::Union{PointMass,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 
   
    return Uninformative()
end

@rule ARxI(:outprev2, Marginalisation) (q_out::AbstractMvNormal, 
                                        q_outprev1::AbstractMvNormal, 
                                        q_in::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_inprev1::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_inprev2::Union{PointMass,unBoltzmann,AbstractMvNormal}, 
                                        q_Φ::MatrixNormalWishart, ) = begin 

    return Uninformative()
end
