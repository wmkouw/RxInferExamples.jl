module Positioning

using Distributions
using LinearAlgebra

export XYZStage

mutable struct XYZStage

    Δt :: Float64

    # Dynamic parameters
    m :: Vector{Real}          # Mass for X, Y, Z [kg]
    c :: Vector{Real}          # Damping [N·s/m]
    k :: Vector{Real}          # Stiffness [N/m]

    # Cross-axis coupling terms
    c_cross :: Vector{Real}    # Damping coupling [N·s/m] (c_xy, c_xz, c_yz)
    k_cross :: Vector{Real}    # Stiffness coupling [N/m] (k_xy, k_xz, k_yz)

    Q  :: Matrix{Float64}
    R  :: Matrix{Float64}

    control_lims ::Tuple{Float64,Float64}

    function XYZStage(ρ::Vector, 
                      σ::Vector; 
                      mass::Vector=[1.0,1.0,1.0],
                      damping::Vector=[0.1, 0.1, 0.1],
                      stiffness:Vector=[10.0, 10.0, 10.0],
                      cross_damping::Vector=[0.01, 0.01, 0.01],
                      cross_stiffness:Vector=[0.5, 0.5, 0.5],
                      Δt::Float64=1.0, 
                      control_lims=(-1.,1.)
                      )

        # Process noise covariance matrix
        Q = [Δt^3/3*σ[1]          0.0  Δt^2/2*σ[1]          0.0;
                     0.0  Δt^3/3*σ[2]          0.0  Δt^2/2*σ[2];
             Δt^2/2*σ[1]          0.0      Δt*σ[1]          0.0;
                     0.0  Δt^2/2*σ[2]          0.0      Δt*σ[2]]

        # Measurement noise covariance matrix
        R = diagm(ρ)
       
        return new(Δt,mass,damping,stiffness,coupling_damping,coupling_stiffness)
    end
end

function step(system::XYZStage, state, input)
    "Stochastic state transition"

    # Compute accelerations with cross-coupling
    x_accel = (input[1] - c[1]*x_dot[i] - c_cross[1]*y_dot[i] - c_cross[2]*z_dot[i]
               - k[1]*x[i] - k_cross[1]*y[i] - k_cross[2]*z[i]) / m[1]

    y_accel = (input[2] - c_cross[1]*x_dot[i] - c[2]*y_dot[i] - c_cross[3]*z_dot[i]
               - k_cross[1]*x[i] - k[2]*y[i] - k_cross[3]*z[i]) / m[2]

    z_accel = (input[3] - c_cross[2]*x_dot[i] - c_cross[3]*y_dot[i] - c[3]*z_dot[i]
               - k_cross[2]*x[i] - k_cross[3]*y[i] - k[3]*z[i]) / m[3]

    # Update velocities and positions using Euler method
    state[4] = state[1] + x_accel * system.Δt
    state[5] = state[2] + y_accel * system.Δt
    state[6] = state[3] + z_accel * system.Δt

    state[1] = state[1] + state[4] * system.Δt
    state[2] = state[2] + state[5] * system.Δt
    state[3] = state[3] + state[6] * system.Δt
    
end

function emit(system::XYZStage, state)
    "Stochastic observation"

    # Mask velocities
    M = [1. 0. 0. 0. 0. 0.;
         0. 1. 0. 0. 0. 0.;
         0. 0. 1. 0. 0. 0.]

    return rand(MvNormal(M*state, system.R))
end

function update(system::XYZStage, state, input)
    "Update environment" 
     
    # State transition
    state = step(system, state, input)
     
    # Emit noisy observation
    observation = emit(system, state)
     
    return observation, state
end

end