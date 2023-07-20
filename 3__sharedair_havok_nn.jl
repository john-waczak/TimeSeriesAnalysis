using CSV, DataFrames
using Plots, StatsPlots
using TimeSeriesTools
using Statistics, StatsBase, Distributions, KernelDensity
using LinearAlgebra
using DataInterpolations
using StaticArrays

# see: https://docs.sciml.ai/Overview/stable/showcase/missing_physics/#Definition-of-the-Universal-Differential-Equation
using DifferentialEquations, SciMLSensitivity, DiffEqFlux
using Zygote
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ModelingToolkit, DataDrivenDiffEq, DataDrivenSparse
using Lux, ComponentArrays
using StableRNGs


# set random seed for reproduciblity
rng = StableRNG(42)


include("plot_defaults.jl")
add_mints_theme()
theme(:mints)

include("plot_recipes.jl")
include("utils.jl")

if !ispath("figures/sharedair/havok")
    mkpath("figures/sharedair/havok")
end


# 0. Load in data and set window size (for ncol argument)
println("loading in the data...")

function get_data(path, cols_to_use)
    df = CSV.File("data/sharedair/data.csv") |> DataFrame
    ts = df.t
    return ts, Matrix(df[:, cols_to_use])
end

#ts, Data = get_data("data/sharedair/data.csv", [:pm1_0, :pm2_5, :pm10_0])
ts, Data = get_data("data/sharedair/data.csv", [:pm2_5]) #, :pm2_5, :pm10_0])

size(ts)
size(Data)

Nwindow = 81 # determined in step 1 by range of variogram fit

# 1. Generate Hankel matrix for each individual time series and concatenate together
println("computing time delay embedding Hankel-Takens matrix...")
H = vcat([TimeDelayEmbedding(Data[:,i]; nrow=Nwindow, method=:backward) for i∈axes(Data,2)]...)

size(H)  # so we have 10 columns of data with 81 times each giving 810 elements per embedding vector


# 2. Decompose via SVD
println("computing SVD... this could take a while")
U, σ, V = svd(H)

println("U has dimensions ", size(U))
println("V has dimensions ", size(V))
println("we have $(length(σ)) singular values")


@assert all(H .≈ U*Diagonal(σ)*V')  # verify that decomposition works
# 3. compute cut-off value for singular values based on magnitude
r = r_cutoff(σ,ratio=0.005, rmax=30)


# r = r_optimal_approx(σ, size(V,2), size(V,1))
r = 18  # maybe just right?

# add extra dimensions to r for > 1 control variable
n_control = 10

r = r + n_control - 1


# 4. truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


# 5. compute derivative using fourth order central difference scheme
dt = mean(ts[2:end] .- ts[1:end-1])
@assert dt == 1.0

dVr = zeros(size(Vr,1)-5, r-n_control)

Threads.@threads for k ∈ 1:r-n_control
    for i ∈ 3:size(Vr,1)-3
        @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
    end
end

@assert size(dVr,2) == r-n_control


# 6. chop off edges to size of data matches size of derivative
X = @view Vr[3:end-3, :]
dX = @view dVr[:,:]
@assert size(dX,2) == size(X,2)  - n_control


ts = range(ts[3], step=dt, length=size(dVr,1))

# let's try it out on 3 days worth of data with 1 day of data for testing
tend1 = dt*(3*24*60*60)
tend2 = dt*(4*24*60*60)

# pinch time values for entire range: train + test
ts = ts[ts .< tend2]
L = 1:length(ts[ts .< tend1])
Ltest = L[end]+1:length(ts)

@assert length(L) + length(Ltest) == length(ts)

# chop things nicely
Xtest = X[Ltest, :]
dXtest = dX[Ltest, :]
X = X[L,:]
dX = dX[L,:]


# 7. Compute model matrix via least squares
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B


# now that we've fit these, A and B wont change

# 8. define interpolation function for forcing coordinate(s)
#     +3 because of offset from derivative...
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]

# 14. Integrate model forward in time
sA = @SMatrix[A[i,j] for i ∈ axes(A,1), j ∈ axes(A,2)]
sB = @SMatrix[B[i,j] for i ∈ axes(B,1), j ∈ axes(B,2)]



# 9. Define NeuralNetworks
rbf(x) = exp.(-(x .^2))  # rbf activation function

dim_in = r-n_control
dim_out = r-n_control
dim_hidden = 2*max(dim_in, dim_out)

# NN(x)  for nonlinear dynamics
U₁ = Lux.Chain(
    Lux.Dense(dim_in, dim_hidden, rbf),
    Lux.Dense(dim_hidden, dim_hidden, rbf),
    Lux.Dense(dim_hidden, dim_out)
)

# collect parameters and state of NN
p, st = Lux.setup(rng, U₁)


# define RHS function
function f!(dx, x, p, t, p_true)
    # form NN prediction
    NNx = U₁(x, p, st)[1] # NN prediction for nonlinear dynamics
    A,B = p_true

    # dx[i] .= 0.0  # zero everything out

    # for j ∈ axes(A,2)
    #     for i ∈ axes(A,1)
    #         dx[i]
    #     end
    # end


    dx .= A*x + NNx + B*u(t)
end



# define a closure with known parameters
nn_dynamics!(dx, x, p, t) = f!(dx, x, p, t, (sA,sB))


prob_nn = ODEProblem(nn_dynamics!, X[1, 1:r-n_control], (ts[L[1]], ts[L[end]]), p)


# we should specify an X₀ and npoints so that we don't spend so much time solving, perhaps split up X, ts[L] into batches of size 1000 or so...
function predict(θ)
    _prob = remake(prob_nn, p = θ)
    Array(solve(_prob, saveat = ts, abstol = 1e-6, reltol = 1e-6))
end

predict(p)



function loss(θ)
    X̂ = predict(θ)

    # return mean squared error, penalizing large errors...
    # for real numbers abs2(x) = x²
    mean(abs2, X'[1:r-n_control,:] .- X̂)
end



loss(p)

# U₁(u(ts[1]))

# # NN(x)
# U₂ = Lux.

# # NN(x,u)
# U₃


# # define function and integrate to get model predictions
# params = (sA, sB)
# x₀ = X[1,1:r-n_control]
# dx = copy(x₀)

# # A*x₀ + sB*u(ts[1])

# @assert size(x₀) == size(dx)
# @benchmark f!(dx, x₀, params, ts[1])


# # integrate over all times so we have model predictions on holdout data
# prob = ODEProblem(f!, x₀, (ts[1], ts[end]), params)
# sol = solve(prob, saveat=ts)# , abstol=1e-12, reltol=1e-12);
# size(sol)

# # split up solutions
# X̂ = sol[:,L]'
# X̂test = sol[:,Ltest]'

