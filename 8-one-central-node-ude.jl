
# File Handling
using CSV, DataFrames, Dates, JSON, JLD2, FileIO

# SciML Tools
using DifferentialEquations, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, DataInterpolations
using StatsBase, Distributions, KernelDensity

# External Libraries
using ComponentArrays, Lux, Zygote, StableRNGs



using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")


# Seed stable random number generator
rng = StableRNG(42)



# load in data for central node
datapathbase = "./data/processed"
figpath = "./figures"


figpath = joinpath(figpath, "central-nodes")
fig_savepath = joinpath(figpath, "central-hub-4", "1s-cont-UDE")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


datapath = joinpath(datapathbase, "central-nodes", "central-hub-4", "df-1-sec.csv")
datasummarypath = joinpath(datapathbase, "central-nodes", "central-hub-4", "df-1-sec_summary.csv")


# load in a csv and it's associated summary
t1 = Date(2023, 5, 29);
t2 = Date(2023, 6, 8);
col_to_use = :pm2_5

# set up ticks for plots
x_tick= (round(t1, Day)-Day(1)):Day(1):(round(t2, Day) + Day(1))
x_tick_pos = [Second(d - t1).value for d ∈ x_tick]
x_tick_strings = [Dates.format.(d, "dd/mm/yy") for d ∈ x_tick]


Zs, ts = load_data(datapath, t1, t2, col_to_use)
dt = ts[2] - ts[1]
@assert dt == 1.0


# visualize the time-series
fig = Figure();
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="t",
    ylabel="PM 2.5 (μg/m³)",
    xticks=(x_tick_pos, x_tick_strings),
    xticklabelrotation=π/3,
    xticklabelsize = 12,
)
lines!(ax, ts, Zs)
fig


# construct Hankel Matrix
H = TimeDelayEmbedding(Zs; method=:backward);


# Decompose via SVD
U, σ, V = svd(H);
@assert all(H .≈ U*Diagonal(σ)*V')  # verify that decomposition works


# Define parameters for HAVOK
nmin = 1
n_embedding = max(round(Int, nmin*60 / dt), 100)
# r_cutoff = r_expvar(σ, cutoff=0.975)
# n_control = 1
r_cutoff = 18
n_control = 10
r = r_cutoff + n_control - 1

# truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]

V_small = @view V[:, 1:r-n_control]
U_small = @view U[:, 1:r-n_control]
σ_small = @view σ[1:r-n_control]


# examine the reconstruction
let
    println("Verify SVD approximates orignal to 95% rel acc")
    Ĥ = Ur * diagm(σr) * Vr'
    Ĥ2 = U_small * diagm(σ_small) * V_small'

    # we are able to maintain the
    println("Ĥ: ", isapprox(Ĥ, H, rtol=0.05))
    println("Ĥ2: ", isapprox(Ĥ2, H, rtol=0.05))
end


# compute derivative using fourth order central difference scheme
X = Vr
dX = zeros(size(Vr,1), r-n_control)
ts_x = range(ts[n_embedding], step=dt, length=size(Vr,1))
# use cubic spline to get derivatives instead...


size(X)
size(dX)

for j ∈ axes(dX, 2)
    itp = CubicSpline(X[:,j], ts_x)
    for i ∈ axes(dX, 1)
        dX[i,j] = DataInterpolations.derivative(itp, ts_x[i])
    end
end



# use 9 days for training and the rest for testing
t_train_end= dt*(9*24*60*60)
idx_train = 1:findfirst(ts .≥ t_train_end)
idx_test = (idx_train[end] + 1) : length(ts_x)
idx_train_short = 1:60*60

ts_train  = ts_x[idx_train]
ts_train_short  = ts_x[idx_train_short]
ts_test = ts_x[idx_test]


# partition test/train split
Xtrain = X[idx_train,:];
dXtrain = dX[idx_train,:];
Xtrain_short = X[idx_train_short,:];
dXtrain_short = dX[idx_train_short,:];
Xtest = X[idx_test, :];
dXtest = dX[idx_test, :];

# Compute model matrix via least squares
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B

out_dict = Dict(
    :A => A,
    :B => B,
)

open(joinpath(fig_savepath, "havok_params.json"), "w") do f
    JSON.print(f, out_dict)
end


# visualize learned matrix
heatmap(A)


# create interpolant for the forcing term
#   +3 because of offset from derivative...
itps = [DataInterpolations.LinearInterpolation(X[:,j], ts_x; extrapolate=true) for j ∈ r-n_control+1:r]
forcing(t) = [itp(t) for itp ∈ itps]

# test out forcing function
forcing(ts_x[10])
forcing.(ts_x[10:20])

# Set up UDE
rbf(x) = exp.(-(x .^ 2))

act_func = rbf
# act_func = relu
# act_func = tanh


# Multilayer Feed-Forward Network
n_hidden = round(Int, 2 * n_control)

const NN = Lux.Chain(
    Lux.Dense(n_control, n_hidden,  act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, r - n_control)
)

# Get initial parameters and state of NN
p, st = Lux.setup(rng, NN)
const _st = st

# test the neural network functionality
NN(X[1, r-n_control+1:end], p, st)[1]


# Define UDE as dynamical system u̇ = known(u) + NN(u)
# p are NN paramters, p_true are known model coefficients, i.e. α, δ
function ude_dynamics!(dx, x, p, t, A)
    nn = NN(forcing(t), p, _st)[1]  # NN prediction
    dx .= A*x + nn
end


# Set up closure to handle extra arguments
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, A)

# set up ODE problem
x₀ = X[1,1:r-n_control]
ts_train_short = ts[idx_train_short]
tspan = (ts_train_short[1], ts_train_short[end])
prob_nn = ODEProblem(nn_dynamics!, x₀, tspan, p)

# test solve
solve(prob_nn);


# set up function for generating solution with new parameters
function predict(θ, x0_new = X[1, 1:r-n_control], T = ts_train_short)
    _prob = remake(prob_nn, u0=x0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

predict(p)


function embedd_to_timeseries(Xpred, U_small, σ_small)
    Ĥ = U_small * diagm(σ_small) * X̂
    return Ĥ[end, :]
end


# compute loss by reconstructing original time series
Z_orig = H[end, idx_train_short]
# verify recovery of original time series from SVD
Ĥ2 = Ur * diagm(σr) * Vr'
Z_embedd = Ĥ2[end, idx_train_short]
# time series reconstruction from embedding
X̂ = predict(p)
Z_pred = embedd_to_timeseries(X̂, U_small, σ_small)


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t (min)", ylabel="PM 2.5 (μg/m³)")
lines!(ax, ts_train_short ./ 60, Z_orig)
lines!(ax, ts_train_short ./ 60, Z_pred2)
lines!(ax, ts_train_short ./ 60, Z_pred)
fig



function loss(θ)
    X̂ = predict(θ)
    Z_pred = embedd_to_timeseries(X̂, U_small, σ_small)

    mean(abs2, Z_pred .- Z_orig)
end

loss(p)


# set up array to track loss function with callback
losses = Float64[]

function callback(p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end

    return false  # true signals early stopping
end


# Build Optimization Problem for Training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
# ComponentVector is used here to flatten the parameters in an easy way with views
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))


# Train first using ADAM for rapid convergence to local minimum
# this step may take a while...
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=5000)

save(joinpath(fig_savepath, "model_params.jld2"), Dict(:ps_trained => res1.u, :st => _st))


# Train second round using LBFGS to get better minimum using derivative information
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=1000)

save(joinpath(fig_savepath, "model_params.jld2"), Dict(:ps_trained => res2.u, :st => _st))

# Rename the best candidate
p_trained = res2.u








