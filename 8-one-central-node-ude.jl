# HAVOK Model for Single Central Node

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
datapath = "./data/processed"
@assert ispath(datapath)
figpath = "./figures"
@assert ispath(figpath)


figpath = joinpath(figpath, "central-nodes")
if !ispath(figpath)
    mkpath(figpath)
end

datapaths_cn = [
    joinpath(datapath, "central-nodes", "central-hub-4"),
    joinpath(datapath, "central-nodes", "central-hub-7"),
    joinpath(datapath, "central-nodes", "central-hub-10"),
]

fig_savepaths = [
    joinpath(figpath, "central-hub-4"),
    joinpath(figpath, "central-hub-7"),
    joinpath(figpath, "central-hub-10"),
]

if !any(ispath.(fig_savepaths))
    mkpath.(fig_savepaths)
end

@assert all(ispath.(datapaths_cn))
@assert all(ispath.(fig_savepaths))


time_types = [
    "1-sec",
    "1-min",
    "15-min",
    "1-hour",
]


f_to_use = 1
datapath_cn = datapaths_cn[f_to_use]
fig_savepath = joinpath(fig_savepaths[f_to_use], "ude-cont-9-day")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


# load in a csv and it's associated summary
t1 = Date(2023, 5, 29);
t2 = Date(2023, 6, 8);
col_to_use = :pm2_5

# set up ticks for plots
x_tick= (round(t1, Day)-Day(1)):Day(1):(round(t2, Day) + Day(1))
x_tick_pos = [Second(d - t1).value for d ∈ x_tick]
x_tick_strings = [Dates.format.(d, "dd/mm/yy") for d ∈ x_tick]

csv_path = joinpath(datapath_cn, "df-"*time_types[1]*".csv")
Zs, ts = load_data(csv_path, t1, t2, col_to_use)
dt = ts[2] - ts[1]
@assert dt == 1.0


# construct Hankel Matrix
H = TimeDelayEmbedding(Zs; method=:backward)


# Decompose via SVD
println("computing SVD... this could take a while")
U, s, V = svd(H)
Σ = diagm(s)
@assert all(H .≈ U*Σ*V')  # verify that decomposition works


# Define parameters for HAVOK
n_embedding = 100
n_derivative = 5
r_cutoff = r_expvar(s, cutoff=0.99)
# r_cutoff = 18
n_control = 1
r = r_cutoff + n_control - 1



# Verify our Indexing scheme for the Hankel Matrix 
fig = Figure();
ax = CairoMakie.Axis(fig[1,1]; xlabel="t", ylabel="PM 2.5")
l_orig = lines!(ax, ts[n_embedding:(n_embedding+100)], Zs[n_embedding:(n_embedding+100)])
l_H = lines!(ax, ts[n_embedding:(n_embedding+100)], H[1,1:101])
fig


# truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view s[1:r]


V_small = @view V[:, 1:r-n_control]
U_small = @view U[:, 1:r-n_control]
σ_small = @view s[1:r-n_control]


# visualize impact of truncation error from SVD
Z_test = U_small[1,:]' * diagm(σ_small) * V_small'
Z_test2 = Ur[1,:]' * diagm(σr) * Vr'

fig = Figure();
ax = CairoMakie.Axis(fig[1,1]);
lines!(ax, Zs[n_embedding:(n_embedding+25)], label="orig")
lines!(ax, Z_test[1:26], label="Z approx small")
lines!(ax, Z_test2[1:26], label= "Z approx approx")
axislegend(ax)
fig


# examine the reconstruction
let
    println("Verify SVD approximates orignal to 95% rel acc")
    Ĥ = Ur * diagm(σr) * Vr'
    Ĥ2 = U_small * diagm(σ_small) * V_small'

    # we are able to maintain the
    println("Ĥ: ", isapprox(Ĥ, H, rtol=0.05))
    println("Ĥ2: ", isapprox(Ĥ2, H, rtol=0.05))
end




# 7. compute derivative using fourth order central difference scheme
dt = mean(ts[2:end] .- ts[1:end-1])
@assert dt == 1.0

ts_x = range(ts[n_embedding], step=dt, length=size(Vr,1))
X = Vr
dX = zeros(size(Vr,1), r-n_control)

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
#idx_train_short = 1:60*60
idx_train_short = 1:(5*60)

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


# set up interpolation
#itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_x; extrapolate=true) for j ∈ r-n_control+1:r]
itps = [DataInterpolations.CubicSpline(Vr[:,j], ts_x; extrapolate=true) for j ∈ r-n_control+1:r]
forcing(t) = [itp(t) for itp ∈ itps]

# try out forcing function
forcing(ts_x[1])



# Set up UDE
rbf(x) = exp.(-(x .^ 2))

# act_func = rbf
act_func = relu
# act_func = tanh


# Multilayer Feed-Forward Network
n_hidden = round(Int, 2 * n_control)

const NN = Lux.Chain(
    Lux.Dense(r - n_control, n_hidden,  act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, r - n_control)
)

# const NN = Lux.Chain(
#     Lux.Dense(n_control, n_hidden,  act_func),
#     Lux.Dense(n_hidden, n_hidden, act_func),
#     Lux.Dense(n_hidden, n_hidden, act_func),
#     Lux.Dense(n_hidden, r - n_control)
# )



# Get initial parameters and state of NN
p, st = Lux.setup(rng, NN)
const _st = st

# test the neural network functionality
# NN(X[1, r-n_control+1:end], p, st)[1]
NN(X[1, 1:r - n_control], p, st)[1]


# Define UDE as dynamical system u̇ = known(u) + NN(u)
# p are NN paramters, p_true are known model coefficients, i.e. α, δ
function ude_dynamics!(dx, x, p, t, A, B)
    # nn = NN(forcing(t), p, _st)[1]  # NN prediction
    nn = NN(x, p, _st)[1]  # NN prediction
    dx .= A*x + B*forcing(t) + nn
    # dx .= A*x + nn
end


# Set up closure to handle extra arguments
nn_dynamics!(dx, x, p, t) = ude_dynamics!(dx, x, p, t, A, B)


# set up ODE problem
x₀ = X[1,1:r-n_control]
tspan = (ts_train_short[1], ts_train_short[end])
prob_nn = ODEProblem(nn_dynamics!, x₀, tspan, p)



# test solve
sol = solve(prob_nn, saveat=ts_train_short);
X̂ = Array(sol)

Z_orig = H[1, idx_train_short]
Ẑ = (U_small[1,:]' * diagm(σ_small) * X̂)'

fig = Figure()
ax = CairoMakie.Axis(fig[1,1])
lines!(ax, Z_orig, label="Orig")
lines!(ax, Ẑ, label="HAVOK")
fig



# set up function for generating solution with new parameters
function predict(θ, x0_new = X[1, 1:r-n_control], T = ts_train_short)
    _prob = remake(prob_nn, u0=x0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

predict(p)


function embedd_to_timeseries(Xpred, U_small, σ_small)
    U_small[1, :]' * diagm(σ_small) * X̂
end


# compute loss by reconstructing original time series
Z_orig = H[1, idx_train_short]

function loss(θ)
    X̂ = predict(θ)
    Z_pred = (U_small[end,:]' * diagm(σ_small) * X̂)'
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

    if length(losses) % 1000 == 0
        println("Saving current param vals...")
        save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => p, "st" => _st))
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
# res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=1)


res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=5000)
save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => res1.u, "st" => _st))


# Train second round using LBFGS to get better minimum using derivative information
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=1000)

save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => res2.u, "st" => _st))

# Rename the best candidate
p_trained = res2.u




# test solve
loss(p_trained)
X̂ = predict(p_trained)

Z_orig = H[1, idx_train_short]
Ẑ = (U_small[1,:]' * diagm(σ_small) * X̂)'

fig = Figure()
ax = CairoMakie.Axis(fig[1,1])
lines!(ax, Z_orig, label="Orig")
lines!(ax, Ẑ, label="HAVOK")
fig





