# File Handling
using CSV, DataFrames, Dates, JSON, JLD2, FileIO

# SciML Tools
using DifferentialEquations, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, DataInterpolations
using StatsBase, Distributions, KernelDensity

# External Libraries
using ComponentArrays, Lux, Zygote, StableRNGs, JSON

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")

# Seed stable random number generator
rng = StableRNG(42)



datapath = "./data/processed"
time_type = "1-sec"
datapath = joinpath(datapath, "central-nodes", "central-hub-4", "df-"*time_type*".csv")
@assert ispath(datapath)


figpath = joinpath("./figures", "central-nodes", "central-hub-4")
if !ispath(figpath)
    mkpath(figpath)
end
@assert ispath(figpath)

fig_savepath = joinpath(figpath, "HAVOK-cont")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


# load in data as single time-series
t1 = Date(2023, 5, 29);
t2 = Date(2023, 6, 8);
col_to_use = :pm2_5
Zs, ts = load_data(datapath, t1, t2, col_to_use)
dt = ts[2] - ts[1]
@assert dt == 1.0



# set up ticks for plots
x_tick= (round(t1, Day)-Day(1)):Day(1):(round(t2, Day) + Day(1))
x_tick_pos = [Second(d - t1).value for d ∈ x_tick]
x_tick_strings = [Dates.format.(d, "dd/mm/yy") for d ∈ x_tick]


params_path = joinpath(fig_savepath, "havok_params.json")
params_dict = JSON.parsefile(params_path)


param_set = "mid"
method = :backward
const n_embedding = 100
const r_model = params_dict[param_set][1]
# const n_control = params_dict[param_set][2]
const n_control = 1
const r = r_model + n_control



# cutoff time for training vs testing partition
Zs_x = Zs[n_embedding:end]
ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

t_train_end = dt*(9*24*60*60)
idx_train = 1:findfirst(ts_x .≥ t_train_end)
idx_test = (idx_train[end] + 1 : length(ts_x))

# construct Hankel Matrix
H = TimeDelayEmbedding(Zs; n_embedding=n_embedding, method=method);

# Decompose via SVD
U, σ, V = svd(H);

# truncate the matrices
Vr = @view V[:,1:r];
Ur = @view U[:,1:r];
σr = @view σ[1:r];


X = Vr;
dX = zeros(size(Vr,1), r-n_control);

for j ∈ axes(dX, 2)
    itp = CubicSpline(X[:,j], ts_x)
    for i ∈ axes(dX, 1)
        dX[i,j] = DataInterpolations.derivative(itp, ts_x[i])
    end
end


# partition into training and testing sets
Xtrain = X[idx_train, :];
Xtest = X[idx_test, :];

dXtrain = dX[idx_train, :];
dXtest = dX[idx_test, :];

# Compute model matrix via least squares
Ξ = (Xtrain\dXtrain)'  # now Ξx = dx for a single column vector view

A = Ξ[:, 1:r_model]   # State matrix A
B = Ξ[:, r_model+1:end]      # Control matrix B


# define interpolation function for forcing coordinate(s)
itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:r];
forcing(t) = [itp(t) for itp ∈ itps]



# Let's try to train a UDE to learn the missing physics:
rbf(x) = exp.(-(x.^2))
out_func(x) = scale_factor * tanh(x)
act_func = tanh

# estimate output scale factor for nonlinear-terms so that
# so that correction is weaker than A
scale_factor = 10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))) - 1)



const n_hidden = 20
const NN = Lux.Chain(
    Lux.Dense(r_model, n_hidden, act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, r_model, out_func),
)


# Get initial parameters and state of NN
p, st = Lux.setup(rng, NN)
const _st = st

# test the NN
NN(X[1, 1:r_model], p, _st)[1]


function f_havok!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*forcing(t)
end

# Define UDE as dynamical system
function ude_dynamics!(dx, x, p, t, A, B)
    # nn = NN(forcing(t), p, _st)[1]  # NN prediction
    dx .= A*x + B*forcing(t) + NN(x, p, _st)[1]
end


# Set up closure to handle extra arguments
nn_dynamics!(dx, x, p, t) = ude_dynamics!(dx, x, p, t, A, B)

function get_Zs_from_X(X)
    return X' * diagm(σr[1:r_model]) * Ur[1, 1:r_model]
end

# set up ODE problem
idx_train_short = 1:(15*60)
# idx_train_short = 1:90
ts_train_short = ts_x[idx_train_short]
x₀ = X[1,1:r_model]
tspan = (ts_train_short[1], ts_train_short[end])
prob_nn = ODEProblem(nn_dynamics!, x₀, tspan, p)
prob_hk = ODEProblem(f_havok!, x₀, tspan, (A,B))


# Perform initial solve and compare against base HAVOK model
sol = solve(prob_nn, saveat=ts_train_short);
X̂ = Array(sol)

sol = solve(prob_hk, saveat=ts_train_short);
X̂hk = Array(sol)

Ẑs = get_Zs_from_X(X̂)
Ẑs_hk = get_Zs_from_X(X̂hk)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short, Zs_x[idx_train_short], linewidth=3)
l_havok = lines!(ax, ts_train_short, Ẑs_hk, linestyle=:dot, linewidth=2)
l_ude = lines!(ax, ts_train_short, Ẑs, linestyle=:dash, linewidth=2)
fig[1,1] = Legend(fig, [l_orig, l_havok, l_ude], ["Original", "HAVOK", "UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig




# set up function for generating solution with new parameters
function predict(θ, x0_new = X[1, 1:r_model], T = ts_train_short)
    _prob = remake(prob_nn, u0=x0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

predict(p)

function loss(θ)
    X̂ = predict(θ)
    Ẑs = get_Zs_from_X(X̂)
    mean(abs2, Ẑs .- Zs_x[idx_train_short])
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
# res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=1)
# res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=1000)
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=5000)
save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => res1.u, "st" => _st))

optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=1000)

save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => res2.u, "st" => _st))

# Rename the best candidate
p_trained = res2.u


# visualize the fit
Ẑs_fit = get_Zs_from_X(predict(p_trained))
fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short, Zs_x[idx_train_short], linewidth=3)
l_havok = lines!(ax, ts_train_short, Ẑs_hk, linestyle=:dot, linewidth=2)
l_ude = lines!(ax, ts_train_short, Ẑs_fit, linestyle=:dash, linewidth=2)
fig[1,1] = Legend(fig, [l_orig, l_havok, l_ude], ["Original", "HAVOK", "UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig




