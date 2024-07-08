using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations
using ProgressMeter


# SciML Tools
using DifferentialEquations, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays, Lux, Zygote, StableRNGs, JSON

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")

rng = StableRNG(42)


# -----------------------------------------------------------------------
# 1. Load data and set up figure paths
# -----------------------------------------------------------------------


# load in data for central node
datapath = "./data/processed"
@assert ispath(datapath)
figpath = "./figures"
@assert ispath(figpath)

fig_savepath = joinpath(figpath, "1__havok-1min")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end

c4_path = joinpath(datapath, "multi-sensor", "central-hub-4")
c7_path = joinpath(datapath, "multi-sensor", "central-hub-7")
c10_path = joinpath(datapath, "multi-sensor", "central-hub-10")


# load in CSVs and associated summaries
readdir(c4_path)

df4 = CSV.read(joinpath(c4_path, "df.csv"), DataFrame);
df4_summary = CSV.read(joinpath(c4_path, "df_summary.csv"), DataFrame);

df7 = CSV.read(joinpath(c7_path, "df.csv"), DataFrame);
df7_summary = CSV.read(joinpath(c7_path, "df_summary.csv"), DataFrame);

df10 = CSV.read(joinpath(c10_path, "df.csv"), DataFrame);
df10_summary = CSV.read(joinpath(c10_path, "df_summary.csv"), DataFrame);


extrema(df4_summary.nrow)   # (1, 720)
extrema(df7_summary.nrow)   # (1, 10606)   # <---- winner!
extrema(df10_summary.nrow)  # (1, 1439)

idx_winner = argmax(df7_summary.nrow)
df = df7[df7.group .== idx_winner, :]

df.pressure .= 10 .* df.pressure


cols_to_use = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0, :temperature, :pressure, :humidity]
col_names = ["PM 0.1", "PM 0.3", "PM 0.5", "PM 1.0", "PM 2.5", "PM 5.0", "PM 10.0", "Temperature", "Pressure", "Relative Humidity"]
col_units = ["μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "°C", "mbar", "%"]



ts = df.dt .- df.dt[1]
dt = ts[2]-ts[1]
Zs_orig = Array(df[:, :pm2_5])


# smooth with trialing 10 min average
trailing(Z, n) = [i < n ? mean(Z[1:i]) : mean(Z[i-n+1:i]) for i in 1:length(Z)]
Zs = trailing(Zs_orig, 10)


method = :backward
n_embedding = 100
r_model = 5
n_control = 1
r = r_model + n_control

# cutoff time for training vs testing partition
Zs_x = Zs[n_embedding:end]
ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

# set up training indices
t_train_end = dt*(7*24*60)
idx_train = 1:findfirst(ts_x .≥ t_train_end)
idx_test = (idx_train[end] + 1 : length(ts_x))

# construct Hankel Matrix
H = TimeDelayEmbedding(Zs; n_embedding=n_embedding, method=method);

# Decompose via SVD
U, σ, V = svd(H)

# truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


X = Vr
dX = zeros(size(Vr,1), r_model)

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


# define interpolation function for forcing coordinate(s)
itps_full = [DataInterpolations.LinearInterpolation(V[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:n_embedding];
forcing_full(t) = [itp(t) for itp ∈ itps_full]

forcing_full(ts_x[1])


# Compute scale factor for non-linear model
scale_factor = 10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))) - 1)

rbf(x) = exp.(-(x.^2))
out_func(x) = scale_factor * tanh(x)
act_func = tanh

n_hidden = 30
NN = Lux.Chain(
    Lux.Dense(n_embedding - r_model, n_hidden, act_func),
    Lux.Dense(n_hidden, n_hidden, act_func),
    Lux.Dense(n_hidden, 1, tanh),
)


# Get initial parameters and state of NN
p, st = Lux.setup(rng, NN)
const _st = st


# test the NN
NN(forcing_full(ts_x[1]), p, _st)[1]



function f_havok!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*forcing(t)
end

# Define UDE as dynamical system
function ude_dynamics!(dx, x, p, t, A, B)
    # dx .= A*x + NN(x, p, _st)[1] + B*forcing(t)
    dx .= A*x + B*NN(forcing_full(t), p, _st)[1]
end

# Set up closure to handle extra arguments
nn_dynamics!(dx, x, p, t) = ude_dynamics!(dx, x, p, t, A, B)

function get_Zs_from_X(X)
    return X' * diagm(σr[1:r_model]) * Ur[1, 1:r_model]
end




# set up ODE problem
idx_train_short = 1:(5*60)
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
ax = CairoMakie.Axis(fig[2,1], xlabel="time", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short, Zs_x[idx_train_short])
l_hk = lines!(ax, ts_train_short, Ẑs_hk)
l_nn = lines!(ax, ts_train_short, Ẑs)
fig[1,1] = Legend(fig, [l_orig, l_hk, l_nn], ["Original", "HAVOK", "HAVOK + NN"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig





# set up function for generating solution with new parameters
function predict(θ, x0_new = X[1, 1:r_model], T = ts_train_short)
    _prob = remake(prob_nn, u0=x0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

function loss(θ)
    X̂ = predict(θ, X[1,1:r_model], ts_x[idx_train_short])
    Ẑs = get_Zs_from_X(X̂)
    sqrt(mean((Ẑs .- Zs_x[idx_train_short]) .^2 ))
end

predict(p)
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



# set up array to track loss function with callback
losses = Float64[]

function callback(p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end

    return false  # true signals early stopping
end

adtype = Optimization.AutoZygote()

# First training round using ADAM
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=1000)
p = res.u



Ẑs_trained = get_Zs_from_X(X̂)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="time", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short, Zs_x[idx_train_short])
l_hk = lines!(ax, ts_train_short, Ẑs_hk)
l_nn = lines!(ax, ts_train_short, Ẑs_trained)
fig[1,1] = Legend(fig, [l_orig, l_hk, l_nn], ["Original", "HAVOK", "HAVOK + NN"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

fig





