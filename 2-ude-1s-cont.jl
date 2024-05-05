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

fig_savepath = joinpath(figpath, "UDE-cont")
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


params_path = joinpath("./figures", "central-nodes", "central-hub-4", "HAVOK-cont", "havok_params.json")
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

# scale_factor = 10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))))
# scale_factor = 5*10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))) - 1)
scale_factor = 10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))) - 1)
# scale_factor = 5*10^(round(log10(maximum(abs.(A * X[:, 1:r_model]')))) - 2)

rbf(x) = exp.(-(x.^2))
out_func(x) = scale_factor * tanh(x)
act_func = tanh




const n_hidden = 30
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
idx_train_short = 1:(5*60)
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


# set up function for generating solution with new parameters
function predict(θ, x0_new = X[1, 1:r_model], T = ts_train_short)
    _prob = remake(prob_nn, u0=x0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Tsit5(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

predict(p)


idx_ude_train = 1:(5*60)  # 5 minutes

function loss(θ)
    X̂ = predict(θ, X[1,1:r_model], ts_x[idx_ude_train])
    Ẑs = get_Zs_from_X(X̂)
    sqrt(mean((Ẑs .- Zs_x[idx_ude_train]) .^2 ))
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



# visualize the fit
Ẑs_fit = get_Zs_from_X(predict(p))

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t (min)", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short ./ 60, Zs_x[idx_train_short], linewidth=3)
l_havok = lines!(ax, ts_train_short ./ 60, Ẑs_hk, linestyle=:dot, linewidth=2)
l_ude = lines!(ax, ts_train_short ./ 60, Ẑs_fit, linestyle=:dash, linewidth=2)
fig[1,1] = Legend(fig, [l_orig, l_havok, l_ude], ["Original", "HAVOK", "HAVOK + NN"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

save(joinpath(fig_savepath, "1__UDE-fit-NN-pre-training.png"), fig)
save(joinpath(fig_savepath, "1__UDE-fit-NN-pre-training.pdf"), fig)

fig





adtype = Optimization.AutoZygote()
p_best = p

# First training round using ADAM
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=10_000)
p = res.u


# Second training round using LBFGS
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))
res = Optimization.solve(optprob, LBFGS(), callback=callback, maxiters=1_000)
p = res.u


save(joinpath(fig_savepath, "model_params.jld2"), Dict("ps_trained" => p, "st" => _st, "A" => A, "B" => B, "scale_factor" => scale_factor, "Ur" => Ur, "σr" => σr, "Vr" => Vr))



# plot the losses:
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="iteration", ylabel="RMSE", title="UDE Training Loss", yscale=log10, xscale=log10)
lines!(ax, 1:length(losses), losses)
fig


save(joinpath(fig_savepath, "2__nn-training-loss.png"), fig)
save(joinpath(fig_savepath, "2__nn-training-loss.pdf"), fig)


# visualize the fit
Ẑs_fit = get_Zs_from_X(predict(p))
fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t (min)", ylabel="PM 2.5");
l_orig = lines!(ax, ts_train_short ./ 60, Zs_x[idx_train_short], linewidth=3)
l_havok = lines!(ax, ts_train_short ./ 60, Ẑs_hk, linestyle=:dot, linewidth=2)
l_ude = lines!(ax, ts_train_short ./ 60, Ẑs_fit, linestyle=:dash, linewidth=2)
fig[1,1] = Legend(fig, [l_orig, l_havok, l_ude], ["Original", "HAVOK", "UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)


fig


# now we should evaluate on a longer time-frame, say 15-min
idx_15 = 1:(15*60)
Ẑs_fit = get_Zs_from_X(predict(p_trained, X[1,1:r_model], ts_x[idx_15]))
fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t (min)", ylabel="PM 2.5");
l_orig = lines!(ax, ts_x[idx_15]./ 60, Zs_x[idx_15], linewidth=3)
# l_havok = lines!(ax, ts_x[idx_15]./ 60, Ẑs_hk, linestyle=:dot, linewidth=2)
l_ude = lines!(ax, ts_x[idx_15] ./ 60, Ẑs_fit, linestyle=:dash, linewidth=2)
#fig[1,1] = Legend(fig, [l_orig, l_havok, l_ude], ["Original", "HAVOK", "UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig[1,1] = Legend(fig, [l_orig, l_ude], ["Original", "UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

save(joinpath(fig_savepath, "3__UDE-fit-NN-trained.png"), fig)
save(joinpath(fig_savepath, "3__UDE-fit-NN-trained.pdf"), fig)

fig



# visualize forcing function time series
fig = Figure();
idx_plot = 1:(60*60)
ax = CairoMakie.Axis(fig[1,1], xlabel="t (min)", ylabel="forcing")
lines!(ax, ts_x[idx_plot] ./ 60, vcat(forcing.(ts_x[idx_plot])...))
save(joinpath(fig_savepath, "4__forcing-timeseries.png"), fig)
save(joinpath(fig_savepath, "4__forcing-timeseries.pdf"), fig)
fig

xlims!(ax, ts_x[1] / 60, 15)
save(joinpath(fig_savepath, "5__forcing-timeseries-15min.png"), fig)
save(joinpath(fig_savepath, "5__forcing-timeseries-15min.pdf"), fig)
fig





# --------------------------------------------------------------------------------------------
#   Do Sparse Regression on Learned NN
# --------------------------------------------------------------------------------------------


# Set up sparse regression problem
# the Dataset may be too big for this to be efficient...
@variables x[1:r_model]
# b = polynomial_basis(x, 2);
b_2 = []
b_3 = []
for i ∈ 1:r_model
    for j ∈ 1:r_model
        if j ≥ i
            push!(b_2, x[i]*x[j])
        end
    end
end

for i ∈ 1:r_model
    for j ∈ 1:r_model
        for k ∈ 1:r_model
            if (j ≥ i) && (k ≥ j)
                push!(b_3, x[i]*x[j]*x[k])
            end
        end
    end
end

# basis = Basis(vcat(b_2, b_3), x);
basis = Basis(b_2, x);


# create extra data by integrating trained ODE at a finer timestep
ts_reg = ts_train_short[1]:0.5:ts_train_short[end]

X̂ = predict(p, X[1, 1:r_model], ts_reg)
Ŷ = NN(X̂, p, _st)[1]

size(X̂)
size(Ŷ)

nn_prob = DirectDataDrivenProblem(X̂, Ŷ, name= :UDE_REG)


λ = exp10.(-1:0.025:3)
# opt = STLSQ(0.125)   # Nonzero percent: 69.2
opt = STLSQ(λ)  # Nonzero percent: 88.1

# λ          | %
# -------------------
# -1:0.05:1  | 19.436
# -1:0.025:3 | 19.141
# -1:0.02:3  | 19.196
# -1:0.1:2   | 20.47
# -1:0.1:1   | 20.47
# -1:0.1:0   | 26.4
# 0.1        | 63.2
# 0.01       | 94.4

options = DataDrivenCommonOptions(
    maxiters = 10_000,
    normalize = DataNormalization(),  # ZScoreTransform caused failure
    selector = aicc, # 20.47
    #selector = bic,  # 20.997
    #selector = aic,   # 21.38
    #selector = rss,   # 32.779
    digits = 10,
    data_processing = DataProcessing(
        split = 0.9,
        # batchsize=32,  # 22.35
        # batchsize=28,  # 18.11
        batchsize=26,  # 17.174
        shuffle=true,
        rng=StableRNG(42)
    )
)


nn_res = solve(nn_prob, basis, opt, options=options)

# evaluate the result
nn_eqs = get_basis(nn_res)
p_nn = get_parameter_values(nn_eqs)
(length(p_nn)/(length(basis) * r_model))*100



# Now that we have equation candidates, learn new parameters to make this work:
function recovered_dynamics!(dx, x, p, t, A, B)
    dx .= A*x + B*forcing(t) + nn_eqs(u, p)
end

# define closure
rec_dynamics!(dx, x, p, t) = recovered_dynamics!(dx, x, p, t, A, B)

function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
    sum(abs2, Ŷ .- Y)
end



# THIS SEEMS REALLY SLOW

# losses = Float64[]
# optf = Optimization.OptimizationFunction((x,p) -> parameter_loss(x), adtype)
# optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
# parameter_res = Optimization.solve(optprob, Optim.LBFGS(), callback=callback, maxiters=1000)






# tspan = (ts_x[1], ts_x[idx_15[end]])
# prob_rec = ODEProblem(rec_dynamics!, x₀, tspan, p_nn)
# sol = solve(prob_nn, saveat=ts_x[idx_15]);
# X̂ = Array(sol)
# Ẑs_rec = get_Zs_from_X(X̂)



# # okay, so these values aren't great... let's try and optimize the equations we recovered
# fig = Figure();
# ax = CairoMakie.Axis(fig[2,1], xlabel="t (min)", ylabel="PM 2.5");
# l_orig = lines!(ax, ts_train_short ./ 60, Zs_x[idx_train_short], linewidth=3)
# l_havok = lines!(ax, ts_train_short ./ 60, Ẑs_hk, linestyle=:dot, linewidth=2)
# l_rec = lines!(ax, ts_train_short ./ 60, Ẑs_rec[idx_train_short], linestyle=:dashdot, linewidth=2)
# fig[1,1] = Legend(fig, [l_orig, l_havok, l_rec], ["Original", "HAVOK", "Recovered UDE"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
# fig



