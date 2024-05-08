# HAVOK Model for Single Central Node

using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations
using ProgressMeter, JSON

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")



# load in data for central node
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



# visualize base time-series
fig = Figure();
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="time",
    ylabel="PM 2.5 (μg/m³)",
    xticks=(x_tick_pos, x_tick_strings),
    xticklabelrotation=π/3
);
ll = lines!(ax, ts, Zs)
xlims!(ax, ts[1], ts[end])
fig

save(joinpath(fig_savepath, "time-series-original.png"), fig)
save(joinpath(fig_savepath, "time-series-original.pdf"), fig)


function eval_havok(Zs, ts, n_embedding, r_model, n_control; method=:backward)
    r = r_model + n_control

    # cutoff time for training vs testing partition
    Zs_x = Zs[n_embedding:end]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

    t_train_end = dt*(9*24*60*60)
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

    A = Ξ[:, 1:r-n_control]   # State matrix A
    B = Ξ[:, r-n_control+1:end]      # Control matrix B


    # define interpolation function for forcing coordinate(s)
    itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:r];
    forcing(t) = [itp(t) for itp ∈ itps]

    params = (A, B)
    x₀ = Xtrain[1,1:r_model]


    # define function and integrate to get model predictions
    function f!(dx, x, p, t)
        A,B = p
        dx .= A*x + B*forcing(t)
    end


    # define ODE problem and solve
    idx_int = 1:(3*60*60)
    ts_int = ts_x[idx_int]

    prob = ODEProblem(f!, x₀, (ts_int[1], ts_int[end]), params);
    sol = solve(prob, saveat=ts_int);
    X̂ = Array(sol)'


    Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]


    # compute mse for 15 min, 60 min, and 3 hr
    rmse_15 = sqrt(mean(abs2, Ẑs_x[1:15*60] .- Zs_x[1:15*60]))
    rmse_60 = sqrt(mean(abs2, Ẑs_x[1:60*60] .- Zs_x[1:60*60]))
    rmse_180 = sqrt(mean(abs2, Ẑs_x[1:180*60] .- Zs_x[1:180*60]))

    return rmse_15, rmse_60, rmse_180
end


rmse_15, rmse_60, rmse_180 = eval_havok(Zs, ts, 100, 17, 10)


# explore hyperparameter valid ranges

rs_model = 5:45
ns_control = 1:15

RMSE_15 = zeros(length(rs_model), length(ns_control))
RMSE_60 = zeros(length(rs_model), length(ns_control))
RMSE_180 = zeros(length(rs_model), length(ns_control))

pm = Progress(length(rs_model))
for i ∈ 1:length(rs_model)
    r_model = rs_model[i]
    for j ∈ 1:length(ns_control)
        n_control = ns_control[j]

        rmse_15, rmse_60, rmse_180 = eval_havok(Zs, ts, 100, r_model, n_control)

        RMSE_15[i,j] = rmse_15
        RMSE_60[i,j] = rmse_60
        RMSE_180[i,j] = rmse_180
    end

    next!(pm; showvalues=[(:r_model, r_model),])
end


cmap = cgrad(:thermal, 10; categorical=true)

fig = Figure();
idx_min = argmin(RMSE_15)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_15, colorrange=(0, 50),  colormap=cmap)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="15 min RMSE")
fig
save(joinpath(fig_savepath, "havok-rmse-15min.png"), fig)
save(joinpath(fig_savepath, "havok-rmse-15min.pdf"), fig)

fig = Figure();
idx_min = argmin(RMSE_60)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_60, colorrange=(0, 50),  colormap=cmap)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="60 min RMSE")
fig
save(joinpath(fig_savepath, "havok-rmse-60min.png"), fig)
save(joinpath(fig_savepath, "havok-rmse-60min.pdf"), fig)



fig = Figure();
idx_min = argmin(RMSE_180)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_180, colorrange=(0, 50),  colormap=cmap)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="3 hour RMSE")
fig
save(joinpath(fig_savepath, "havok-rmse-180min.png"), fig)
save(joinpath(fig_savepath, "havok-rmse-180min.pdf"), fig)


fig


# consider 3 cases, the best model, the best model with low n_control, and the best model with n_control = 1



idx_min

r_model_best = rs_model[idx_min[1]]
n_control_best = ns_control[idx_min[2]]

r_model_mid = 19
n_control_mid = 5

r_model_one = argmin(RMSE_180[:,1])
n_control_one = 1


# save these results to a dictionary
# save results to JSON file
out_dict = Dict(
    :r_model   => collect(rs_model),
    :n_control => collect(ns_control),
    :rmse_15   => RMSE_15,
    :rmse_60   => RMSE_60,
    :rmse_180  => RMSE_180,
    :best => [r_model_best, n_control_best],
    :mid => [r_model_mid, n_control_mid],
    :one => [r_model_one, n_control_one]
)


open(joinpath(fig_savepath, "havok_params.json"), "w") do f
    JSON.print(f, out_dict)
end



# -----------------------------------------------------------------------------------
# Explore Best Solutions
# -----------------------------------------------------------------------------------
param_set = :mid
method = :backward
n_embedding = 100
r_model, n_control = out_dict[param_set]
r = r_model + n_control

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

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B


# define interpolation function for forcing coordinate(s)
itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:r];
forcing(t) = [itp(t) for itp ∈ itps]

params = (A, B)
x₀ = Xtrain[1,1:r_model]


# define function and integrate to get model predictions
function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*forcing(t)
end


# define ODE problem and solve
idx_int = 1:(3*60*60)
ts_int = ts_x[idx_int]

prob = ODEProblem(f!, x₀, (ts_int[1], ts_int[end]), params);
sol = solve(prob, saveat=ts_int);
X̂ = Array(sol)'

Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]


fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="t (min)", ylabel="PM 2.5 (μg/m³)");
l_z = lines!(ax, ts_int ./ (60), Zs_x[idx_int], linewidth=3)
l_ẑ = lines!(ax, ts_int ./ (60), Ẑs_x[idx_int], linestyle=:dash, linewidth=2)
fig[1,1] = Legend(fig, [l_z, l_ẑ], ["Original", "HAVOK Reconstruction"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
xlims!(ax, ts_int[1]/60, ts_int[end]/60)
save(joinpath(fig_savepath, "havok-reconstruction-full.png"), fig)
save(joinpath(fig_savepath, "havok-reconstruction-full.pdf"), fig)
fig


xlims!(ax, ts_int[1]/60, 30)
save(joinpath(fig_savepath, "havok-reconstruction-30min.png"), fig)
save(joinpath(fig_savepath, "havok-reconstruction-30min.pdf"), fig)

fig


