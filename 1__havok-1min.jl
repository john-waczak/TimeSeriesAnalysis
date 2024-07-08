using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations
using ProgressMeter

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")




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

outpath = "./data/output/1__havok-1min"
if !ispath(outpath)
    mkpath(outpath)
end


c4_path = joinpath(datapath, "central-hub-4")
c7_path = joinpath(datapath, "central-hub-7")
c10_path = joinpath(datapath, "central-hub-10")


# load in CSVs and associated summaries
readdir(c4_path)

df4 = CSV.read(joinpath(c4_path, "df.csv"), DataFrame);
df4_summary = CSV.read(joinpath(c4_path, "df_summary.csv"), DataFrame);

df7 = CSV.read(joinpath(c7_path, "df.csv"), DataFrame);
df7_summary = CSV.read(joinpath(c7_path, "df_summary.csv"), DataFrame);

df10 = CSV.read(joinpath(c10_path, "df.csv"), DataFrame);
df10_summary = CSV.read(joinpath(c10_path, "df_summary.csv"), DataFrame);


df4_summary

extrema(df4_summary.nrow)   # (1, 720)
extrema(df7_summary.nrow)   # (1, 10606)   # <---- winner!
extrema(df10_summary.nrow)  # (1, 1439)

idx_winner = argmax(df7_summary.nrow)
df = df7[df7.group .== idx_winner, :]

df.pressure .= 10 .* df.pressure

cols_to_use = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0, :temperature, :pressure, :humidity]
col_names = ["PM 0.1", "PM 0.3", "PM 0.5", "PM 1.0", "PM 2.5", "PM 5.0", "PM 10.0", "Temperature", "Pressure", "Relative Humidity"]
col_units = ["μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "°C", "mbar", "%"]



# let's focus on PM 2.5
ts = df.dt .- df.dt[1]
dt = ts[2]-ts[1]
Zs = Array(df[:, :pm2_5])


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
lines!(ax, ts ./ (24*60), Zs, color=mints_colors[3])
xlims!(ax, ts[1]/(24*60), ts[end]/(24*60))
fig

# # smooth with trialing 10 min average
# trailing(Z, n) = [i < n ? mean(Z[1:i]) : mean(Z[i-n+1:i]) for i in 1:length(Z)]
# Zs_smooth = trailing(Zs, 10)


# fig = Figure();
# ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
# lines!(ax, ts ./ (24*60), Zs, color=mints_colors[3])
# xlims!(ax, ts[1]/(24*60), ts[end]/(24*60))
# fig


function eval_havok(Zs, ts, n_embedding, r_model, n_control; method=:backward)
    r = r_model + n_control

    # cutoff time for training vs testing partition
    Zs_x = Zs[n_embedding:end]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

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

    params = (A, B)
    x₀ = Xtrain[1,1:r_model]


    # define function and integrate to get model predictions
    function f!(dx, x, p, t)
        A,B = p
        dx .= A*x + B*forcing(t)
    end


    # define ODE problem and solve
    idx_int = 1:(7*24*60)
    ts_int = ts_x[idx_int]

    prob = ODEProblem(f!, x₀, (ts_int[1], ts_int[end]), params);
    sol = solve(prob, saveat=ts_int);
    X̂ = Array(sol)'

    Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]

    return Zs_x[idx_int], Ẑs_x, ts_int
end



# test it out
n_embedding = 100
r_model = 5
n_control = 20
method = :backward

Zs_x, Ẑs_x, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model, n_control; method=method)
# Zs_x, Ẑs_x, ts_x = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)




# fig = Figure();
# ax = CairoMakie.Axis(fig[1,1], xlabel="time", ylabel="PM")
# l_orig = lines!(ax, ts_x ./(24*60), Zs_x)
# l_havok = lines!(ax, ts_x ./(24*60), Ẑs_x)

# fig




# compute mse for 15 min, 60 min, and 3 hr
method=:backward
n_embedding = 100
rs_model = 3:50
ns_control = 1:40

# preallocate output arrays
RMSE_1 = zeros(length(rs_model), length(ns_control));
RMSE_12 = zeros(length(rs_model), length(ns_control));
RMSE_24 = zeros(length(rs_model), length(ns_control));
RMSE_48 = zeros(length(rs_model), length(ns_control));
RMSE_72 = zeros(length(rs_model), length(ns_control));

pm = Progress(length(rs_model))
for i ∈ 1:length(rs_model)
    r_model = rs_model[i]
    for j ∈ 1:length(ns_control)
        n_control = ns_control[j]


        Zs_x, Ẑs_x, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model, n_control; method=method)
        # Zs_x, Ẑs_x, ts_x = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)


        rmse_1 = sqrt(mean(abs2, Ẑs_x[1:60] .- Zs_x[1:60]))
        rmse_12 = sqrt(mean(abs2, Ẑs_x[1:12*60] .- Zs_x[1:12*60]))
        rmse_24 = sqrt(mean(abs2, Ẑs_x[1:24*60] .- Zs_x[1:24*60]))
        rmse_48 = sqrt(mean(abs2, Ẑs_x[1:48*60] .- Zs_x[1:48*60]))
        rmse_72 = sqrt(mean(abs2, Ẑs_x[1:72*60] .- Zs_x[1:72*60]))

        RMSE_1[i,j] = rmse_1
        RMSE_12[i,j] = rmse_12
        RMSE_24[i,j] = rmse_24
        RMSE_48[i,j] = rmse_48
        RMSE_72[i,j] = rmse_72
    end

    next!(pm; showvalues=[(:r_model, r_model),])
end


fig = Figure();
idx_min = argmin(RMSE_1)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_1, colormap=:thermal)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="1 hour RMSE")
fig

save(joinpath(fig_savepath, "2__havok-rmse-1-hour.png"), fig)
save(joinpath(fig_savepath, "2__havok-rmse-1-hour.pdf"), fig)


fig = Figure();
idx_min = argmin(RMSE_12)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_12, colormap=:thermal)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="12 hour RMSE")
fig

save(joinpath(fig_savepath, "3__havok-rmse-12-hour.png"), fig)
save(joinpath(fig_savepath, "3__havok-rmse-12-hour.pdf"), fig)



fig = Figure();
idx_min = argmin(RMSE_24)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_24, colormap=:thermal)
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="24 hour RMSE")
fig

save(joinpath(fig_savepath, "4__havok-rmse-24-hour.png"), fig)
save(joinpath(fig_savepath, "4__havok-rmse-24-hour.pdf"), fig)



cmax = quantile(RMSE_48[1:length(RMSE_48)], 0.95)

fig = Figure();
idx_min = argmin(RMSE_48)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_48, colormap=:thermal, colorrange=(0,cmax))
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="48 hour RMSE")
fig

save(joinpath(fig_savepath, "5__havok-rmse-48-hour.png"), fig)
save(joinpath(fig_savepath, "5__havok-rmse-48-hour.pdf"), fig)




cmax = quantile(RMSE_72[1:length(RMSE_72)], 0.95)

fig = Figure();
idx_min = argmin(RMSE_72)
ax = CairoMakie.Axis(fig[1,1], xlabel="r", ylabel="n control")
hm = heatmap!(ax, rs_model, ns_control, RMSE_72, colormap=:thermal, colorrange=(0,cmax))
sc = scatter!(ax, [rs_model[idx_min[1]]], [ns_control[idx_min[2]]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="72 hour RMSE")
fig


save(joinpath(fig_savepath, "6__havok-rmse-72-hour.png"), fig)
save(joinpath(fig_savepath, "6__havok-rmse-72-hour.pdf"), fig)




# -------------------------------------------------------------
# Collect results
# -------------------------------------------------------------


r_model_best = rs_model[idx_min[1]]
n_control_best = ns_control[idx_min[2]]

r_model_mid = 5
n_control_mid = ns_control[argmin(RMSE_72[3,:])]

r_model_small = 3
n_control_small = ns_control[argmin(RMSE_72[1,:])]


# Create plot comparing the 3 best models:
Zs_x, Ẑs_x_best, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_best, n_control_best; method=method)
Zs_x, Ẑs_x_mid, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_mid, n_control_mid; method=method)
Zs_x, Ẑs_x_small, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_small, n_control_small; method=method)


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
l_orig = lines!(ax, ts_x[1:3*24*60]./(24*60), Zs_x[1:3*24*60], linewidth=3)
l_best = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_best[1:3*24*60], linewidth=2)
l_mid = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_mid[1:3*24*60], linewidth=2)
l_small = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_small[1:3*24*60], linewidth=2)
xlims!(ax, ts_x[1]./(24*60), 3)
fig[1,2] = Legend(fig, [l_orig, l_best, l_mid, l_small], ["Original", "HAVOK ($(r_model_best), $(n_control_best))","HAVOK ($(r_model_mid), $(n_control_mid))", "HAVOK ($(r_model_small), $(n_control_small))" ], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17, height=-5)
fig

save(joinpath(fig_savepath, "7__havok-reconstruction.png"), fig)
save(joinpath(fig_savepath, "7__havok-reconstruction.pdf"), fig)



fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
l_orig = lines!(ax, ts_x./(24*60), Zs_x, linewidth=3)
l_best = lines!(ax, ts_x./(24*60), Ẑs_x_best, linewidth=2)
l_mid = lines!(ax, ts_x./(24*60), Ẑs_x_mid, linewidth=2)
l_small = lines!(ax, ts_x./(24*60), Ẑs_x_small, linewidth=2)
xlims!(ax, ts_x[1]./(24*60), ts_x[end] ./(24*60))
fig[1,2] = Legend(fig, [l_orig, l_best, l_mid, l_small], ["Original", "HAVOK ($(r_model_best), $(n_control_best))","HAVOK ($(r_model_mid), $(n_control_mid))", "HAVOK ($(r_model_small), $(n_control_small))" ], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17, height=-5)
fig

save(joinpath(fig_savepath, "8__havok-reconstruction-long.png"), fig)
save(joinpath(fig_savepath, "8__havok-reconstruction-ling.pdf"), fig)

