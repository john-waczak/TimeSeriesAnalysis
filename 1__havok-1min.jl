using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations
using ProgressMeter
using SolarGeometry

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")
include("./performance-metrics.jl")



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

outpath = "./output/1__havok-1min"
if !ispath(outpath)
    mkpath(outpath)
end


# c4_path = joinpath(datapath, "central-hub-4")
# c7_path = joinpath(datapath, "central-hub-7")
# c10_path = joinpath(datapath, "central-hub-10")
v1_path = joinpath(datapath, "valo-node-01")



# create dictionary with lat/lon/alt information
pos_dict = Dict(
    "central-hub-4" => (;lat=33.02064, lon=-96.69869, alt=216.),
    "central-hub-7" => (;lat=32.71571, lon=-96.74801, alt=129.),
    "central-hub-10" => (;lat=33.0101, lon=-96.64375, alt=179.),
    "valo-node-1" => (;lat=32.967465, lon=-96.725647, alt=207.)
)

# load in CSVs and associated summaries
df = CSV.read(joinpath(v1_path, "df.csv"), DataFrame);
df_summary = CSV.read(joinpath(v1_path, "df_summary.csv"), DataFrame);

println("N days: ", maximum(df_summary.nrow) / (24*60))

names(df)


node_to_use = "valo-node-1"


cols_to_use = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0, :temperature, :pressure, :humidity, :dewPoint, :co2, :rainPerInterval]
col_names = ["PM 0.1", "PM 0.3", "PM 0.5", "PM 1.0", "PM 2.5", "PM 5.0", "PM 10.0", "Temperature", "Pressure", "Relative Humidity", "Dew Point", "CO₂", "Rainfall"]
col_units = ["μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "°C", "mbar", "%", "°C", "ppm", "mm/hr"]

df.sol_az = [solar_azimuth_altitude(df.datetime[i], pos_dict[node_to_use].lat, pos_dict[node_to_use].lon, pos_dict[node_to_use].alt)[1] for i in 1:nrow(df)]
df.sol_el = [solar_azimuth_altitude(df.datetime[i], pos_dict[node_to_use].lat, pos_dict[node_to_use].lon, pos_dict[node_to_use].alt)[2] for i in 1:nrow(df)]
df.sol_zenith = 90 .- df.sol_el


# save the data in case we need to investigate further:
CSV.write(joinpath(outpath, "df-joined.csv"), df)



# let's focus on PM 2.5
gp_long = df_summary.group[argmax(df_summary.nrow)]

# chop the dataframe to the selected range
df = df[df.group .== gp_long, :]
println("Data length: ", Day(Date(df.datetime[end]) - Date(df.datetime[1])))

day_map = Dict(
    1 => "Mon",
    2 => "Tue",
    3 => "Wed",
    4 => "Thu",
    5 => "Fri",
    6 => "Sat",
    7 => "Sun",
)

# 1 = Monday, 2 = Tuesday...
day_map[dayofweek(Date(df.datetime[1] + Day(1)))]

cutoff = 7*24*60

idx_start = findfirst([dayofweek(d) == Dates.Monday for d in df.datetime])
ts = df.dt[idx_start:idx_start + cutoff] .- df.dt[idx_start]
dt = ts[2]-ts[1]
Zs = Array(df[idx_start:idx_start + cutoff, :pm2_5])

zenith = df.sol_zenith[idx_start:idx_start + cutoff]
elevation = df.sol_el[idx_start:idx_start + cutoff]
temperature = df.temperature[idx_start:idx_start + cutoff]
dewpoint = df.dewPoint[idx_start:idx_start + cutoff]
humidity = df.humidity[idx_start:idx_start + cutoff]
rainfall = [ismissing(v) ? 0.0 : v for v ∈ df.rainPerInterval][idx_start:idx_start+cutoff]


# visualize the time series

cline = colorant"#ebc334"
cfill = colorant"#f0dfa1"
crain = colorant"#2469d6"

fig = Figure();
gl = fig[1,1] = GridLayout();

yticklabelsize=10
xticklabelsize=12
titlesize=10
titlegap=2

# temperature, humidity, solar elevation, rain_level
ax_rainfall = Axis(gl[1,1],
                   xticksvisible=false, xticklabelsvisible=false,
                   xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
                   yminorgridvisible=true, yticklabelsize=yticklabelsize,
                   yticks=([0, 8], ["0", "8"]),
                   title="Rainfall (mm/hr)", titlefont=:regular, titlealign=:left, titlesize=titlesize, titlegap=titlegap,
                   )

ax_humidity = Axis(gl[2,1],
                   xticksvisible=false, xticklabelsvisible=false,
                   xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
                   yminorgridvisible=true, yticklabelsize=yticklabelsize,
                   yticks=([0, 100], ["0", "100"]),
                   title="Humidity (%)", titlefont=:regular, titlealign=:left, titlesize=titlesize, titlegap=titlegap,
                   )

ax_temp = Axis(gl[3,1],
               xticksvisible=false, xticklabelsvisible=false,
               xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
               yminorgridvisible=true, yticklabelsize=yticklabelsize,
               yticks=([0, 50], ["0", "50"]),
               title="— Temperature (°C)    -- Dew Point (°C)", titlefont=:regular, titlealign=:left, titlesize=titlesize, titlegap=titlegap,
               )


ax_sun = Axis(gl[4,1],
              xticksvisible=false, xticklabelsvisible=false,
              xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
              yminorgridvisible=true,yticklabelsize=yticklabelsize,
              yticks=([0, 90], ["0", "90"]),
              title="Solar Elevation (degrees)", titlefont=:regular, titlealign=:left, titlesize=titlesize, titlegap=titlegap,
              )

ax = Axis(gl[5,1],
          xlabel="$(Date(df.datetime[idx_start])) to $(Date(df.datetime[idx_start+cutoff])) (UTC)",
          xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
          yticklabelsize=yticklabelsize,
          xticklabelsize=xticklabelsize,
          ylabel="PM 2.5 (μg/m³)",
          ylabelsize=15,
          xlabelsize=15,
          #title="PM 2.5 (μg/m³)", titlefont=:regular, titlealign=:left, titlesize=titlesize,
);


linkxaxes!(ax_sun, ax_temp, ax_humidity, ax_rainfall, ax)


# plot solar elevation
l_s = lines!(ax_sun, ts ./ (24*60), elevation, color=cline, linewidth=3)
band!(ax_sun, ts ./ (24*60), zeros(length(ts)), elevation, color=cfill)
ylims!(ax_sun, 0, 90)

# plot temperature
l_t = lines!(ax_temp, ts ./ (24*60), temperature, color=mints_colors[2], linewidth=2, label="Temperature")
l_tt = lines!(ax_temp, ts ./ (24*60), dewpoint, color=mints_colors[2], linewidth=1, linestyle=:dash, label="Dew Point")
ylims!(ax_temp, 0, 50)

# plot humidity
l_h = lines!(ax_humidity, ts ./ (24*60), humidity, color=crain, linewidth=2)
ylims!(ax_humidity, 0, 100)

# plot rainfall
l_r = lines!(ax_rainfall, ts ./ (24*60), rainfall, color=(:purple, 0.75), linewidth=2)
ylims!(ax_rainfall, 0, 8)


# plot PM 2.5
lines!(ax, ts ./ (24*60), Zs, color=mints_colors[3])
xlims!(ax, ts[1]/(24*60), ts[end]/(24*60))

rowsize!(gl, 1, Relative(0.125))
rowsize!(gl, 2, Relative(0.125))
rowsize!(gl, 3, Relative(0.125))
rowsize!(gl, 4, Relative(0.125))
rowsize!(gl, 5, Relative(0.5))


rowgap!(gl, 5)

# fig[1,1] = Legend(fig, [l_s, l_t, l_h, l_r], ["Solar Elevation", "Temperature", "Humidity", "Rainfall (mm/hr)"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=13, height=-5)

fig

save(joinpath(fig_savepath, "0_timeseries-original.png"), fig)



# let's combine the datasets for training and then
# train the model on the combined dataset

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
    # idx_int = 1:(7*24*60)
    # ts_int = ts_x[idx_int]

    prob = ODEProblem(f!, x₀, (ts_x[1], ts_x[end]), params);
    sol = solve(prob, saveat=ts_x);
    X̂ = Array(sol)'

    Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]

    return Zs_x, Ẑs_x, ts_x
end



# use the entire time series with 27 days
ts = df.dt .- df.dt[1]
dt = ts[2]-ts[1]
Zs = Array(df[:, :pm2_5])


# test it out
n_embedding = 6*60
r_model = 11
n_control = 1
method = :backward

Zs_x, Ẑs_x, ts_x = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM") #, xticks=0:7)
l_orig = lines!(ax, ts_x ./(24*60), Zs_x)
l_havok = lines!(ax, ts_x ./(24*60), Ẑs_x)
xlims!(ax, ts_x[1]/24/60, ts_x[end]/24/60)
ylims!(ax, 0, maximum(Zs_x))
fig



# test the performance measures
mean_bias(Ẑs_x, Zs_x)
mean_absolute_error(Ẑs_x, Zs_x)
normalized_mean_bias(Ẑs_x, Zs_x)
normalized_mae(Ẑs_x, Zs_x)
rmse(Ẑs_x, Zs_x)
corr_coef(Ẑs_x, Zs_x)
coefficient_of_efficiency(Ẑs_x, Zs_x)
r_squared(Ẑs_x, Zs_x)



## -------------------------------------------
##   EVALUATE HAVOK FOR PARAMETER VALUES
## -------------------------------------------


method=:backward
n_embeddings = Int.(60*[0.5, 1, 2, 3, 6, 8, 12, 16, 24])
rs_model = 3:25
ns_control = 1:5

# rs_model = 3:75
# ns_control = 1:50


# duration, mean_bias, mae, mean_bias_norm, mae-norm, rmse, corr_coef, coe, r2
eval_res = []

pm = Progress(length(rs_model))
for i ∈ 1:length(rs_model)
    r_model = rs_model[i]
    for j ∈ 1:length(ns_control)
        n_control = ns_control[j]
        for k ∈ 1:length(n_embeddings)
            n_embedding = n_embeddings[k]

            Zs_x, Ẑs_x, ts_x = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)

            durations = Dict(
                "1_hr" => 1:60,
                "12_hr" => 1:12*60,
                "1_day" => 1:24*60,
                "2_day" => 1:2*24*60,
                "3_day" => 1:3*24*60,
                "4_day" => 1:4*24*60,
                "5_day" => 1:5*24*60,
                "6_day" => 1:6*24*60,
                "7_day" => 1:7*24*60,
            )

            for (dur, idxs) in durations
                push!(eval_res, Dict(
                    "n_embedding" => n_embedding,
                    "r_model" => r_model,
                    "n_control" => n_control,
                    "duration" => dur,
                    "mean_bias" => mean_bias(Ẑs_x[idxs], Zs_x[idxs]),
                    "mae" => mean_absolute_error(Ẑs_x[idxs], Zs_x[idxs]),
                    "mean_bias_norm" => normalized_mean_bias(Ẑs_x[idxs], Zs_x[idxs]),
                    "mae_norm" => normalized_mae(Ẑs_x[idxs], Zs_x[idxs]),
                    "rmse" => rmse(Ẑs_x[idxs], Zs_x[idxs]),
                    "corr_coef" => corr_coef(Ẑs_x[idxs], Zs_x[idxs]),
                    "coe" => coefficient_of_efficiency(Ẑs_x[idxs], Zs_x[idxs]),
                    "r2" => r_squared(Ẑs_x[idxs], Zs_x[idxs]),
                ))
            end

        end
    end

    next!(pm; showvalues=[(:r_model, r_model),])
end


# turn into CSV and save to output
df_res = DataFrame(eval_res)
CSV.write(joinpath(outpath, "param_sweep.csv"), df_res)


# create plots for results:
dur = "3_day"

df_plot = df_res[df_res.duration .== dur .&& df_res.n_control .== 1, :]
names(df_plot)


fig = Figure();

idx_min = argmin(df_plot.rmse)
ax = CairoMakie.Axis(fig[1,1], xlabel="embedding length (hr)", ylabel="r", xticks = n_embeddings ./ 60, xticklabelsize=10)
hm = heatmap!(ax, df_plot.n_embedding ./ 60, df_plot.r_model, df_plot.rmse, colormap=:thermal, colorrange=(0, 10))
sc = scatter!(ax, [df_plot.n_embedding[idx_min] ./60], [df_plot.r_model[idx_min]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="1 day RMSE", ticks=0:10, minorticksvisible=true)

println(df_plot.rmse[idx_min])

fig

# save(joinpath(fig_savepath, "2__havok-rmse-1-hour.png"), fig)
# save(joinpath(fig_savepath, "2__havok-rmse-1-hour.pdf"), fig)


# -------------------------------------------------------------
# Collect results
# -------------------------------------------------------------

df_res_best = df_res[df_res.duration .== "3_day", :]

idx_best = argmin(df_res_best.rmse)
r_model_best = df_res_best[idx_best, :r_model]
n_control_best = df_res_best[idx_best, :n_control]
n_embedding_best = df_res_best[idx_best, :n_embedding]
println("rmse best: ", df_res_best.rmse[idx_best])

# n_control == 3
df_res_best = df_res[df_res.duration .== "3_day" .&& df_res.n_control .== 3, :]
idx_best = argmin(df_res_best.rmse)
n_control_3 = 3
r_model_3 = df_res_best[idx_best, :r_model]
n_embedding_3 = df_res_best[idx_best, :n_embedding]
println("rmse n=3: ", df_res_best.rmse[idx_best])

# n_control == 1
df_res_best = df_res[df_res.duration .== "3_day" .&& df_res.n_control .== 1, :]
idx_best = argmin(df_res_best.rmse)
n_control_1 = 1
r_model_1 = df_res_best[idx_best, :r_model]
n_embedding_1 = df_res_best[idx_best, :n_embedding]
println("rmse n=1: ", df_res_best.rmse[idx_best])




# # Create plot comparing the 3 best models:
# Zs_x, Ẑs_x_best, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_best, n_control_best; method=method)
# Zs_x, Ẑs_x_mid, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_mid, n_control_mid; method=method)
# Zs_x, Ẑs_x_small, ts_x = eval_havok(Zs_smooth, ts, n_embedding, r_model_small, n_control_small; method=method)


# fig = Figure();
# ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
# l_orig = lines!(ax, ts_x[1:3*24*60]./(24*60), Zs_x[1:3*24*60], linewidth=3)
# l_best = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_best[1:3*24*60], linewidth=2)
# l_mid = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_mid[1:3*24*60], linewidth=2)
# l_small = lines!(ax, ts_x[1:3*24*60]./(24*60), Ẑs_x_small[1:3*24*60], linewidth=2)
# xlims!(ax, ts_x[1]./(24*60), 3)
# fig[1,2] = Legend(fig, [l_orig, l_best, l_mid, l_small], ["Original", "HAVOK ($(r_model_best), $(n_control_best))","HAVOK ($(r_model_mid), $(n_control_mid))", "HAVOK ($(r_model_small), $(n_control_small))" ], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17, height=-5)
# fig

# save(joinpath(fig_savepath, "7__havok-reconstruction.png"), fig)
# save(joinpath(fig_savepath, "7__havok-reconstruction.pdf"), fig)



# fig = Figure();
# ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)");
# l_orig = lines!(ax, ts_x./(24*60), Zs_x, linewidth=3)
# l_best = lines!(ax, ts_x./(24*60), Ẑs_x_best, linewidth=2)
# l_mid = lines!(ax, ts_x./(24*60), Ẑs_x_mid, linewidth=2)
# l_small = lines!(ax, ts_x./(24*60), Ẑs_x_small, linewidth=2)
# xlims!(ax, ts_x[1]./(24*60), ts_x[end] ./(24*60))
# fig[1,2] = Legend(fig, [l_orig, l_best, l_mid, l_small], ["Original", "HAVOK ($(r_model_best), $(n_control_best))","HAVOK ($(r_model_mid), $(n_control_mid))", "HAVOK ($(r_model_small), $(n_control_small))" ], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17, height=-5)
# fig

# save(joinpath(fig_savepath, "8__havok-reconstruction-long.png"), fig)
# save(joinpath(fig_savepath, "8__havok-reconstruction-ling.pdf"), fig)

