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

v1_path = joinpath(datapath, "valo-node-01")
df = CSV.read(joinpath(v1_path, "df.csv"), DataFrame);
df.datetime .= ZonedDateTime.(String.(df.datetime))
sort!(df, :datetime)

df_summary = CSV.read(joinpath(v1_path, "df_summary.csv"), DataFrame);
println("N days (max): ", maximum(df_summary.nrow) / (24*60))
names(df)


# let's focus on PM 2.5
gp_long = df_summary.group[argmax(df_summary.nrow)]

# chop the dataframe to the selected range
df = df[df.group .== gp_long, :];
println("Data length: ", Day(Date(df.datetime[end]) - Date(df.datetime[1])))

dt_start = ZonedDateTime(Date(df.datetime[1]), timezone(df.datetime[1]))
t_days = [ d.value/(1000*60*60*24) for d ∈ df.datetime .- dt_start]



# visualize the time series

fig = Figure();
ax = Axis(fig[1,1], xlabel="Days since $(Date(dt_start))", ylabel="PM 2.5 (μg/m³");
lines!(ax, t_days, df.pm2_5, color=mints_colors[3])
xlims!(ax, 0, t_days[end])
ylims!(ax, 0, nothing)
save(joinpath(fig_savepath, "0__timeseries-original.png"), fig)
fig

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

# visualize the time series
fig = Figure();
ax = Axis(fig[1,1],
          xlabel="$(Date(df.datetime[idx_start])) to $(Date(df.datetime[idx_start+cutoff]))",
          xticks=(0:7, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun", "Mon"]),
          ylabel="PM 2.5 (μg/m³)",
          );
lines!(ax, ts ./ (24*60), Zs, color=mints_colors[3])
xlims!(ax, ts[1]/(24*60), ts[end]/(24*60))
ylims!(ax, 0, 15)
fig

save(joinpath(fig_savepath, "0b_timeseries-1-week.png"), fig)




# use the entire time series with 27 days
t_days
ts = df.dt .- df.dt[1]
dt = ts[2]-ts[1]
Zs = Array(df[:, :pm2_5])

minimum(Zs)
mean(Zs)
median(Zs)
maximum(Zs)
std(Zs)

# test it out
n_embedding = 6*60
r_model = 11
n_control = 1
method = :backward

Zs_x, Ẑs_x, ts_x, idx_ts = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="PM")
l_orig = lines!(ax, t_days[idx_ts], Zs_x)
l_havok = lines!(ax, t_days[idx_ts], Ẑs_x)
xlims!(ax, t_days[1], t_days[end])
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
n_embeddings = Int.(60*[0.5, 1, 2, 3, 6, 8, 12, 16, 24])  # in hr increments
rs_model = 3:25
ns_control = 1:5

println("N models: ", length(n_embeddings)*length(rs_model)*length(ns_control))

# duration, mean_bias, mae, mean_bias_norm, mae-norm, rmse, corr_coef, coe, r2
eval_res = []

for i ∈ 1:length(rs_model)
    r_model = rs_model[i]
    for j ∈ 1:length(ns_control)
        n_control = ns_control[j]
        println("r: ", r_model, "\tn: ", n_control)
        @showprogress for k ∈ 1:length(n_embeddings)
            n_embedding = n_embeddings[k]

            Zs_x, Ẑs_x, ts_x, idx_ts = eval_havok(Zs, ts, n_embedding, r_model, n_control; method=method)

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
end


# turn into CSV and save to output
df_res = DataFrame(eval_res)
CSV.write(joinpath(outpath, "param_sweep.csv"), df_res)

extrema(Zs[1:3*24*60])
mean(Zs[1:3*24*60])

# create plots for results:
names(df_res)

dur = "12_hr"
dur = "1_day"
sort(df_res[df_res.duration .== dur .&& df_res.n_control .== 1 .&& df_res.r_model .<= 10, [:r_model, :n_control, :n_embedding, :rmse, :mae]], :rmse)

# r_model = 5, n_control=1, n_embedding=30


dur = "1_hr"
df_plot = df_res[df_res.duration .== dur .&& df_res.n_control .== 1, :]

fig = Figure();
idx_min = argmin(df_plot.rmse)
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="embedding length (hr)", ylabel="r",
    xticks = (1:length(n_embeddings), string.(n_embeddings ./ 60)),
)
n_plot = [findfirst(n .== n_embeddings) for n ∈ df_plot.n_embedding]
hm = heatmap!(ax, n_plot, df_plot.r_model, df_plot.rmse, colormap=:thermal, colorrange=(0, 10))
sc = scatter!(ax, [n_plot[idx_min]], [df_plot.r_model[idx_min]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="RMSE - 1 hour", ticks=0:10, minorticksvisible=true)
fig

save(joinpath(fig_savepath, "1__rmse-compare-1-hr.png"), fig)


dur = "12_hr"
df_plot = df_res[df_res.duration .== dur .&& df_res.n_control .== 1, :]
fig = Figure();
idx_min = argmin(df_plot.rmse)
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="embedding length (hr)", ylabel="r",
    xticks = (1:length(n_embeddings), string.(n_embeddings ./ 60)),
)
n_plot = [findfirst(n .== n_embeddings) for n ∈ df_plot.n_embedding]
hm = heatmap!(ax, n_plot, df_plot.r_model, df_plot.rmse, colormap=:thermal, colorrange=(0, 10))
sc = scatter!(ax, [n_plot[idx_min]], [df_plot.r_model[idx_min]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="RMSE - 12 hours", ticks=0:10, minorticksvisible=true)
fig

save(joinpath(fig_savepath, "1b_rmse-compare-12-hr.png"), fig)


dur = "1_day"
df_plot = df_res[df_res.duration .== dur .&& df_res.n_control .== 1, :]
fig = Figure();
idx_min = argmin(df_plot.rmse)
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="embedding length (hr)", ylabel="r",
    xticks = (1:length(n_embeddings), string.(n_embeddings ./ 60)),
)
n_plot = [findfirst(n .== n_embeddings) for n ∈ df_plot.n_embedding]
hm = heatmap!(ax, n_plot, df_plot.r_model, df_plot.rmse, colormap=:thermal, colorrange=(0, 10))
sc = scatter!(ax, [n_plot[idx_min]], [df_plot.r_model[idx_min]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="RMSE - 1 day", ticks=0:10, minorticksvisible=true)
fig

save(joinpath(fig_savepath, "1c_rmse-compare-1-day.png"), fig)


dur = "3_day"
df_plot = df_res[df_res.duration .== dur .&& df_res.n_control .== 1, :]
fig = Figure();
idx_min = argmin(df_plot.rmse)
ax = CairoMakie.Axis(
    fig[1,1],
    xlabel="embedding length (hr)", ylabel="r",
    xticks = (1:length(n_embeddings), string.(n_embeddings ./ 60)),
)
n_plot = [findfirst(n .== n_embeddings) for n ∈ df_plot.n_embedding]
hm = heatmap!(ax, n_plot, df_plot.r_model, df_plot.rmse, colormap=:thermal, colorrange=(0, 10))
sc = scatter!(ax, [n_plot[idx_min]], [df_plot.r_model[idx_min]], marker=:star5, color=:white, markersize=10)
cb = Colorbar(fig[1,2], hm, label="RMSE - 3 day", ticks=0:10, minorticksvisible=true)
fig

save(joinpath(fig_savepath, "1c_rmse-compare-1-day.png"), fig)



# -------------------------------------------------------------
# Plot vanilla model for r=5, n=1
# -------------------------------------------------------------

n_emb_best = 30
n_best = 1
r_best = 5
Zs_x, Ẑs_x, ts_x, idx_ts = eval_havok(Zs, ts, n_emb_best, r_best, n_best; method=method)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)")
l_orig = lines!(ax, t_days[idx_ts], Zs_x)
l_havok = lines!(ax, t_days[idx_ts], Ẑs_x, alpha=0.75)
xlims!(ax, t_days[1], t_days[1]+3)
ylims!(ax, 0, 12)
fig[1,1] = Legend(fig, [l_orig, l_havok], ["Original", "HAVOK ($(r_best), $(n_best))",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig

save(joinpath(fig_savepath, "2__havok-reconstruction-3-day.png"), fig)


fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)")
l_orig = lines!(ax, t_days[idx_ts], Zs_x)
l_havok = lines!(ax, t_days[idx_ts], Ẑs_x, alpha=0.75)
xlims!(ax, t_days[1], t_days[1]+1)
ylims!(ax, 3, 9)
fig[1,1] = Legend(fig, [l_orig, l_havok], ["Original", "HAVOK ($(r_best), $(n_best))",], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig

save(joinpath(fig_savepath, "2b_havok-reconstruction-1-day.png"), fig)


