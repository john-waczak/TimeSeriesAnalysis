using CairoMakie
using MintsMakieRecipes

using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones

include("utils.jl")
include("viz.jl")


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

@assert all(ispath.(datapaths_cn))


# generate dictionaries holding the datapaths for central hub

time_types = [
    "1-min",
    "5-min",
    "15-min",
    "hour",
    "eight-hour",
    "day"
]


# start with 1 min dataset for cn 4
basepath = datapaths_cn[1]
df_summary = CSV.read(joinpath(basepath, "df-"*time_types[1]*"_summary.csv"), DataFrame)

df_summary


df = DataFrame()

df_summary
gdf[(group=540,)].datetime[end]
gdf[(group=541,)].datetime[1]

let
    df_full = CSV.read(joinpath(basepath, "df-"*time_types[1]*".csv"), DataFrame)
    idx_max = argmax(df_summary.nrow)
    group_max = df_summary.group[idx_max]

    gdf = groupby(df_full, :group)
    global df = gdf[(group=group_max,)]

    # handle date column
    df.datetime = String.(df.datetime)
    df.datetime = parse.(ZonedDateTime, df.datetime)
end


t_start = df.datetime[1]
t_end = df.datetime[end]
println("The dataset covers ", round(t_end - t_start, Day))

# get times for plotting
ts = [(dt .- t_start).value ./(1000*60*60*24) for dt in df.datetime]

col_to_use = :pm2_5

z = df[:, col_to_use]


# visualize the time-series
fig = Figure();
ax = Axis(fig[1,1], xlabel="t (days since $(Date(t_start)))", ylabel="PM 2.5 (μg⋅m⁻³)")
xlims!(ax, 0, 5)
ylims!(ax, 0, 15)
lines!(ax, ts, z)
fig


# Get Hankel Matrix of time-delay embeddings
nrow = 100
H = TimeDelayEmbedding(z, nrow=nrow)

# compute singular value decomposition
U, σ, V = svd(H)


# visualize the embedded attractor
fig = Figure();
ax = Axis3(
    fig[1,1];
    xlabel="v₁",
    ylabel="v₂",
    zlabel="v₃",
    # aspect = :data,
    azimuth=-35π/180,
    elevation=37π/180,
    xticksvisible=false,
    yticksvisible=false,
    zticksvisible=false,
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    zticklabelsvisible=false,
    xlabeloffset=5,
    ylabeloffset=5,
    zlabeloffset=5,
    title="Embedded Attractor"
);


l1 = lines!(ax, V[:,1], V[:,2], V[:,3], color=ts[1:size(V,1)], colormap=:inferno, linewidth=3)

fig


size(V)

