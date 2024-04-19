# HAVOK Model for Single Central Node
# Using 15-min data of joined time
# series with gaps


# File Handling
using CSV, DataFrames, Dates

# SciML Tools
using DifferentialEquations, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics, StatsBase, Distributions, KernelDensity

# External Libraries
using ComponentArrays, Lux, Zygote, DataInterpolations, StableRNGs

# Plotting
using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")

# Set a random seed for reproducible behaviour
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
t_to_use = 3
datapath_cn = datapaths_cn[f_to_use]
fig_savepath = joinpath(fig_savepaths[f_to_use], "df-15-min-nn")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


# load in a csv and it's associated summary
df = CSV.read(joinpath(datapath_cn, "df-"*time_types[t_to_use]*".csv"), DataFrame);
df_summary = CSV.read(joinpath(datapath_cn, "df-"*time_types[t_to_use]*"_summary.csv"), DataFrame);

dt = (df.datetime[2]-df.datetime[1]).value / 1000
println(Second(df.datetime[2] - df.datetime[1]))


n_embedding = 100
n_derivative = 5
r_cutoff = 15
n_control = 1
r = r_cutoff + n_control - 1


# create a single dataset interpolated to every second
col_to_use = :pm2_5


t_start = df.datetime[1]
t_end = df.datetime[end]

df.dt = [(Second(dt - t_start)).value for dt ∈ df.datetime]


Zs = []
ts = []

gdf = groupby(df, :group)

for df_g ∈ gdf
    if nrow(df_g) .≥ n_embedding + n_derivative
        push!(Zs, df_g[:, col_to_use])
        push!(ts, df_g.dt)
    end
end



# generate Hankel Matrices
Hs = [TimeDelayEmbedding(Z, nrow=n_embedding; method=:backward) for Z ∈ Zs];
ts_s = [t[n_embedding:end] for t ∈ ts];
H = hcat(Hs...)


# generate indices of each component matrix so we can split
# after doing the SVD
idxs_H = []
i_now = 1
for i ∈ 1:length(Hs)
    Hᵢ = Hs[i]
    push!(idxs_H, i_now:i_now + size(Hᵢ,2) - 1)
    i_now += size(Hᵢ,2)
end


# compute singular value decomposition
U, σ, V = svd(H)

Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]

size(H)

Vrs = [Vr[idx, :] for idx ∈ idxs_H]


# set up data and derivatives
Xs = []
dXs = []

for i ∈ 1:length(Vrs)
    Vrᵢ = Vrs[i]
    dVrᵢ = zeros(size(Vrᵢ,1)-5, r-n_control)

    Threads.@threads for k ∈ 1:r-n_control
        for i ∈ 3:size(Vrᵢ,1)-3
            @inbounds dVrᵢ[i-2,k] = (1/(12*dt)) * (-Vrᵢ[i+2, k] + 8*Vrᵢ[i+1, k] - 8*Vrᵢ[i-1,k] + Vrᵢ[i-2,k])
        end
    end

    push!(Xs, Vrᵢ[3:end-3,:])
    push!(dXs, dVrᵢ)
end


X = vcat(Xs...);
dX = vcat(dXs...);

# fit model matrix

Ξ = (X\dX)'  # now Ξx = dx for a single column vector view
A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B



# define interpolation function for forcing coordinate(s)
#   +3 because of offset from derivative...
ts_x = [range(ts[i][n_embedding+2], step=dt, length=size(dXs[i],1)) for i ∈ 1:length(Xs)]
ts_full = vcat(ts_x...)
itps = [DataInterpolations.LinearInterpolation(X[:,j], ts_full) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# define function and integrate to get model predictions
function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*u(t)
end

X̂s = []

for i ∈ 1:length(Xs)
    params = (A, B)
    x₀ = Xs[i][1,1:r-n_control]
    dx = copy(x₀)

    prob = ODEProblem(f!, x₀, (ts_x[i][1], ts_x[i][end]), params);
    sol = solve(prob, saveat=ts[i]);

    X̂ = Matrix(sol)'
    push!(X̂s, X̂)
end




# Reconstruct original time series from predictions

size(X)
X_f = zeros(size(X,1), n_control)

for i ∈ axes(X_f,1)
    X_f[i,:] .= u(ts_full[i])
end


all(isapprox.(U*diagm(σ)*V', H; rtol=0.1))

Ĥs = []
for i ∈ 1:length(X̂s)
    Xf = hcat(u(ts_x[i])...)
    Ĥ = Ur*diagm(σr)*hcat(X̂s[i], Xf)'
    push!(Ĥs, Ĥ)
end


