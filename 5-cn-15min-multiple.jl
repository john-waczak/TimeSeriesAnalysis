using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")





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
    "1-min",
    "15-min",
    "1-hour",
]


f_to_use = 1
datapath_cn = datapaths_cn[f_to_use]
fig_savepath = fig_savepaths[f_to_use]

# load in a csv and it's associated summary
df = CSV.read(joinpath(datapath_cn, "df-"*time_types[2]*".csv"), DataFrame);
df_summary = CSV.read(joinpath(datapath_cn, "df-"*time_types[2]*"_summary.csv"), DataFrame);


df.datetime[1]
df.datetime[end]

# parse datetime to correct type
df.datetime = String.(df.datetime);
df.datetime = parse.(ZonedDateTime, df.datetime);

# compute embedding dimension to acheive at least 24 hour coverage
dt = (df.datetime[2] - df.datetime[1]).value / (1000*60*60*24)

# use a day's worth of data for the feature vector
#n_embedding = min(ceil(Int,1/dt), 100)
n_embedding = 100
n_derivative = 5
r_cutoff = 18
n_control = 1
r = r_cutoff + n_control - 1


# select groups which have nrow ≥ n_embedding + n_derivative
idx_good = findall(df_summary.nrow .≥ n_embedding + n_derivative)
groups_to_use = df_summary.group[idx_good]
df = df[findall([g ∈ groups_to_use for g ∈ df.group]), :]
gdf = groupby(df, :group)

# get idxs for each group
col_to_use = :pm2_5
t_start = df.datetime[1]
t_end = df.datetime[end]

Zs = []
ts = []

for dfᵢ ∈ gdf
    push!(Zs, dfᵢ[:, col_to_use])
    push!(ts, dfᵢ.dt)
end



# visualize the time-series
x_tick_months = t_start:Month(1):(t_end + Month(1))
x_tick_pos = [Minute(d - t_start).value for d ∈ x_tick_months]
x_tick_strings = [Dates.format.(d, "mm-yy") for d ∈ x_tick_months]



fig = Figure();
ax = Axis(
    fig[1,1],
    xlabel="time", ylabel="PM 2.5 (μg⋅m⁻³)",
    xticks=(x_tick_pos, x_tick_strings), xticklabelrotation=π/3,
);
xlims!(ax, 0, x_tick_pos[end])

ls = []
for i ∈ 1:length(Zs)
    l = lines!(ax, ts[i], Zs[i], linewidth=3)
    push!(ls, l)
end

fig

save(joinpath(fig_savepath, "timeseries-grouped.png"), fig)




# Get Hankel Matrix of time-delay embeddings
# generate individual Hankel matrices
Hs = [TimeDelayEmbedding(Z, nrow=n_embedding) for Z ∈ Zs];
ts_s = [t[n_embedding:end] for t ∈ ts];
H = hcat(Hs...);

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
U, σ, V = svd(H);


# plot singular values
fig = Figure();
ax = Axis(fig[1,1], xlabel="index i", ylabel="σᵢ", title="Scaled Singular Values")
vlines!(ax, [r_cutoff], color=mints_colors[2], label="r = $(r_cutoff)")
scatter!(ax, σ ./ maximum(σ),)
axislegend(ax)
fig

save(joinpath(fig_savepath, "singular-values.pdf"), fig)


Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]

# Vrs = [Vr[idx, :] for idx ∈ idxs_H]
Vrs = [Vr[idxs_H[2], :],]


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


for i ∈ 1:length(Vrs)
    Vᵢ = Vrs[i]
    l1 = lines!(ax, Vᵢ[:,1], Vᵢ[:,2], Vᵢ[:,3], color=ts[i][n_embedding:end], colormap=:inferno, linewidth=3)
end

fig

save(joinpath(fig_savepath, "svd-attarctor.png"), fig)





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


# solve for HAVOK coefficients
Ξ = (X\dX)'

A = Ξ[:, 1:r-n_control]          # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B

fig = Figure();
gl = fig[1,1:2] = GridLayout()
ax1 = Axis(gl[1,1];
           yreversed=true,
           #xlabel="A",
           title="A",
           titlesize=50,
           xticklabelsvisible=false,
           yticklabelsvisible=false,
           xticksvisible=false,
           yticksvisible=false,
           )

ax2 = Axis(gl[1,2];
           yreversed=true,
           #xlabel="B",
           title="B",
           titlesize=50,
           xticklabelsvisible=false,
           yticklabelsvisible=false,
           xticksvisible=false,
           yticksvisible=false,
           )

h1 = heatmap!(ax1, A, colormap=:inferno)
h2 = heatmap!(ax2, B', colormap=:inferno)

colsize!(gl, 2, Relative(n_control/r)) # scale control column to correct size
cb = Colorbar(fig[1,3], limits = extrema(Ξ), colormap=:inferno)
#cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
fig

save(joinpath(fig_savepath, "operator-heatmap.pdf"), fig)





# visualize eigenmodes
fig = Figure();
ax = Axis(fig[1,1], title="Eigenmodes");

ls1 = []
ls2 = []
lr = []

for i ∈ 1:r_cutoff
    if i ≤ 3
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=mints_colors[1], linewidth=3)
        push!(ls1, l)
    elseif i > 3 && i < r_cutoff
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=:grey, alpha=0.5, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

axislegend(ax, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋮", "uᵣ"])

fig

save(joinpath(fig_savepath, "svd-eigenmodes.pdf"), fig)




ts_full = vcat(ts_s...)
itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_full) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


ts_x = [t[3:end-3] for t ∈ ts_s]
ts_x_full = vcat(ts_x...)

xr = vcat(u.(ts_x_full)...)
size(xr)
size(X)


function f!(dx, x, (A,B), t)
    dx .= A*x  + B*u(t)
end

X̂s = []
X̂s_test = []


for i ∈ 1:length(Xs)
    ps = (A, B)

    x₀ = Xs[i][1,1:r-n_control]
    dx = copy(x₀)
    @assert size(x₀) == size(dx)

    prob = ODEProblem(f!, x₀, (ts_x[i][1], ts_x[i][end]), ps)
    sol = solve(prob, saveat=ts_x[i]);

    X̂ = Matrix(sol)'
    X̂test = Matrix(sol)'

    size(Xs[1])
    size(X̂)

    push!(X̂s, X̂)
    push!(X̂s_test, X̂test)
end


