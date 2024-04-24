# HAVOK Model for Single Central Node
# Using 15-min data of joined time
# series with gaps

using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations


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

@assert all(ispath.(datapaths_cn))


time_types = [
    "1-sec",
    "1-min",
    "15-min",
    "1-hour",
]

fig_savepath = joinpath(figpath, "mult-central-hub", "15-min-gap")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end

t_to_use = 3


# load in a csvs and associated summarys
dfs = Dict(
    :4 => Dict(
        :df => CSV.read(joinpath(datapaths_cn[1], "df-"*time_types[t_to_use]*".csv"), DataFrame),
        :summary => CSV.read(joinpath(datapaths_cn[1], "df-"*time_types[t_to_use]*"_summary.csv"), DataFrame),
        :Zs => Vector{Float64}[],
        :ts => Vector{Int64}[],
        :Hs => Matrix{Float64}[],
        :ts_s => Vector{Int64}[],
        :idxs_H => Vector{Int64}[],
    ),
    :7 => Dict(
        :df => CSV.read(joinpath(datapaths_cn[2], "df-"*time_types[t_to_use]*".csv"), DataFrame),
        :summary => CSV.read(joinpath(datapaths_cn[2], "df-"*time_types[t_to_use]*"_summary.csv"), DataFrame),
        :Zs => Vector{Float64}[],
        :ts => Vector{Int64}[],
        :Hs => Matrix{Float64}[],
        :ts_s => Vector{Int64}[],
        :idxs_H => Vector{Int64}[],
    ),
    :10 => Dict(
        :df => CSV.read(joinpath(datapaths_cn[3], "df-"*time_types[t_to_use]*".csv"), DataFrame),
        :summary => CSV.read(joinpath(datapaths_cn[3], "df-"*time_types[t_to_use]*"_summary.csv"), DataFrame),
        :Zs => Vector{Float64}[],
        :ts => Vector{Int64}[],
        :Hs => Matrix{Float64}[],
        :ts_s => Vector{Int64}[],
        :idxs_H => Vector{Int64}[],
    ),
)


dt4 = (dfs[:4][:df].datetime[2]-dfs[:4][:df].datetime[1]).value / 1000
dt7 = (dfs[:7][:df].datetime[2]-dfs[:7][:df].datetime[1]).value / 1000
dt10 = (dfs[:10][:df].datetime[2]-dfs[:10][:df].datetime[1]).value / 1000

@assert (dt4 == dt7) && (dt7 == dt10)

dt = dt4

nhours = 48
n_embedding = max(round(Int, nhours*60*60 / dt), 100)
n_derivative = 5
r_cutoff = 18
n_control = 10
r = r_cutoff + n_control - 1

col_to_use = :pm2_5


# get global start and end time
t_starts = []
t_ends = []
for (node, df_dict) ∈ dfs
    t_start = df_dict[:df].datetime[1]
    t_end = df_dict[:df].datetime[end]
    push!(t_starts, t_start)
    push!(t_ends, t_end)
end

t_start = minimum(t_starts)
t_end = minimum(t_ends)


# add dt to each dataframe
for (node, df_dict) ∈ dfs
    df_dict[:df].dt = [(Second(dt - t_start)).value for dt ∈ df_dict[:df].datetime]
end


# pick out the column we want
for (node, df_dict) ∈ dfs

    gdf = groupby(df_dict[:df], :group)
    for df_g ∈ gdf
        if nrow(df_g) .≥ n_embedding + n_derivative
            push!(df_dict[:Zs], Vector(df_g[:, col_to_use]))
            push!(df_dict[:ts], Vector(df_g.dt))
        end
    end
end



# visualize the time-series
#x_tick= t_start:Day(1):t_end
x_tick= (round(t_start, Month)-Month(1)):Month(2):(round(t_end, Month) + Month(1))
x_tick_pos = [Second(d - t_start).value for d ∈ x_tick]
x_tick_strings = [Dates.format.(d, "mm/yy") for d ∈ x_tick]

fig = Figure();
ax = Axis(fig[2,1], xlabel="time", ylabel="PM 2.5 (μg⋅m⁻3)", xticks=(x_tick_pos, x_tick_strings), xticklabelrotation=π/3);

ls = []
idx_color = 1
for node ∈ keys(dfs)
    Zs = dfs[node][:Zs]
    ts = dfs[node][:ts]

    for i ∈ 1:length(Zs)
        li = lines!(ax, ts[i], Zs[i], linewidth=2, color=(mints_colors[idx_color], 0.75))
        if i == 1
            push!(ls, li)
        end
    end

    idx_color += 1
end

fig[1,1] = Legend(fig, ls, "Central Node " .* string.(keys(dfs)), framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

save(joinpath(fig_savepath, "original-timeseries.png"), fig)


# Generate time delay embeddings for each slice
for (node, df_dict) ∈ dfs
    df_dict[:Hs] = [TimeDelayEmbedding(Z, nrow=n_embedding; method=:backward) for Z ∈ df_dict[:Zs]];
    df_dict[:ts_s] = [t[n_embedding:end] for t ∈ df_dict[:ts]];
end


# generate indices of each component matrix so we can split
# after doing the SVD
i_now = 1
for i ∈ 1:length(dfs[:4][:Hs])
    Hᵢ = dfs[:4][:Hs][i]
    push!(dfs[:4][:idxs_H], i_now:i_now + size(Hᵢ,2) - 1)
    i_now += size(Hᵢ,2)
end

for i ∈ 1:length(dfs[:7][:Hs])
    Hᵢ = dfs[:7][:Hs][i]
    push!(dfs[:7][:idxs_H], i_now:i_now + size(Hᵢ,2) - 1)
    i_now += size(Hᵢ,2)
end

for i ∈ 1:length(dfs[:10][:Hs])
    Hᵢ = dfs[:10][:Hs][i]
    push!(dfs[:10][:idxs_H], i_now:i_now + size(Hᵢ,2) - 1)
    i_now += size(Hᵢ,2)
end


size(dfs[:4][:Hs])
size(dfs[:4][:ts_s])
size(dfs[:4][:Zs])
size(dfs[:4][:ts])
size(dfs[:4][:idxs_H])


H = hcat(
    dfs[:4][:Hs]...,
    dfs[:7][:Hs]...,
    dfs[:10][:Hs]...,
);



# compute singular value decomposition
U, σ, V = svd(H);

# r_cut(σ, ratio=0.05, rmax=100)              # 42
# r_optimal_approx(σ, size(V,2), size(V,1))   # 162

Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]



Vrs = [Vr[idx, :] for idx ∈ vcat(dfs[:4][:idxs_H], dfs[:7][:idxs_H], dfs[:10][:idxs_H])]
ts = vcat(dfs[:4][:ts_s], dfs[:7][:ts_s], dfs[:10][:ts_s])


# visualize the attractor:
fig = Figure();
ax1 = Axis3(fig[1,1];
            xlabel="v₁",
            ylabel="v₂",
            zlabel="v₃",
            # aspect = :data,
            azimuth=-35π/180,
            elevation=30π/180,
            xticksvisible=false,
            yticksvisible=false,
            zticksvisible=false,
            xticklabelsvisible=false,
            yticklabelsvisible=false,
            zticklabelsvisible=false,
            xlabeloffset=5,
            ylabeloffset=5,
            zlabeloffset=5,
            title="Original Attractor"
            );
for i ∈ 1:length(Vrs)
    lines!(ax1, Vrs[i][:,1], Vrs[i][:,2], Vrs[i][:,3], color=ts[i], colormap=:inferno, linewidth=3)
end

fig

save(joinpath(fig_savepath, "svd-attractor.png"), fig)



# visualize singular values
fig = Figure();
ax = Axis(fig[1,1]; xlabel="index", ylabel="Normalized singular value")
lines!(ax, σ./sum(σ), linewidth=3)
vlines!(ax, [r_cutoff], linewidth=3, color=mints_colors[2], label="r = $(r_cutoff)")
axislegend(ax)
fig

save(joinpath(fig_savepath, "singular-values.pdf"), fig)



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


# visualize the learned operators
fig = Figure();
gl = fig[1,1:2] = GridLayout()
ax1 = Axis(gl[1,1];
           yreversed=true,
           xlabel="A",
           xticklabelsvisible=false,
           yticklabelsvisible=false,
           xticksvisible=false,
           yticksvisible=false,
           )

ax2 = Axis(gl[1,2];
           yreversed=true,
           xlabel="B",
           xticklabelsvisible=false,
           yticklabelsvisible=false,
           xticksvisible=false,
           yticksvisible=false,
           )

h1 = heatmap!(ax1, A, colormap=:inferno)
h2 = heatmap!(ax2, B', colormap=:inferno)
colsize!(gl, 2, Relative(n_control/r))
cb = Colorbar(fig[1,3], limits=extrema(Ξ), colormap=:inferno)
fig

fig
save(joinpath(fig_savepath, "operator-heatmap.pdf"), fig)



# visualize eigenmodes
fig = Figure();
ax = Axis(fig[1,1], title="Eigenmodes");

ls1 = []
ls2 = []
lr = []
# p = plot([], yticks=[-0.3, 0.0, 0.3], legend=:outerright, label="")

for i ∈ 1:r
    if i ≤ 3
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=mints_colors[1], linewidth=3)
        push!(ls1, l)
    elseif i > 3 && i < r
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=:grey, alpha=0.2, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:n_embedding, Ur[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

axislegend(ax, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋮", "uᵣ₁"])

fig

save(joinpath(fig_savepath, "svd-eigenmodes.pdf"), fig)





# define interpolation function for forcing coordinate(s)
#   +3 because of offset from derivative...

# set up interpolation for single sensor now...
Zs = dfs[:4][:Zs];
Hs = dfs[:4][:Hs];
Vrs = Vrs[1:length(dfs[:4][:Hs])];
Xs = Xs[1:length(dfs[:4][:Hs])];
dXs = dXs[1:length(dfs[:4][:Hs])];
X = vcat(Xs...);
dX = vcat(dXs...);
ts_s = dfs[:4][:ts_s];
ts = dfs[:4][:ts];

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



# 15. visualize results
fig = Figure();
ax = Axis(fig[2,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false, xticks=(x_tick_pos, x_tick_strings), yticklabelsize=15,)
ax2 = Axis(fig[3,1], ylabel="v₂", xticksvisible=false, xticklabelsvisible=false, xticks=(x_tick_pos, x_tick_strings), yticklabelsize=15,)
ax3 = Axis(fig[4,1], ylabel="v₃", xticks=(x_tick_pos, x_tick_strings), xticklabelrotation=π/3, yticklabelsize=15, xticklabelsize=15)

linkxaxes!(ax, ax2, ax3)

ls = []
for i ∈ 1:length(Xs)
    l1 = lines!(ax, ts_x[i], Xs[i][:,1], linewidth=2, color=mints_colors[1])
    l2 = lines!(ax, ts_x[i], X̂s[i][:,1], linewidth=2, alpha=0.75, color=mints_colors[2])


    l1 = lines!(ax2, ts_x[i], Xs[i][:,2], linewidth=2, color=mints_colors[1])
    l2 = lines!(ax2, ts_x[i], X̂s[i][:,2], linewidth=2, alpha=0.75, color=mints_colors[2])


    l1 = lines!(ax3, ts_x[i], Xs[i][:,3], linewidth=2, color=mints_colors[1])
    l2 = lines!(ax3, ts_x[i], X̂s[i][:,3], linewidth=2, alpha=0.75, color=mints_colors[2])

    if i == 1
        push!(ls, l1)
        push!(ls, l2)
    end
end

#axislegend(ax, ls, ["Embedding", "Fit"])
fig[1,1] = Legend(fig, ls, ["Embedding", "Fit"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

# xlims!(ax3, ts_full[1], ts_full[end])
fig

save(joinpath(fig_savepath, "reconstructed-embedding-coords.png"), fig)

xlims!(ax3, ts_s[1][1], ts_s[2][end])
fig

save(joinpath(fig_savepath, "reconstructed-embedding-coords__zoomed.png"), fig)



# Plot statistics of forcing function
fig = Figure();
ax = Axis(fig[2,1], yscale=log10, xlabel="vᵣ");


forcing_pdf = kde(X[:, r_cutoff])
idxs_nozero = forcing_pdf.density .> 0
gauss = fit(Normal, X[:, r_cutoff])

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3, color=(mints_colors[1], 1.0))
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3, color=(mints_colors[2], 1.0))



for i ∈ (r_cutoff+1):r
    forcing_pdf = kde(X[:, i])
    idxs_nozero = forcing_pdf.density .> 0
    gauss = fit(Normal, X[:, i])

    lines!(ax, gauss, linestyle=:dash, linewidth=1.5, color=(mints_colors[1],0.35))
    lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=1.5, color=(mints_colors[2],0.35))
end

# ylims!(10^(-0.5), 10^3)
xlims!(-0.01, 0.01)
fig[1,1] = Legend(fig, [l1, l2], ["Gaussian Fit", "Actual PDF"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
fig

save(joinpath(fig_savepath, "forcing-statistics.pdf"), fig)





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



fig = Figure();
ax = Axis(fig[2,1], ylabel="PM 2.5 (μg⋅m⁻³)", xticks=(x_tick_pos, x_tick_strings), xticklabelrotation=π/3);
# tscale = 1/(60*60)
tscale = 1

ls = []
for i ∈ 1:length(Ĥs)
    l1 = lines!(ax, ts_x[i] .* tscale, Zs[i][n_embedding+2:end-3], linewidth=3, color=mints_colors[1])
    l2 = lines!(ax, ts_x[i] .* tscale, Ĥs[i][1,:], linewidth=3, color=mints_colors[2])
    if i == 1
        push!(ls, l1)
        push!(ls, l2)
    end
end


# axislegend(ax, ls, ["Original time series", "HAVOK model"])
fig[1,1] = Legend(fig, ls, ["Original time series", "HAVOK model"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

fig

save(joinpath(fig_savepath, "havok-predictions.pdf"), fig)




fig = Figure();
ax = Axis(fig[2,1], ylabel="PM 2.5 (μg⋅m⁻³)", xlabel="time (days)");

tscale = 1/(60*60*24)

idx_plot = 1

l1 = lines!(ax, ts_x[idx_plot] .* tscale, Zs[idx_plot][n_embedding+2:end-3], linewidth=3, color=mints_colors[1])
l2 = lines!(ax, ts_x[idx_plot] .* tscale, Ĥs[idx_plot][1,:], linewidth=3, color=mints_colors[2])

fig[1,1] = Legend(fig, [l1, l2], ["Original time series", "HAVOK model"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

xlims!(ax, ts_x[idx_plot][1]*tscale, ts_x[idx_plot][1]*tscale + 2)

fig

save(joinpath(fig_savepath, "havok-predictions-zoomed.pdf"), fig)
save(joinpath(fig_savepath, "havok-predictions-zoomed.png"), fig)
