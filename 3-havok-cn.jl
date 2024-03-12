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

# parse datetime to correct type
df.datetime = String.(df.datetime);
df.datetime = parse.(ZonedDateTime, df.datetime);



# compute embedding dimension to acheive at least 24 hour coverage
dt = (df.datetime[2] - df.datetime[1]).value / (1000*60*60*24)

# use a day's worth of data for the feature vector
n_embedding = min(ceil(Int,1/dt), 100)


# number of samples needed to compute derivative with fourth order scheme
n_derivative = 5


# select groups which have nrow ≥ n_embedding + n_derivative
idx_good = findall(df_summary.nrow .≥ n_embedding + n_derivative)
groups_to_use = df_summary.group[idx_good]

df = df[findall([g ∈ groups_to_use for g ∈ df.group]), :]

idxs_groups = []
for group ∈ groups_to_use
    idxs = findall(df.group .== group)
    push!(idxs_groups, (; group=group, idxs=idxs))
end

# get times for plotting
t_start = df.datetime[1]
ts = [(dt .- t_start).value ./(1000*60*60*24) for dt in df.datetime]  # time in days
col_to_use = :pm2_5
z = df[:, col_to_use]


# visualize the time-series
fig = Figure();
ax = Axis(fig[1,1], xlabel="t (days since $(Date(t_start)))", ylabel="PM 2.5 (μg⋅m⁻³)");
ls = []
for group ∈ idxs_groups
    l = lines!(ax, ts[group.idxs], z[group.idxs], linewidth=3, label="group =  $(group.group)")
    push!(ls, l)
end
# leg = Legend(fig[1,2], ls, ["group = $(g.group)" for g ∈ idxs_groups])

fig
xlims!(ax, 0, ts[end])
fig

save(joinpath(fig_savepath, "timeseries-grouped.png"), fig)


# Get Hankel Matrix of time-delay embeddings
Hs = [TimeDelayEmbedding(z[g.idxs], nrow=n_embedding) for g ∈ idxs_groups];
@assert length(Hs) == length(idxs_groups)


# set up row indices for the combined Hankel matrices so we can
# break them up again later when computing the derivatives...
idx_group_mat = []
i_start = 1
for i ∈ 1:length(Hs)
    idxs = i_start:i_start + size(Hs[i],2) - 1
    push!(idx_group_mat, (;group = idxs_groups[i].group, idxs=idxs))
    i_start += size(Hs[i],2)
end
idx_group_mat


# combine
H = hcat(Hs...);

# compute singular value decomposition
U, σ, V = svd(H);


# plot singular values
fig = Figure();
ax = Axis(fig[1,1], xlabel="i", ylabel="σᵢ")
vlines!(ax, [15], color=mints_colors[2], label="r = 15")
scatter!(ax, σ ./ maximum(σ),)
axislegend(ax)
fig
save(joinpath(fig_savepath, "singular-values.pdf"), fig)


# fix the cutoff value to 15 as in original HAVOK paper
r_cutoff = 18
n_control = 10
r = r_cutoff +n_control - 1



Vrs = [V[g.idxs,1:r] for g ∈ idx_group_mat];
trs = [ts[g.idxs] for g ∈ idx_group_mat];
Ur = @view U[:,1:r];
σr = @view σ[1:r];


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
    # l = lines!(ax, Vᵢ[:,1], Vᵢ[:,2], Vᵢ[:,3], linewidth=3)
    l1 = lines!(ax, Vᵢ[:,1], Vᵢ[:,2], Vᵢ[:,3], color=idx_group_mat[i].idxs, colormap=:inferno, linewidth=3)
end

save(joinpath(fig_savepath, "svd-attarctor.png"), fig)

# compute derivatives

Xs = []
dXs = []
ts_x = []

for i ∈ 1:length(Vrs)
    Vr = Vrs[i]
    dVr = zeros(size(Vr,1)-5, r-n_control)

    # fourth order
    Threads.@threads for k ∈ 1:r-n_control
        for i ∈ 3:size(Vr,1)-3
            @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
        end
    end

    push!(Xs, Vr[3:end-3,:])
    push!(ts_x, trs[i][3:end-3])
    push!(dXs, dVr)
end

# Form training/testing splits
for i ∈ 1:length(Xs)
    println(i, "\t", length(Xs[i]), "\t", size(Xs[i]))
end

# use the final collection for testing


X = vcat(Xs[1:end-1]...);
Xtest = Xs[end];
dX = vcat(dXs[1:end-1]...);
dXtest = dXs[end];


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

colsize!(gl, 2, Relative(1/r)) # scale control column to correct size
#cb = Colorbar(fig[1,3], limits = extrema(Ξ), colormap=:inferno)
cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
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




# define interpolation function for forcing coordinate


# fit these on the full Vr matrix so we get all the times for predictions on test points
V_full = vcat(Vrs...)
t_full = vcat(trs...)

itps = [DataInterpolations.LinearInterpolation(V_full[:,j], t_full) for j ∈ r-n_control+1:r]

# create function to get forcing term
u(t) = [itp(t) for itp ∈ itps]


# apply to each time interval and collect results
Xrs = [hcat(u(tx)...) for tx ∈ ts_x ];

@assert size(Xrs[1],1) == size(Xs[1],1)
@assert size(Xrs[1],1) == size(dXs[1],1)



fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time");
linkxaxes!(ax,ax2);

for i ∈ 1:length(Xs)
    l1 = lines!(ax, ts_x[i], Xs[i][:,1], linewidth=2, color=mints_colors[1])
    l1 = lines!(ax2, ts_x[i], Xrs[i][:,1] .^ 2, linewidth=2, color=mints_colors[2])
end

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts_x[1][1], ts_x[end][end]);
# ylims!(ax2, 0, 0.001);
fig

save(joinpath(fig_savepath, "v1_with_forcing.png"), fig)


# define function and integrate to get model predictions
function f!(dx, x, (A,B), t)
    dx .= A*x + B*u(t)
end

ps = (A, B)

X̂s = [zeros(size(X)) for X ∈ Xs]

for i ∈ 1:length(Xs)
    # get initial condition
    i = 2 
    x₀ = Xs[i][1,1:r-n_control]
    dx = copy(x₀)
    @assert size(x₀) == size(dx)

    prob = ODEProblem(f!, x₀, (ts_x[i][1], ts_x[i][end]), ps)
    sol = solve(prob, saveat=ts_x[i]);
    X̂ = Matrix(sol)'

    X̂s[i][:,1:r-n_control] .= X̂
    X̂s[i][:,r-n_control+1:end] .= Xrs[i][:,:]
end


# visualize results
fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="v₁", title="HAVOK Fit for v₁")

ls = []
for i ∈ 1:length(Xs)

    i = 2
    l1 = lines!(ax, ts_x[i], Xs[i][:,1], linewidth=2, color=mints_colors[1])
    l2 = lines!(ax, ts_x[i], X̂s[i][:,1], linewidth=2, color=mints_colors[2], linestyle=:dot)

    if i == 1
        push!(ls, l1)
        push!(ls, l2)
    end
end

Xs[1][:,1]

axislegend(ax, ls..., ["Embedding", "Fit"])
# xlims!(ax, 0, 50)

fig

fig = Figure();
i = 2
ax = Axis(fig[1,1]);
lines!(trs[2], Vrs[2][:,i])
fig


df.datetime[1]
df_summary

fig = Figure();
ax = Axis(fig[1,1]);
lines!(ax, df.pm2_5[idxs_groups[2].idxs])
fig


Minute(df.datetime[2] - df.datetime[1])
