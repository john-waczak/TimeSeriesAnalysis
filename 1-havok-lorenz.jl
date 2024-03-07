using CairoMakie
using MintsMakieRecipes

set_theme!(mints_theme)
update_theme!(
    figure_padding=30,
    Axis=(
        xticklabelsize=20,
        yticklabelsize=20,
        xlabelsize=22,
        ylabelsize=22,
        titlesize=25,
    ),
    Colorbar=(
        ticklabelsize=20,
        labelsize=22
    )
)

include("utils.jl")
include("viz.jl")

using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity


# create sample "time-series" to see the embedding
z = 1:10

TimeDelayEmbedding(z; nrow=4)

# Result:
# 1  2  3  4  5  6   7
# 2  3  4  5  6  7   8
# 3  4  5  6  7  8   9
# 4  5  6  7  8  9  10

TimeDelayEmbedding(z; nrow=4, method=:backward)
# Result:
# 4  5  6  7  8  9  10
# 3  4  5  6  7  8   9
# 2  3  4  5  6  7   8
# 1  2  3  4  5  6   7


datapath = "./data/processed"
@assert ispath(datapath)
figpath = "./figures"
@assert ispath(figpath)


datapath_lorenz = joinpath(datapath, "lorenz", "data.csv")
figpath_lorenz = joinpath(figpath, "lorenz")
if !ispath(figpath_lorenz)
    mkpath(figpath_lorenz)
end

datapaths_cn = [
    joinpath(datapath, "central-nodes", "central-hub-4"),
    joinpath(datapath, "central-nodes", "central-hub-7"),
    joinpath(datapath, "central-nodes", "central-hub-10"),
]

@assert all(ispath.(datapaths_cn))

# ------------------------------------------------------------------------------------------------
# Standard HAVOK for Lorenz System
# ------------------------------------------------------------------------------------------------

# 0. load data
df = CSV.read(datapath_lorenz, DataFrame)

# visualize time-series
fig = Figure();
ga = fig[1,1] = GridLayout();
ax1 = Axis(ga[1,1], ylabel="x", xticklabelsvisible=false, xticksvisible=false)
ax2 = Axis(ga[2,1], ylabel="y", xticklabelsvisible=false, xticksvisible=false)
ax3 = Axis(ga[3,1], xlabel="time", ylabel="z")

linkxaxes!(ax1,ax2,ax3)

lx = lines!(ax1, df.t, df.x, color=mints_colors[1])
ly = lines!(ax2, df.t, df.y, color=mints_colors[2])
lz = lines!(ax3, df.t, df.z, color=mints_colors[3])

xlims!(ax3, df.t[1], df.t[end])

rowgap!(ga, 5)

fig

save(joinpath(figpath_lorenz, "xyz-time-series.png"), fig)
# save(joinpath(figpath_lorenz, "xyz-time-series.pdf"), fig)


# convert data to matrix
Data = Matrix(df)

# construct Hankel Matrix
nrow = 100
H = TimeDelayEmbedding(Data[:,2], nrow=100)

# compute singular value decomposition
U, σ, V = svd(H)

# visualize the attractor
dt = df.t[2] - df.t[1]
tspan = Data[:,1]
Nmax = 50000  # max value for plotting

fig = Figure(; resolution=(1200,700), figure_padding=100);
ax1 = Axis3(fig[1,1];
            xlabel="x",
            ylabel="y",
            zlabel="z",
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

ax2 = Axis3(fig[1,2];
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

# hidedecorations!(ax1);
# hidedecorations!(ax2);

L = 1:Nmax
l1 = lines!(ax1, Data[L,2], Data[L,3], Data[L,4], color=Data[L,1], colormap=:inferno, linewidth=3)
l2 = lines!(ax2, V[L,1], V[L,2], V[L,3], color=tspan[L], colormap=:inferno, linewidth=3)

fig

save(joinpath(figpath_lorenz, "original-vs-svd-attractor.png"), fig)


# fix the cutoff value to 15 as in original HAVOK paper
r = 15
n_control = 1
r = r+n_control - 1

# truncate matrices to cutoff value
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]

# compute derivatives with fourth order finite difference scheme
dVr = zeros(size(Vr,1)-5, r-n_control)
Threads.@threads for k ∈ 1:r-n_control
    for i ∈ 3:size(Vr,1)-3
        @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
    end
end

@assert size(dVr,2) == r-n_control

# chop off edges so size of data matches size of derivative
X = @view Vr[3:end-3, :]
dX = @view dVr[:,:]
ts = range(3*dt, step=dt, length=size(dVr,1))


# parition into training/testing points
n_test_points = 10000
Xtest = X[end-n_test_points+1:end, :]
dXtest = dX[end-n_test_points+1:end, :]

X = X[1:end-n_test_points,:]
dX = dX[1:end-n_test_points,:]

L = 1:size(X,1)
Ltest = size(X,1)+1:size(X,1)+n_test_points
@assert size(X,2) == size(dX,2)  + n_control

# We want to find Ξ such that ΞX' = dX' (since X and dX have records as rows)
# therefore we use \ to solve XΞ' = dX for Ξ' (since Au=b ⟹ u=A\b)
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B


# visualize matrices of learned operators
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

colsize!(gl, 2, Relative(1/r)) # scale control column to correct size
#cb = Colorbar(fig[1,3], limits = extrema(Ξ), colormap=:inferno)
cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
fig

save(joinpath(figpath_lorenz, "operator-heatmap.png"), fig)
# save(joinpath(figpath_lorenz, "operator-heatmap.pdf"), fig)


# visualize eigenmodes
fig = Figure();
ax = Axis(fig[1,1], title="Eigenmodes");


ls1 = []
ls2 = []
lr = []
# p = plot([], yticks=[-0.3, 0.0, 0.3], legend=:outerright, label="")

for i ∈ 1:r
    if i ≤ 3
        l = lines!(ax, 1:100, Ur[:,i], color=mints_colors[1], linewidth=3)
        push!(ls1, l)
    elseif i > 3 && i < r
        l = lines!(ax, 1:100, Ur[:,i], color=:grey, alpha=0.5, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:100, Ur[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

axislegend(ax, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋅⋅⋅", "uᵣ"])

fig

save(joinpath(figpath_lorenz, "svd-eigenmodes.png"), fig)
# save(joinpath(figpath_lorenz, "svd-eigenmodes.pdf"), fig)


# define interpolation function for forcing coordinate
# fit these on the full Vr matrix so we get all the times for predictions on test points
# starting at 3 due to derivative
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# visualize first embedding coordinate we are modelling
# together with the forcing function we've learned
xr = vcat(u.(ts[L])...)

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts[1:Nmax], X[1:Nmax,1], linewidth=3)
#l2 = lines!(ax2, ts[1:Nmax], map(x->x[1]^2, xs[1:Nmax]), linewidth=3, color=mints_colors[2])
l2 = lines!(ax2, ts[1:Nmax], xr[1:Nmax].^2, linewidth=3, color=mints_colors[2])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts[1], ts[Nmax])

fig

save(joinpath(figpath_lorenz, "v1_with_forcing.png"), fig)
# save(joinpath(figpath_lorenz, "v1_with_forcing.pdf"), fig)


# define function and integrate to get model predictions

function f!(dx, x, (A,B), t)
    dx .= A*x + B*u(t)
end

ps = (A, B)
x₀ = X[1,1:r-n_control]
dx = copy(x₀)
@assert size(x₀) == size(dx)

prob = ODEProblem(f!, x₀, (ts[1], ts[end]), ps)
sol = solve(prob, saveat=ts);

X̂ = Matrix(sol[:,L])'
X̂test = Matrix(sol[:,Ltest])'

# visualize results
fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="v₁", title="HAVOK Fit for v₁")

l1 = lines!(ax, tspan[L], X[:,1], linewidth=2)
l2 = lines!(ax, ts[L], X̂[:,1], linewidth=2, linestyle=:dot)

axislegend(ax, [l1, l2], ["Embedding", "Fit"])
xlims!(ax, 0, 50)

fig

save(joinpath(figpath_lorenz, "v1-reconstruction.png"), fig)
save(joinpath(figpath_lorenz, "v1_reconstruction.png"), fig)



# visualize the fitted attractor:
fig = Figure();
ax = Axis3(fig[1,1];
           xlabel="v₁",
           ylabel="v₂",
           zlabel="v₃",
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
           title="Reconstructed Attractor"
           );
l1 = lines!(ax, X̂[:,1], X̂[:,2], X̂[:,3], linewidth=3, color=ts[L], colormap=:plasma)
fig

save(joinpath(figpath_lorenz, "havok-attractor.png"), fig)


# 17. scatter plot and quantile quantile of fit

fig = scatter_results(
    X[:,1],
    X̂[:,1],
    Xtest[:,1],
    X̂test[:,1],
    "v₁"
)
fig

save(joinpath(figpath_lorenz, "scatterplot.png"), fig)
# save(joinpath(figpath_lorenz, "scatterplot.pdf"), fig)

fig = quantile_results(
    X[:,1],
    X̂[:,1],
    Xtest[:,1],
    X̂test[:,1],
    "v₁"
)
fig

save(joinpath(figpath_lorenz, "quantile-quantile.png"), fig)
# save(joinpath(figpath_lorenz, "quantile-quantile.pdf"), fig)




# Statistics of forcing function
forcing_pdf = kde(X[:, r - n_control + 1] .- mean(X[:, r - n_control + 1]), npoints=256)
idxs_nozero = forcing_pdf.density .> 0

# create gaussian using standard deviation of
gauss = Normal(0.0, std(X[:, r-n_control+1]))

fig = Figure();
ax = Axis(fig[1,1], yscale=log10, xlabel="vᵣ", title="Forcing Statistics");
l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3)
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3)
ylims!(1e-1, 1e3)
xlims!(-0.02, 0.02)
axislegend(ax, [l1, l2], ["Gaussian", "Estiamted PDF"])

fig

save(joinpath(figpath_lorenz, "forcing-statistics.png"), fig)
# save(joinpath(figpath_lorenz, "forcing-statistics.pdf"), fig)





# Compute indices where forcing is active
thresh = 4.0e-6
inds = X[:, r-n_control+1] .^ 2 .> thresh
Δmax = 500

idx_start = []
idx_end = []

start = 1
new_hit = 1

while !isnothing(new_hit)
    push!(idx_start, start)

    endmax = min(start + 500, size(X,1)) # 500 must be max window size for forcing

    interval = start:endmax
    hits = findall(inds[interval])
    endval = start + hits[end]

    push!(idx_end, endval)

    # now move to next hit:
    new_hit = findfirst(inds[endval+1:end])

    if !isnothing(new_hit)
        start = endval + new_hit
    end
end

# set up index dictionaries to make this easier
forcing_dict = Dict(
    :on => [idx_start[i]:idx_end[i] for i ∈ 2:length(idx_start)],
    :off => [idx_end[i]:idx_start[i+1] for i ∈ 2:length(idx_start)-1]
)



fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1];
          ylabel="v₁",
          xticksvisible=false,
          xticklabelsvisible=false
          );
ax2 = Axis(gl[2,1]; xlabel="time", ylabel="vᵣ");

linkxaxes!(ax, ax2);

# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    lines!(
        ax,
        ts[idxs],
        X[idxs,1],
        color=mints_colors[2],
        linewidth=1,
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax,
        ts[idxs],
        X[idxs,1],
        color=mints_colors[1],
        linewidth=1
    )
end

for idxs ∈ forcing_dict[:on]
    lines!(
        ax2,
        ts[idxs],
        xr[idxs],
        color=mints_colors[2],
        linewidth=1
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax2,
        ts[idxs],
        xr[idxs],
        color=mints_colors[1],
        linewidth = 1
    )
end

rowsize!(gl, 2, Relative(0.2))

fig

save(joinpath(figpath_lorenz, "v1-with-forcing.png"), fig)
# save(joinpath(figpath_lorenz, "v1-with-forcing.pdf"), fig)


# Color-code attractor by forcing
fig = Figure();
ax = Axis3(
    fig[1,1];
    xlabel="x",
    ylabel="y",
    zlabel="z",
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
    title="Attractor with Intermittent Forcing"
);

for idxs ∈ forcing_dict[:on]
    lines!(
        ax,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_colors[2],
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_colors[1],
    )
end

fig

save(joinpath(figpath_lorenz, "xyz-attractor-w-forcing.png"), fig)



# Plot time series predictions on test data
fig = Figure();
ax = Axis(fig[1,1]; xlabel="time", ylabel="v₁", title="Predictions on Test Set")

l1 = lines!(
    ax,
    tspan[Ltest],
    Xtest[:,1],
    linewidth=3
)

l2 = lines!(
    ax,
    ts[Ltest],
    X̂test[:,1],
    linestyle=:dot,
    linewidth=3
)

leg = Legend(fig[1,2], [l1, l2], ["embedding", "prediction"])

fig

save(joinpath(figpath_lorenz, "v1-test-points.png"), fig)
# save(joinpath(figpath_lorenz, "v1-test-points.pdf"), fig)


# 23. reconstruct original time-series

Ur*diagm(σr)*X'

size(Ur)
size(σr)
size(Vr)

all(isapprox.(Ur*diagm(σr)*Vr', H; rtol=0.0000001))

# reform predicted Hankel matrix
Ĥ = Ur*diagm(σr)*hcat(X̂, xr)'


println(size(Ĥ))


fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="x(t)", title="HAVOK Model for x(t)");

l1 = lines!(ax, Data[3:3+size(Ĥ,2)-1,1], Data[3:3+size(Ĥ,2)-1,2])
l2 = lines!(ax, Data[3:3+size(Ĥ,2)-1,1], Ĥ[1,:], linestyle=:dash)

xlims!(ax, 0, 10)

leg = Legend(fig[1,2], [l1, l2], ["Original", "HAVOK"])

fig

save(joinpath(figpath_lorenz, "havok-predictions-x.png"), fig)
# save(joinpath(figpath_lorenz, "havok-predictions-x.pdf"), fig)





