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
datapath_cn = datapaths_cn[f_to_use]
fig_savepath = joinpath(fig_savepaths[f_to_use], "single-df-9-day")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


# load in a csv and it's associated summary
df = CSV.read(joinpath(datapath_cn, "df-"*time_types[1]*".csv"), DataFrame);
df_summary = CSV.read(joinpath(datapath_cn, "df-"*time_types[1]*"_summary.csv"), DataFrame);

# parse datetime to correct type
df.datetime = String.(df.datetime);
df.datetime = parse.(ZonedDateTime, df.datetime);

println(Second(df.datetime[2] - df.datetime[1]))


t1 = Date(2023, 5, 29);
t2 = Date(2023, 6, 8);


idx_start = argmin([abs((Date(dt) - t1).value) for dt ∈ df.datetime])
idx_end = argmin([abs((Date(dt) - t2).value) for dt ∈ df.datetime])


df = df[idx_start:idx_end,:]

# create a single dataset interpolated to every second
col_to_use = :pm2_5
zs_df = df[:, col_to_use];
ts_df = [(dt_f .- df.datetime[1]).value ./ 1000 for dt_f ∈ df.datetime];

z_itp = LinearInterpolation(zs_df, ts_df)

ts = ts_df[1]:ts_df[end]
Zs = z_itp.(ts)



# Define parameters for HAVOK
n_embedding = 100
n_derivative = 5
r_cutoff = 18
n_control = 10
r = r_cutoff + n_control - 1


# construct Hankel Matrix
H = TimeDelayEmbedding(Zs; method=:backward)


# Decompose via SVD
println("computing SVD... this could take a while")
U, σ, V = svd(H)
@assert all(H .≈ U*Diagonal(σ)*V')  # verify that decomposition works


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

l = lines!(ax1, V[:,1], V[:,2], V[:,3], color=ts[n_embedding:end], colormap=:inferno, linewidth=3)
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


# truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


# 7. compute derivative using fourth order central difference scheme
dt = mean(ts[2:end] .- ts[1:end-1])
@assert dt == 1.0

dVr = zeros(size(Vr,1)-n_derivative, r-n_control)

Threads.@threads for k ∈ 1:r-n_control
    for i ∈ 3:size(Vr,1)-3
        @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
    end
end

@assert size(dVr,2) == r-n_control

# chop off edges to size of data matches size of derivative
X = @view Vr[3:end-3, :]
dX = @view dVr[:,:]
@assert size(dX,2) == size(X,2)  - n_control


# if we do it this way we will be off by n_embedding...
# ts = range(ts[3], step=dt, length=size(dVr,1))
ts = range(ts[n_embedding+2], step=dt, length=size(dVr,1))
println("max time: ", ts[end]./(60^2*24), " (days)")

tend1 = dt*(9*24*60*60)
tend2 = ts[end-3]


# pinch time values for entire range: train + test
L = 1:length(ts[ts .< tend1])
Ltest = L[end]+1:length(ts)
@assert length(L) + length(Ltest) == length(ts)

# partition test/train split
Xtest = X[Ltest, :]
dXtest = dX[Ltest, :]
X = X[L,:]
dX = dX[L,:]

# Compute model matrix via least squares
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B


# visualize model matrices
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
        l = lines!(ax, 1:100, Ur[:,i], color=mints_colors[1], linewidth=3)
        push!(ls1, l)
    elseif i > 3 && i < r
        l = lines!(ax, 1:100, Ur[:,i], color=:grey, alpha=0.2, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:100, Ur[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

axislegend(ax, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋮", "uᵣ₁"])

fig

save(joinpath(fig_savepath, "svd-eigenmodes.pdf"), fig)



# define interpolation function for forcing coordinate(s)
#   +3 because of offset from derivative...
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts; extrapolate=true) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# visualize first embedding coordinate + the forcing term
xs = zeros(length(ts[L]), n_control)
xs_test = zeros(length(ts[Ltest]), n_control)
for i ∈ axes(xs,1)
    xs[i,:] .= u(ts[i])
end

for i ∈ axes(xs_test,1)
    xs_test[i,:] .= u(ts[Ltest[i]])
end

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time (days)");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts[1:Nmax] ./ (60^2*24), X[1:Nmax,1], linewidth=3)
l2 = lines!(ax2, ts[1:Nmax] ./ (60^2*24), xs[1:Nmax,1].^2, linewidth=3, color=mints_colors[2])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts[1] ./ (60^2*24), ts[Nmax] ./ (60^2*24))
fig

save(joinpath(fig_savepath, "v1-with-forcing.pdf"), fig)


# define function and integrate to get model predictions
function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*u(t)
end

params = (A, B)
x₀ = X[1,1:r-n_control]
dx = copy(x₀)

prob = ODEProblem(f!, x₀, (ts[L[1]], ts[L[end]]), params);
sol = solve(prob, saveat=ts[L]);
size(sol)

# integrate separately for test set
x₀ = Xtest[1,1:r-n_control]
# integrate over all times so we have model predictions on holdout data
prob_test = ODEProblem(f!, x₀, (ts[Ltest[1]], ts[Ltest[end]]), params)
sol_test = solve(prob_test, saveat=ts[Ltest]);
size(sol_test)


# split up solutions
X̂ = Matrix(sol[:,L])'
X̂test = Matrix(sol_test)'



# 15. visualize results
Nmax=200000
fig = Figure();
ax = Axis(fig[1,1], ylabel="v₁")
ax2 = Axis(fig[2,1], ylabel="v₂")
ax3 = Axis(fig[3,1], xlabel="time [hours]", ylabel="v₃")

linkxaxes!(ax, ax2, ax3)

l1 = lines!(ax, ts[1:Nmax] ./ (60^2), X[1:Nmax,1], linewidth=2)
l2 = lines!(ax, ts[1:Nmax] ./ (60^2), X̂[1:Nmax,1], linewidth=2, alpha=0.75, color=mints_colors[2])

l1 = lines!(ax2, ts[1:Nmax] ./ (60^2), X[1:Nmax,2], linewidth=2)
l2 = lines!(ax2, ts[1:Nmax] ./ (60^2), X̂[1:Nmax,2], linewidth=2, alpha=0.75, color=mints_colors[2])

l1 = lines!(ax3, ts[1:Nmax] ./ (60^2), X[1:Nmax,3], linewidth=2)
l2 = lines!(ax3, ts[1:Nmax] ./ (60^2), X̂[1:Nmax,3], linewidth=2, alpha=0.75, color=mints_colors[2])

fig
axislegend(ax, [l1, l2], ["Embedding", "Fit"])
fig

save(joinpath(fig_savepath, "reconstructed-embedding-coords.png"), fig)

xlims!(ax3, 0, 3)
fig

save(joinpath(fig_savepath, "reconstructed-embedding-coords__zoomed.png"), fig)


# visualize fitted attractor for initial times while predictions are still good
Nmax = 20000
fig = Figure(;size=(1200, 700), figure_padding=50);
ax1 = Axis3(fig[1,1];
            xlabel="v₁",
            ylabel="v₂",
            zlabel="v₃",
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
ax2 = Axis3(fig[1,2];
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


# l1 = lines!(ax1, X[1:Nmax,1], X[1:Nmax,2], X[1:Nmax,3], linewidth=3, color=ts[L[1:Nmax]], colormap=:plasma)
# l2 = lines!(ax2, X̂[1:Nmax,1], X̂[1:Nmax,2], X̂[1:Nmax,3], linewidth=3, color=ts[L[1:Nmax]], colormap=:plasma)


l1 = lines!(ax1, X[1:Nmax,1], X[1:Nmax,2], X[1:Nmax,3], linewidth=3, color=ts[L[1:Nmax]], colormap=:inferno)
l2 = lines!(ax2, X̂[1:Nmax,1], X̂[1:Nmax,2], X̂[1:Nmax,3], linewidth=3, color=ts[L[1:Nmax]], colormap=:inferno)

fig

save(joinpath(fig_savepath, "fitted-attractor-embedding.png"), fig)




# Plot statistics of forcing function
forcing_pdf = kde(X[:, r-n_control + 1])
idxs_nozero = forcing_pdf.density .> 0
gauss = fit(Normal, X[:, r-n_control+1])

fig = Figure();
ax = Axis(fig[1,1], yscale=log10, xlabel="vᵣ", title="Forcing Statistics");

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3)
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3)
ylims!(1e-1, nothing)
xlims!(-0.01, 0.01)
axislegend(ax, [l1, l2], ["Gaussian Fit", "Actual PDF"])

fig

save(joinpath(fig_savepath, "forcing-statistics.pdf"), fig)



# reconstruct original time-series
all(isapprox.(U*diagm(σ)*V', H; rtol=0.1))

# reform predicted Hankel matrix
Ĥ = Ur*diagm(σr)*hcat(X̂, xs)'
Ĥtest = Ur*diagm(σr)*hcat(X̂test, xs_test)'

println(size(Ĥ))
tscale = 1/(60*60)

fig = Figure();
ax = Axis(fig[1,1], xlabel="time (hours)", ylabel="PM 2.5 (μg⋅m⁻³)", title="HAVOK Model for PM 2.5");

l1 = lines!(ax, ts[3:5002] .* tscale, Zs[3:5002])
l2 = lines!(ax, ts[3:5002] .* tscale, Ĥ[end,1:5000])

axislegend(ax, [l1, l2], ["Original time series", "HAVOK model"])

xlims!(ax, ts[3] * tscale, 0.5)

fig

save(joinpath(fig_savepath, "havok-predictions.pdf"), fig)



fig = Figure();
ax = Axis(fig[1,1], xlabel="time (hours)", ylabel="PM 2.5 (μg⋅m⁻³)", title="HAVOK Model for PM 2.5");

l1 = lines!(ax, ts[(3 + Ltest[1]):(5002 + Ltest[1])] .* tscale, Zs[(3 + Ltest[1]):(5002 + Ltest[1])])
l2 = lines!(ax, ts[(3 + Ltest[1]):(5002 + Ltest[1])] .* tscale, Ĥtest[end,1:5000])

axislegend(ax, [l1, l2], ["Original time series", "HAVOK model"])
xlims!(ax, ts[(3 + Ltest[1])]*tscale, ts[(5002 + Ltest[1])]*tscale)
fig

save(joinpath(fig_savepath, "havok-predictions-testset.pdf"), fig)
