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


datapath = "./data/processed"
datapath_lorenz = joinpath(datapath, "lorenz", "data.csv")
@assert ispath(datapath_lorenz)


# load data

df = CSV.read(datapath_lorenz, DataFrame);
z = df.x
t = df.t
dt = t[2] - t[1]


# split single df into multiple so we can combining disparate data

idxs = 1:10_000:length(z)
idxs = [idxs[i-1]:idxs[i] for i ∈ 2:length(idxs)]
push!(idxs, idxs[end][end]:length(z))


Zs = [z[idxs[i]] for i ∈ 1:2:length(idxs)]
ts = [t[idxs[i]] for i ∈ 1:2:length(idxs)]


# test out joining Hankel Matrices from disparate time series
let
    z₁ = 1:10
    z₂ = 21:30

    H₁ = TimeDelayEmbedding(z₁; nrow=4)
    H₂ = TimeDelayEmbedding(z₂; nrow=4)

    hcat(H₁, H₂)  # we want to horizontally concatenate them
end


n_embedding = 100
n_derivative = 5
r_cutoff = 15
n_control = 1
r = r_cutoff + n_control - 1

# generate individual Hankel matrices
Hs = [TimeDelayEmbedding(Z, nrow=n_embedding) for Z ∈ Zs];
ts_s = [t[n_embedding:end] for t ∈ ts]
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

size(X)
size(dX)


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

colsize!(gl, 2, Relative(1/r)) # scale control column to correct size
#cb = Colorbar(fig[1,3], limits = extrema(Ξ), colormap=:inferno)
cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
fig

# visualize the eigenmodes
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


# set up interpolation
ts_full = vcat(ts_s...)
itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_full) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


ts_x = [t[3:end-3] for t ∈ ts_s]
ts_x_full = vcat(ts_x...)

xr = vcat(u.(ts_x_full)...)
size(xr)
size(X)


function f!(dx, x, (A,B), t)
    dx .= A*x + B*u(t)
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

    push!(X̂s, X̂)
    push!(X̂s_test, X̂test)
end


fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="v₁", title="HAVOK Fit for v₁")

ls = []
for i ∈ 1:length(Xs)
    l1 = lines!(ax, ts_x[i], Xs[i][:,1], linewidth=2, color=mints_colors[1])
    l2 = lines!(ax, ts_x[i], X̂s[i][:,1], linewidth=2, linestyle=:dot, color=mints_colors[2])

    if i == 1
        push!(ls, l1)
        push!(ls, l2)
    end
end

axislegend(ax, [l1, l2], ["Embedding", "Fit"])
xlims!(ax, 0, 50)

fig

# yay! it looks like it works!



