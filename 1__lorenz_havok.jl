using CSV, DataFrames
using Plots
using TimeSeriesTools
using ParameterHandling
using Dates, TimeZones
using Unitful
using Markdown, LaTeXStrings
using Statistics, StatsBase, Distributions, KernelDensity
using BenchmarkTools
using LinearAlgebra, StaticArrays
using DifferentialEquations
using DataInterpolations

include("plot_defaults.jl")
add_mints_theme()
theme(:mints)

include("plot_recipes.jl")
include("utils.jl")

if !ispath("figures/lorenz/havok")
    mkpath("figures/lorenz/havok")
end


# 0. load data
Data = Matrix(CSV.File("data/lorenz/data.csv") |> DataFrame)


# 1. compute time delay embedding
H = TimeDelayEmbedding(Data[:,2], method=:backward)

# 2. compute singular value decomposition
U, σ, V = svd(H)

size(V)

# 3. visualize attractor
dt = 0.001
tspan = range(dt, step=dt, length=size(Data,1))
Nmax = 50000  # max value for plotting

p1 = scatter(
    Data[1:Nmax,2], Data[1:Nmax,3], Data[1:Nmax,4],
    ms=2,
    msw=0,
    msa=0,
    marker_z = tspan[1:Nmax],
    frame=:semi,
    ticks=nothing,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    label="",
    cbar=false,
    margins=0*Plots.mm,
    background_color=:transparent,
    title="attractor"
)


p2 = scatter(
    V[1:Nmax,1], V[1:Nmax,2], V[1:Nmax,3],
    ms=2,
    msw=0,
    msa=0,
    marker_z = tspan[1:Nmax],
    frame=:semi,
    ticks=nothing,
    xlabel="v₁",
    ylabel="v₂",
    zlabel="v₃",
    label="",
    cbar=false,
    margins=0*Plots.mm,
    background_color=:transparent,
    title="embedded attractor"
)

plot(p1, p2)

savefig("figures/lorenz/havok/attractors1.png")
savefig("figures/lorenz/havok/attractors1.pdf")


# 5. set r value to 15 as in paper
r = 15
n_control = 1
r = r + n_control - 1

# 6. truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


# 7. compute derivatives with fourth order finite difference scheme
dVr = zeros(size(Vr,1)-5, r-n_control)
Threads.@threads for k ∈ 1:r-n_control
    for i ∈ 3:size(Vr,1)-3
        @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
    end
end

@assert size(dVr,2) == r-n_control


# 8. chop off edges so size of data matches size of derivative
X = @view Vr[3:end-3, :]
dX = @view dVr[:,:]

ts = range(3*dt, step=dt, length=size(dVr,1))

# chop off final 1000 points for prediction
n_test_points = 10000
Xtest = X[end-n_test_points+1:end, :]
dXtest = dX[end-n_test_points+1:end, :]

X = X[1:end-n_test_points,:]
dX = dX[1:end-n_test_points,:]

L = 1:size(X,1)
Ltest = size(X,1)+1:size(X,1)+n_test_points

@assert size(X,2) == size(dX,2)  + 1
size(X)
size(dX)



# 9. compute matrix such that dX = XΞ'
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view
# Ξ = dX' / X' equivalent version...

A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B

# 10. visualize matrices
p1 = heatmap(A, yflip=true, xlabel="A", ylabel="", showaxis=false,link=:y, cbar=false,clims=(minimum(A), maximum(A)), leftmargin=0*Plots.mm,rightmargin=0*Plots.mm)
p2 = heatmap(B, yflip=true, xlabel="B", ylabel="", showaxis=false, link=:y, clims=(minimum(A),maximum(A)), color=:inferno,rightmargin=10*Plots.mm, leftmargin=0*Plots.mm)
plot(p1, p2, layout = @layout([a{0.8w} b{0.2w} ]))

savefig("figures/lorenz/havok/heatmap.png")
savefig("figures/lorenz/havok/heatmap.pdf")

# 11. visualize eigenmodes
p = plot([], yticks=[-0.3, 0.0, 0.3], legend=:outerright, label="")
for i ∈ 1:r
    if i ≤ 3
        plot!(Ur[:,i], label=L"u_{%$i}", color=:blue)
    elseif i > 3 && i < r
        plot!(Ur[:,i], label=L"u_{%$i}", color=:grey, alpha=0.5)
    else
        plot!(Ur[:,i], label=L"u_{%$i}", color=:red)
    end
end

display(p)

savefig("figures/lorenz/havok/eigenmodes.png")
savefig("figures/lorenz/havok/eigenmodes.pdf")


# 12. define interpolation function for forcing coordinate
# fit these on the full Vr matrix so we get all the times for predictions on test points

itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]

# 13. visualize first embedding coordinate that we want to fit:
xs = u.(ts[L])

p1 = plot(
    ts[1:Nmax],
    X[1:Nmax,1],
    xlabel="",
    ylabel="v₁",
    label=""
)

p2 = plot(
    ts[1:Nmax],
    map(x->x[1]^2, xs[1:Nmax]),
    ylabel="vᵣ²",
    xlabel="time",
    label="",
    color=:red,
    link=:x,
#    grid=false,
#    minorgrid=false,
    ygrid=false,
    yminorgrid=false,
    xgrid=true,
    xminorgrid=true,
    yticks=[0.0],
    lw=1,
)

l = @layout [
    a{0.8h}
    b
]
plot(p1, p2, layout=l)

savefig("figures/lorenz/havok/v1_with_forcing.png")
savefig("figures/lorenz/havok/v1_with_forcing.pdf")


# 14. Generate our A and B matrices as static arrays
sA = @SMatrix[A[i,j] for i ∈ axes(A,1), j ∈ axes(A,2)]
sB = @SMatrix[B[i,j] for i ∈ axes(B,1), j ∈ axes(B,2)]

# define function and integrate to get model predictions
function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*u(t)
end

params = (sA, sB)
x₀ = X[1,1:r-n_control]
dx = copy(x₀)

@assert size(x₀) == size(dx)
@benchmark f!(dx, x₀, params, ts[1])

prob = ODEProblem(f!, x₀, (ts[1], ts[end]), params)
sol = solve(prob, saveat=ts);

size(sol)
X̂ = sol[:,L]'
X̂test = sol[:,Ltest]'

# 15. visualize results
p1 = plot(
    tspan[L],
    X[:,1],
    xlabel="time",
    ylabel="v₁",
    label="embedding",
    lw=2
)

plot!(
    ts[L],
    X̂[:,1],
    label="fit",
    ls=:dot,
    lw=2
)

xlims!(0, 50)
savefig("figures/lorenz/havok/timeseries_reconstructed.png")
savefig("figures/lorenz/havok/timeseries_reconstructed.pdf")



# 16. visualize the fitted attractor:
p1 = scatter(
    X̂[:,1], X̂[:,2], X̂[:,3],
    ms=2,
    msw=0,
    msa=0,
    marker_z = ts[L],
    frame=:none,
    ticks=nothing,
    xlabel="",
    ylabel="",
    zlabel="",
    label="",
    cbar=false,
    margins=0*Plots.mm,
    background_color=:transparent
)

savefig("figures/lorenz/havok/attractor_reconstructed.png")
savefig("figures/lorenz/havok/attractor_reconstructed.pdf")


# 17. scatter plot and quantile quantile of fit

p1 = scatterresult(
    X[:,1], X̂[:,1],
    Xtest[:,1], X̂test[:, 1],
    xlabel="True v₁",
    ylabel="Predicted v₁",
    plot_title="HAVOK Fit for v₁",
)

savefig("figures/lorenz/havok/scatterplot.png")
savefig("figures/lorenz/havok/scatterplot.pdf")

p1 = quantilequantile(
    X[:,1], X̂[:,1],
    Xtest[:,1], X̂test[:, 1],
    xlabel="True v₁",
    ylabel="Predicted v₁",
    title="HAVOK Fit for v₁",
)

savefig("figures/lorenz/havok/quantile-quantile.png")
savefig("figures/lorenz/havok/quantile-quantile.pdf")



# 18. Statistics of forcing function
forcing_pdf = kde(X[:, r-n_control + 1])
idxs_nozero = forcing_pdf.density .> 0
gauss = fit(Normal, X[:, r-n_control+1])

plot(gauss, label="gaussian fit", yaxis=:log, ls=:dash)
plot!(forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], label="pdf")
ylims!(1e-1, 1e3)
xlims!(-0.01, 0.01)
xlabel!("vᵣ")
title!("Forcing Statistics")

savefig("figures/lorenz/havok/forcing-stats.png")
savefig("figures/lorenz/havok/forcing-stats.pdf")


# 19. Compute indices where forcing is active
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

    # if endval + 1 ≥ size(X,1)
    #     break
    # end

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

# add the final indices since they have inds of 0
#push!(forcing_dict[:off], idx_end[end]:size(X,1))




# 20. visualize the lobe switching behavior
p1  = plot()
# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    plot!(
        p1,
        ts[idxs],
        X[idxs,1],
        xlabel="",
        ylabel="v₁",
        label="",
        color=mints_palette[2],
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    plot!(
        p1,
        ts[idxs],
        X[idxs,1],
        xlabel="",
        ylabel="v₁",
        label="",
        color=mints_palette[1],
    )
end

# do the same for the forcing
p2 = plot(
    link=:x,
    ygrid=false,
    yminorgrid=false,
    xgrid=true,
    xminorgrid=true,
    yticks=[0.0]
)

for idxs ∈ forcing_dict[:on]
    plot!(
        p2,
        ts[idxs],
        map(x->x[1], xs[idxs]),
        ylabel="v₁₅",
        xlabel="time",
        label="",
        color=mints_palette[2],
        lw=1
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    plot!(
        p2,
        ts[idxs],
        map(x->x[1], xs[idxs]),
        ylabel="v₁₅",
        xlabel="time",
        label="",
        color=mints_palette[1],
        lw = 1
    )
end

l = @layout [
    a{0.8h}
    b
]
plot(p1, p2, layout=l)
xlims!(0, 40)

savefig("figures/lorenz/havok/v1_forcing_identified.png")
savefig("figures/lorenz/havok/v1_forcing_identified.pdf")



# 21. Color-code attractor by forcing
p1 = plot(
    frame=:semi,
    ticks=nothing,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    cbar=false,
    margins=0*Plots.mm,
    background_color=:transparent,
    title="Attractor with Intermittent Forcing"

)

for idxs ∈ forcing_dict[:on]
    plot!(
        p1,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_palette[2],
        label="",
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    plot!(
        p1,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_palette[1],
        label="",
    )
end

display(p1)

savefig("figures/lorenz/havok/attractor_w_forcing.png")
savefig("figures/lorenz/havok/attractor_w_forcing.pdf")


# 22. Plot time series predictions on test data
p1 = plot(
    tspan[Ltest],
    Xtest[:,1],
    xlabel="time",
    ylabel="v₁",
    label="embedding",
    lw=2
)

plot!(
    ts[Ltest],
    X̂test[:,1],
    label="fit",
    ls=:dot,
    lw=2
)

title!("Predictions for Test Data")

savefig("figures/lorenz/havok/timeseries_test_points.png")
savefig("figures/lorenz/havok/timeseries_test_points.pdf")


