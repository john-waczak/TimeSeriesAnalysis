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

using CSV, DataFrames
# using TimeSeriesTools
#using ParameterHandling
using Dates, TimeZones
#using Unitful
#using Markdown, LaTeXStrings
using Statistics, StatsBase, Distributions, KernelDensity
using BenchmarkTools
using LinearAlgebra, StaticArrays
using DifferentialEquations
using DataInterpolations

include("utils.jl")

if !ispath("figures/lorenz/havok")
    mkpath("figures/lorenz/havok")
end


# 0. load data
Data = Matrix(CSV.read("data/lorenz/data.csv", DataFrame))

H = TimeDelayEmbedding(Data[:,2]; method=:backward)

# 2. compute singular value decomposition
U, σ, V = svd(H)


size(V)

# 3. visualize attractor
dt = 0.001
tspan = range(dt, step=dt, length=size(Data,1))
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

save("figures/lorenz/havok/attractors1.png", fig)
# save("figures/lorenz/havok/attractors1.pdf", fig)  # file is too big


fig = Figure();
ax = Axis3(fig[1,1];
            xlabel="v₁",
            ylabel="v₂",
            zlabel="v₃",
            # aspect = :data,
            azimuth=34π/180,
            elevation=22π/180,
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
L = 1:170000;
l = lines!(ax, V[L,1], V[L,2], V[L,3]) #, color=tspan[L], colormap=:plasma)

fig



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

colsize!(gl, 2, Relative(1/15)) # scale control column to correct size
#cb = Colorbar(fig[1,3], limits = extrema(Ξ), colormap=:inferno)
cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
fig

save("figures/lorenz/havok/heatmap.png", fig)
save("figures/lorenz/havok/heatmap.pdf", fig)




# 11. visualize eigenmodes
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

save("figures/lorenz/havok/eigenmodes.png", fig)
save("figures/lorenz/havok/eigenmodes.pdf", fig)


# 12. define interpolation function for forcing coordinate
# fit these on the full Vr matrix so we get all the times for predictions on test points
# stat at 3 due to derivative
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# 13. visualize first embedding coordinate that we want to fit:
xs = vcat(u.(ts[L])...)

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts[1:Nmax], X[1:Nmax,1], linewidth=3)
#l2 = lines!(ax2, ts[1:Nmax], map(x->x[1]^2, xs[1:Nmax]), linewidth=3, color=mints_colors[2])
l2 = lines!(ax2, ts[1:Nmax], xs[1:Nmax].^2, linewidth=3, color=mints_colors[2])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts[1], ts[Nmax])
fig

save("figures/lorenz/havok/v1_with_forcing.png", fig)
save("figures/lorenz/havok/v1_with_forcing.pdf", fig)


# 14. Generate our A and B matrices as static (stack) arrays
sA = @SMatrix[A[i,j] for i ∈ axes(A,1), j ∈ axes(A,2)]
sB = @SMatrix[B[i,j] for i ∈ axes(B,1), j ∈ axes(B,2)]

# define function and integrate to get model predictions

function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*u(t)  # we can speed this up 
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
fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="v₁", title="HAVOK Fit for v₁")

l1 = lines!(ax, tspan[L], X[:,1], linewidth=2)
l2 = lines!(ax, ts[L], X̂[:,1], linewidth=2, linestyle=:dot)

axislegend(ax, [l1, l2], ["Embedding", "Fit"])
xlims!(ax, 0, 50)

fig

save("figures/lorenz/havok/timeseries_reconstructed.png", fig)
save("figures/lorenz/havok/timeseries_reconstructed.pdf", fig)



# 16. visualize the fitted attractor:
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

save("figures/lorenz/havok/attractor_reconstructed.png", fig)
# save("figures/lorenz/havok/attractor_reconstructed.pdf", fig) # too big


# 17. scatter plot and quantile quantile of fit
include("viz.jl")

fig = scatter_results(X[:,1],
                      X̂[:,1],
                      Xtest[:,1],
                      X̂test[:,1],
                      "v₁"
                      )

save("figures/lorenz/havok/scatterplot.png", fig)
save("figures/lorenz/havok/scatterplot.pdf", fig)

fig = quantile_results(
    X[:,1],
    X̂[:,1],
    Xtest[:,1],
    X̂test[:,1],
    "v₁"
)

save("figures/lorenz/havok/quantile-quantile.png", fig)
save("figures/lorenz/havok/quantile-quantile.pdf", fig)



# 18. Statistics of forcing function
forcing_pdf = kde(X[:, r-n_control + 1])
idxs_nozero = forcing_pdf.density .> 0
gauss = fit(Normal, X[:, r-n_control+1])

fig = Figure();
ax = Axis(fig[1,1], yscale=log10, xlabel="vᵣ", title="Forcing Statistics");

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3)
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3)
ylims!(1e-1, 1e3)
xlims!(-0.01, 0.01)
axislegend(ax, [l1, l2], ["Gaussian Fit", "Actual PDF"])

fig

save("figures/lorenz/havok/forcing-stats.png", fig)
save("figures/lorenz/havok/forcing-stats.pdf", fig)


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
        xs[idxs],
        color=mints_colors[2],
        linewidth=1
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax2,
        ts[idxs],
        xs[idxs],
        color=mints_colors[1],
        linewidth = 1
    )
end

rowsize!(gl, 2, Relative(0.2))

fig

save("figures/lorenz/havok/v1_forcing_identified.png", fig)
save("figures/lorenz/havok/v1_forcing_identified.pdf", fig)



# 21. Color-code attractor by forcing
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

save("figures/lorenz/havok/attractor_w_forcing.png", fig)
# savefig("figures/lorenz/havok/attractor_w_forcing.pdf")


# 22. Plot time series predictions on test data
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


save("figures/lorenz/havok/timeseries_test_points.png", fig)
save("figures/lorenz/havok/timeseries_test_points.pdf", fig)



# 23. reconstruct original time-series

Ur*diagm(σr)*X'

size(Ur)
size(σr)
size(Vr)

all(isapprox.(Ur*diagm(σr)*Vr', H; rtol=0.0000001))

# reform predicted Hankel matrix
Ĥ = Ur*diagm(σr)*hcat(X̂, xs)'


println(size(Ĥ))


fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="x(t)", title="HAVOK Model for x(t)");

l1 = lines!(ax, Data[3:3+size(Ĥ,2)-1,1], Data[3:3+size(Ĥ,2)-1,2])
l2 = lines!(ax, Data[3:3+size(Ĥ,2)-1,1], Ĥ[end,:], linestyle=:dash)

xlims!(ax, 0, 10)

leg = Legend(fig[1,2], [l1, l2], ["Original", "HAVOK"])

fig

save("figures/lorenz/havok/havok_pred_x.png", fig)
save("figures/lorenz/havok/havok_pred_x.pdf", fig)

