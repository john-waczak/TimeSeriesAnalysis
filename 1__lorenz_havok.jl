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
xs = u.(ts[L])

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts[1:Nmax], X[1:Nmax,1], linewidth=3)
l2 = lines!(ax2, ts[1:Nmax], map(x->x[1]^2, xs[1:Nmax]), linewidth=3, color=mints_colors[2])

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
using Metrics: r2_score


fig = Figure();
ga = fig[1, 1] = GridLayout()
axtop = Axis(ga[1, 1];
             leftspinevisible = false,
             rightspinevisible = false,
             bottomspinevisible = false,
             topspinevisible = false,
             )
axmain = Axis(ga[2, 1], xlabel = "true v₁", ylabel = "predicted v₁")
axright = Axis(ga[2, 2];
               leftspinevisible = false,
               rightspinevisible = false,
               bottomspinevisible = false,
               topspinevisible = false,
               )

linkyaxes!(axmain, axright)
linkxaxes!(axmain, axtop)

minval, maxval = extrema([extrema(X[:,1])..., extrema(Xtest[:,1])..., extrema(X̂[:,1])..., extrema(X̂test[:,1])...])
δ_edge = 0.1*(maxval-minval)

l1 = lines!(axmain, [minval-δ_edge, maxval+δ_edge], [minval-δ_edge, maxval+δ_edge], color=:gray, linewidth=3)
s1 = scatter!(axmain, X[:,1], X̂[:,1], alpha=0.25)
s2 = scatter!(axmain, Xtest[:,1], X̂test[:,1], marker=:cross, alpha=0.25)

labels=[
    "Training R²=$(round(r2_score(X[:,1], X̂[:,1]), digits=3)) (n=$(length(X[:,1])))",
    "Testing   R²=$(round(r2_score(Xtest[:,1], X̂test[:,1]), digits=3)) (n=$(length(Xtest[:,1])))",
    "1:1"
]

# leg = Legend(ga[1, 2], [s1, s2, l1], labels)
leg = axislegend(axmain, [s1, s2, l1], labels; position=:lt)

density!(axtop, X[:,1], color=mints_colors[1])
density!(axtop, Xtest[:,1], color=(mints_colors[2], 0.66))

density!(axright, X̂[:,1], direction = :y, color=mints_colors[1])
density!(axright, X̂test[:,1], direction = :y, color=(mints_colors[2], 0.66))

hidedecorations!(axtop)
hidedecorations!(axright)
#leg.tellheight = true
rowsize!(ga, 1, Relative(0.1))
colsize!(ga, 2, Relative(0.1))

colgap!(ga, 0)
rowgap!(ga, 0)

xlims!(axmain, minval-δ_edge, maxval+δ_edge)
ylims!(axmain, minval-δ_edge, maxval+δ_edge)

fig

save("figures/lorenz/havok/scatterplot.png", fig)
save("figures/lorenz/havok/scatterplot.pdf", fig)



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


