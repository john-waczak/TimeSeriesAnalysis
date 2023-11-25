using CairoMakie
# using GLMakie
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
using Statistics, StatsBase, Distributions, KernelDensity
using BenchmarkTools
using LinearAlgebra, StaticArrays
using DifferentialEquations
using DataInterpolations
using StableRNGs


# set random seed for reproduciblity
rng = StableRNG(42)


include("utils.jl")
include("viz.jl")

if !ispath("figures/sharedair/havok")
    mkpath("figures/sharedair/havok")
end


# 0. Load in data and set window size (for ncol argument)
println("loading in the data...")

function get_data(path, cols_to_use)
    df = CSV.read("data/sharedair/data.csv", DataFrame)
    ts = df.t
    return ts, Matrix(df[:, cols_to_use])
end

#ts, Data = get_data("data/sharedair/data.csv", [:pm1_0, :pm2_5, :pm10_0])
ts, Data = get_data("data/sharedair/data.csv", [:pm2_5])
#ts, Data = get_data("data/sharedair/data.csv", [:pm10_0])

size(ts)
size(Data)


# 1. Generate Hankel matrix for each individual time series and concatenate together
println("computing time delay embedding Hankel-Takens matrix...")
# H = vcat([TimeDelayEmbedding(Data[:,i]; method=:backward) for i∈axes(Data,2)]...)
H = TimeDelayEmbedding(Data[:,1]; method=:backward)

size(H)  # so we have 10 columns of data with 81 times each giving 810 elements per embedding vector


# 2. Decompose via SVD
println("computing SVD... this could take a while")
U, σ, V = svd(H)

println("U has dimensions ", size(U))
println("V has dimensions ", size(V))
println("we have $(length(σ)) singular values")

@assert all(H .≈ U*Diagonal(σ)*V')  # verify that decomposition works

# 3. visualize the attractor:
Nmax = 200000
println("max time: ", ts[Nmax]./(60^2*24), " (days)")

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

Ls = [1:Nmax, Nmax:2*Nmax, 2*Nmax:3*Nmax, 3*Nmax:4*Nmax, 4*Nmax:size(V,1)]

# generate colors to use
cmap = cgrad(:inferno)
cs = cmap[ts./ts[end]]

for L ∈ Ls
    l = lines!(ax1, V[L,1], V[L,2], V[L,3], color=cs[L], linewidth=3)
end

fig

save("figures/sharedair/havok/attractor.png", fig)



# 4. visualize singular values
fig = Figure();
ax = Axis(fig[1,1]; xlabel="index", ylabel="Normalized singular value")
l = lines!(ax, σ./sum(σ), linewidth=3)
fig

save("figures/sharedair/havok/singular_values.png", fig)
save("figures/sharedair/havok/singular_values.pdf", fig)


# 5. compute cut-off value for singular values based on magnitude
r = r_cutoff(σ,ratio=0.005, rmax=30)

size(V)

# r = r_optimal_approx(σ, size(V,2), size(V,1))
r = 18  # maybe just right?

# add extra dimensions to r for > 1 control variable
# n_control = 1
# n_control = 5
n_control = 10

r = r + n_control - 1


# 6. truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


# 7. compute derivative using fourth order central difference scheme
dt = mean(ts[2:end] .- ts[1:end-1])
@assert dt == 1.0  # check that we have even time spacing

dVr = zeros(size(Vr,1)-5, r-n_control)

Threads.@threads for k ∈ 1:r-n_control
    for i ∈ 3:size(Vr,1)-3
        @inbounds dVr[i-2,k] = (1/(12*dt)) * (-Vr[i+2, k] + 8*Vr[i+1, k] - 8*Vr[i-1,k] + Vr[i-2,k])
    end
end

@assert size(dVr,2) == r-n_control


# 8. chop off edges to size of data matches size of derivative
X = @view Vr[3:end-3, :]
dX = @view dVr[:,:]
@assert size(dX,2) == size(X,2)  - n_control


ts = range(ts[3], step=dt, length=size(dVr,1))

println("max time: ", ts[end]./(60^2*24), " (days)")

# let's try it out on 8 days worth of data with 1 day of data for testing
# tend1 = dt*(3*24*60*60)
# tend2 = dt*(4*24*60*60)
tend1 = dt*(9*24*60*60)
tend2 = ts[end-3]


# pinch time values for entire range: train + test
#ts = ts[ts .< tend2]
L = 1:length(ts[ts .< tend1])
Ltest = L[end]+1:length(ts)

@assert length(L) + length(Ltest) == length(ts)

# chop things nicely
Xtest = X[Ltest, :]
dXtest = dX[Ltest, :]
X = X[L,:]
dX = dX[L,:]



# 9. Compute model matrix via least squares
Ξ = (X\dX)'  # now Ξx = dx for a single column vector view

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

colsize!(gl, 2, Relative(n_control/r)) # scale control column to correct size
#cb = Colorbar(fig[1,3], limits =(-60,60), colormap=:inferno)
cb = Colorbar(fig[1,3], limits=extrema(Ξ), colormap=:inferno)

fig

save("figures/sharedair/havok/heatmap.png", fig)
save("figures/sharedair/havok/heatmap.pdf", fig)


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
        l = lines!(ax, 1:100, Ur[:,i], color=:grey, alpha=0.2, linewidth=3)
        push!(ls2, l)
    else
        l = lines!(ax, 1:100, Ur[:,i], color=mints_colors[2], linewidth=3)
        push!(lr, l)
    end
end

axislegend(ax, [ls1..., ls2[1], lr[1]], ["u₁", "u₂", "u₃", "⋅⋅⋅", "uᵣ₁"])

fig

save("figures/sharedair/havok/eigenmodes.png", fig)
save("figures/sharedair/havok/eigenmodes.pdf", fig)



# 12. define interpolation function for forcing coordinate(s)
#     +3 because of offset from derivative...
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts; extrapolate=true) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# 13. visualize first embedding coordinate + the forcing term
xs = zeros(length(ts[L]), n_control)
xs_test = zeros(length(ts[Ltest]), n_control)
for i ∈ axes(xs,1)
    xs[i,:] .= u(ts[i])
end

for i ∈ axes(xs_test,1)
    xs_test[i,:] .= u(ts[Ltest[i]])
end




#xs = u.(ts[L])

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="v₁", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1], ylabel="vᵣ²", xlabel="time [days]");
linkxaxes!(ax,ax2);

l1 = lines!(ax, ts[1:Nmax] ./ (60^2*24), X[1:Nmax,1], linewidth=3)
#l2 = lines!(ax2, ts[1:Nmax], map(x->x[1]^2, xs[1:Nmax]), linewidth=3, color=mints_colors[2])
l2 = lines!(ax2, ts[1:Nmax] ./ (60^2*24), xs[1:Nmax,1].^2, linewidth=3, color=mints_colors[2])

rowsize!(gl, 2, Relative(0.2));
xlims!(ax2, ts[1] ./ (60^2*24), ts[Nmax] ./ (60^2*24))
fig

save("figures/sharedair/havok/v1_with_forcing.png", fig)
save("figures/sharedair/havok/v1_with_forcing.pdf", fig)


fig = Figure();
ax = Axis(fig[1,1], xlabel="time (hours)", ylabel="vᵣ(t)", title="Forcing Function");
lines!(ax, ts[1:1000] ./ (60^2), xs[1:1000, 1], linewidth=3)
fig

save("figures/sharedair/havok/forcing-time-series.png", fig)
save("figures/sharedair/havok/forcing-time-series.png", pdf)


# 14. Integrate model forward in time
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

# A*x₀ + sB*u(ts[1])

@assert size(x₀) == size(dx)
@benchmark f!(dx, x₀, params, ts[1])


# integrate over all times so we have model predictions on holdout data
prob = ODEProblem(f!, x₀, (ts[1], ts[L[end]]), params)
sol = solve(prob, saveat=ts)# , abstol=1e-12, reltol=1e-12);
size(sol)


# integrate separately for the test set
x₀ = Xtest[1,1:r-n_control]
# integrate over all times so we have model predictions on holdout data
prob_test = ODEProblem(f!, x₀, (ts[Ltest[1]], ts[end]), params)
sol_test = solve(prob_test, saveat=ts)# , abstol=1e-12, reltol=1e-12);
size(sol_test)




# split up solutions
X̂ = sol[:,L]'
#X̂test = sol[:,Ltest]'
X̂test = sol_test[:,:]'


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
# xlims!(ax3, 0, 1.2)

fig

save("figures/sharedair/havok/timeseries_reconstructed_embedding.png", fig)
save("figures/sharedair/havok/timeseries_reconstructed_embedding.pdf", fig)





# 16. visualize the fitted attractor:
Nmax = 20000
fig = Figure(;resolution=(1200, 700), figure_padding=100);
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

l1 = lines!(ax1, X[1:Nmax,1], X[1:Nmax,2], X[1:Nmax,3], linewidth=3, color=ts[1:Nmax], colormap=:plasma)
l2 = lines!(ax2, X̂[1:Nmax,1], X̂[1:Nmax,2], X̂[1:Nmax,3], linewidth=3, color=ts[1:Nmax], colormap=:plasma)
fig

save("figures/sharedair/havok/attractor_reconstructed_embedding.png")
#savefig("figures/sharedair/havok/attractor_reconstructed.pdf")


# 17. scatter plot and quantile quantile of fit
fig = scatter_results(X[1:100000,1],
                       X̂[1:100000,1],
                       X[100001:101000,1],
                       X̂[100001:101000,1],
                       "v₁"
                       )
save("figures/sharedair/havok/scatterplot_long-times-test.png", fig)
save("figures/sharedair/havok/scatterplot_long-times-test.pdf", fig)





fig = scatter_results(X[1:20000,1],
                      X̂[1:20000,1],
                      Xtest[1:5000,1],
                      X̂test[1:5000,1],
                      "v₁"
                      )

save("figures/sharedair/havok/scatterplot.png", fig)
save("figures/sharedair/havok/scatterplot.pdf", fig)


fig = quantile_results(X[1:5000,1],
                      X̂[1:5000,1],
                      X[5001:10000,1],
                      X̂[5001:10000,1],
                      "v₁"
                      )


save("figures/sharedair/havok/quantile-quantile.png", fig)
save("figures/sharedair/havok/quantile-quantile.pdf", fig)



# 18. Statistics of forcing function
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

save("figures/sharedair/havok/forcing-stats.png", fig)
save("figures/sharedair/havok/forcing-stats.pdf", fig)



# 19. Compute indices where forcing is active
#thresh = 4.0e-6
thresh = 2.0e-5

inds = X[:, r-n_control+1] .^ 2 .> thresh


median(X[:, r-n_control+1] .^ 2)
mean(X[:, r-n_control+1] .^ 2)
maximum(X[:, r-n_control+1] .^ 2)


Δmax = 10*60

idx_start = []
idx_end = []

start = 1
new_hit = 1

while !isnothing(new_hit)
    push!(idx_start, start)

    endmax = min(start + Δmax, size(X,1)) # 500 must be max window size for forcing

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

if ts[forcing_dict[:on][1][1]] > ts[1]
    push!(forcing_dict[:off], 1:forcing_dict[:on][1][1])
end



length(forcing_dict[:on])
length(forcing_dict[:off])


# 20. visualize the lobe switching behavior
tscale = 1/(24*60*60)

fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1];
          ylabel="v₁",
          xticksvisible=false,
          xticklabelsvisible=false
          );
ax2 = Axis(gl[2,1]; xlabel="time [days]", ylabel="vᵣ");

linkxaxes!(ax, ax2);

# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    lines!(
        ax,
        ts[idxs] .* tscale,
        X[idxs,1],
        color=mints_colors[2],
        linewidth=1,
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax,
        ts[idxs] .* tscale,
        X[idxs,1],
        color=mints_colors[1],
        linewidth=1
    )
end

for idxs ∈ forcing_dict[:on]
    lines!(
        ax2,
        ts[idxs] .* tscale,
        xs[idxs,1],
        color=mints_colors[2],
        linewidth=1
    )
end

# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax2,
        ts[idxs] .* tscale,
        xs[idxs,1],
        color=mints_colors[1],
        linewidth = 1
    )
end

rowsize!(gl, 2, Relative(0.2))

fig

xlims!(ax2, 0, 2)
ylims!(ax, -0.003, 0.0)
ylims!(ax2, -0.01, 0.01)
fig

save("figures/sharedair/havok/v1_forcing_identified.png", fig)
save("figures/sharedair/havok/v1_forcing_identified.pdf", fig)




# 21. Color-code attractor by forcing
fig = Figure();
ax = Axis3(
    fig[1,1];
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

save("figures/sharedair/havok/attractor_w_forcing.png", fig)
save("figures/sharedair/havok/attractor_w_forcing.pdf", fig)




# 22. add thresholding to original timeseries data
fig = Figure();
gl = fig[1:2,1] = GridLayout();
ax = Axis(gl[1,1], ylabel="PM 2.5 [μg/m³]", xticksvisible=false, xticklabelsvisible=false);
ax2 = Axis(gl[2,1]; xlabel="time [days]", ylabel="|vᵣ|²");

linkxaxes!(ax, ax2);

# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    lines!(
        ax,
        ts[idxs] .* tscale,
        Data[3 .+ idxs],
        color=mints_colors[2],
        linewidth=2
    )
end
# add plots for linear times

for idxs ∈ forcing_dict[:off]
    lines!(
        ax,
        ts[idxs] .* tscale,
        Data[3 .+ idxs],
        color=mints_colors[1],
        linewidth=2
    )
end

fig

for idxs ∈ forcing_dict[:on]
    lines!(
        ax2,
        ts[idxs] .* tscale,
        xs[idxs,1].^2,
        color=mints_colors[2],
        linewidth=2
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    lines!(
        ax2,
        ts[idxs] .* tscale,
        xs[idxs,1].^2,
        color=mints_colors[1],
        linewidth = 2
    )
end


rowsize!(gl, 2, Relative(0.2))

fig

save("figures/sharedair/havok/v1_forcing_identified.png", fig)
save("figures/sharedair/havok/v1_forcing_identified.pdf", fig)


# NOTE: need to think about how to visualize thresholding when B is a matrix and u(t) is a vector
#       perhaps just use the norm ||u(t)||^2 for the value we want to threshold
#       would also be interesting to look at the phase of the correction (perhaps a polar plot?)
#       ^ Not entirely sure what I meant here... maybe angle of the Bu(t) vector relative to v(t)?


# 23. reconstruct original time-series

Ur*diagm(σr)*X'

size(Ur)
size(σr)
size(Vr)

all(isapprox.(U*diagm(σ)*V', H; rtol=0.1))

# reform predicted Hankel matrix
Ĥ = Ur*diagm(σr)*hcat(X̂, xs)'
Ĥtest = Ur*diagm(σr)*hcat(X̂test, xs_test)'


println(size(Ĥ))
tscale = 1/(60*60)
#ts, Data = get_data("data/sharedair/data.csv", [:pm10_0])
ts, Data = get_data("data/sharedair/data.csv", [:pm2_5])

fig = Figure();
ax = Axis(fig[1,1], xlabel="time [hours]", ylabel="PM 2.5 [μg/m³]", title="HAVOK Model for PM 2.5");

l1 = lines!(ax, ts[3:5002] .* tscale, Data[3:5002])
l2 = lines!(ax, ts[3:5002] .* tscale, Ĥ[end,1:5000])

axislegend(ax, [l1, l2], ["Original time series", "HAVOK model"])

xlims!(ax, 0, 0.5)

fig

save("figures/sharedair/havok/havok_pred_x.png", fig)
save("figures/sharedair/havok/havok_pred_x.pdf", fig)




fig = Figure();
ax = Axis(fig[1,1], xlabel="time [hours]", ylabel="PM 2.5 [μg/m³]", title="HAVOK Model for PM 2.5");

l1 = lines!(ax, ts[(3 + Ltest[1]):(5002 + Ltest[1])] .* tscale, Data[(3 + Ltest[1]):(5002 + Ltest[1])])
l2 = lines!(ax, ts[(3 + Ltest[1]):(5002 + Ltest[1])] .* tscale, Ĥtest[end,1:5000])

axislegend(ax, [l1, l2], ["Original time series", "HAVOK model"])

fig

save("figures/sharedair/havok/havok_pred_x--test_data.png", fig)
save("figures/sharedair/havok/havok_pred_x--test_data.pdf", fig)



