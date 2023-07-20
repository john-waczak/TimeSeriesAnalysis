using CSV, DataFrames
using Plots, StatsPlots
using TimeSeriesTools
using ParameterHandling
using Dates, TimeZones
using Unitful
using Markdown, LaTeXStrings
using Statistics, StatsBase, Distributions, KernelDensity
using BenchmarkTools
using LinearAlgebra
using DifferentialEquations
using DataInterpolations
using StaticArrays

include("plot_defaults.jl")
add_mints_theme()
theme(:mints)

include("plot_recipes.jl")
include("utils.jl")

if !ispath("figures/sharedair/havok")
    mkpath("figures/sharedair/havok")
end


# 0. Load in data and set window size (for ncol argument)
println("loading in the data...")

function get_data(path, cols_to_use)
    df = CSV.File("data/sharedair/data.csv") |> DataFrame
    ts = df.t
    return ts, Matrix(df[:, cols_to_use])
end

#ts, Data = get_data("data/sharedair/data.csv", [:pm1_0, :pm2_5, :pm10_0])
ts, Data = get_data("data/sharedair/data.csv", [:pm2_5]) #, :pm2_5, :pm10_0])

size(ts)
size(Data)

Nwindow = 81 # determined in step 1 by range of variogram fit

# 1. Generate Hankel matrix for each individual time series and concatenate together
println("computing time delay embedding Hankel-Takens matrix...")
H = vcat([TimeDelayEmbedding(Data[:,i]; nrow=Nwindow, method=:backward) for i∈axes(Data,2)]...)

size(H)  # so we have 10 columns of data with 81 times each giving 810 elements per embedding vector


# 2. Decompose via SVD
println("computing SVD... this could take a while")
U, σ, V = svd(H)

# U₁:Uₙ ---> vᵣ¹, .... vᵣⁿ ---> p(v̂ᵣ) ~N(μ, Σ)

# start with U₁Σ₁V₁' ---> r modes + n controls (forcing terms)

# dictionary of modes
# dictionary of forcing terms

# DMD for long time series...

println("U has dimensions ", size(U))
println("V has dimensions ", size(V))
println("we have $(length(σ)) singular values")

@assert all(H .≈ U*Diagonal(σ)*V')  # verify that decomposition works

# 3. visualize the attractor:
#Nmax = 200000
Nmax = 400000
println("max time: ", ts[Nmax]./(60^2*24), " (hours)")
plot_times =  collect(1:size(V,1)) ./ (60.0)^2
p1 = scatter(
    V[1:Nmax,1], V[1:Nmax,2], V[1:Nmax,3],
    ms=2,
    msw=0,
    msa=0,
    marker_z = plot_times[1:Nmax],
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

savefig("figures/sharedair/havok/attractor.png")
savefig("figures/sharedair/havok/attractor.pdf")

# 4. visualize singular values
plot(σ./sum(σ), xlabel=L"index, $i$", ylabel=L"\sigma_{i}/(\Sigma_{i}\sigma_{i})"*"\n", lw=3, label="")
savefig("figures/sharedair/havok/singular_values.png")
savefig("figures/sharedair/havok/singular_values.pdf")


# 5. compute cut-off value for singular values based on magnitude
r = r_cutoff(σ,ratio=0.005, rmax=30)

size(V)

# r = r_optimal_approx(σ, size(V,2), size(V,1))
r = 18  # maybe just right?

# add extra dimensions to r for > 1 control variable
n_control = 10

r = r + n_control - 1


# 6. truncate the matrices
Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]


# 7. compute derivative using fourth order central difference scheme
dt = mean(ts[2:end] .- ts[1:end-1])
@assert dt == 1.0

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

# let's try it out on 3 days worth of data with 1 day of data for testing
tend1 = dt*(3*24*60*60)
tend2 = dt*(4*24*60*60)

# pinch time values for entire range: train + test
ts = ts[ts .< tend2]
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
p1 = heatmap(A, yflip=true, xlabel="A", ylabel="", showaxis=false,link=:y, cbar=false,clims=(minimum(A), maximum(A)), leftmargin=0*Plots.mm,rightmargin=0*Plots.mm)
p2 = heatmap(B, yflip=true, xlabel="B", ylabel="", showaxis=false, link=:y, clims=(minimum(A),maximum(A)), color=:inferno,rightmargin=10*Plots.mm, leftmargin=0*Plots.mm)
plot(p1, p2, layout = @layout([a{0.75w} b{0.25w} ]))

savefig("figures/sharedair/havok/heatmap.png")
savefig("figures/sharedair/havok/heatmap.pdf")


# 11. visualize eigenmodes
p = plot([], yticks=[0.0,], legend=:outerright, label="")
for i ∈ 1:r
    if i ≤ 3
        plot!(Ur[:,i], label=L"u_{%$i}", color=:blue)
    elseif i > 3 && i ≤ r-n_control
        plot!(Ur[:,i], label="", color=:grey, alpha=0.5)
    elseif i == r - n_control + 1
        plot!(Ur[:,i], label=L"u_{r}", color=:red)
    else
        plot!(Ur[:,i], label="", color=:red)
    end
end

for i ∈ 1:Int(size(Ur, 1) / Nwindow) - 1
    vline!([Nwindow*i], color = :black, alpha=1, label="", lw=2)
end

display(p)

savefig("figures/sharedair/havok/eigenmodes.png")
savefig("figures/sharedair/havok/eigenmodes.pdf")



# 12. define interpolation function for forcing coordinate(s)
#     +3 because of offset from derivative...
itps = [DataInterpolations.LinearInterpolation(Vr[3:end-3,j], ts) for j ∈ r-n_control+1:r]
u(t) = [itp(t) for itp ∈ itps]


# 13. visualize first embedding coordinate + the forcing term
xs = u.(ts[L])
p1 = plot(
    ts[L] ./ (60^2),
    X[L,1],
    xlabel="",
    ylabel="v₁",
    label=""
)

p2  = plot()
for j ∈ 1:1
    plot!(
        ts[L] ./ (60^2),
        map(x->x[j]^2, xs),
        ylabel="vᵣ²",
        xlabel="time",
        label="",
        color=:red,
        link=:x,
#        grid=false,
        #        minorgrid=false,
        ygrid=false,
        yminorgrid=false,
        xgrid=true,
        xminorgrid=true,
        yticks=[0.0]
    )
end

l = @layout [
    a{0.8h}
    b
]
p = plot(p1, p2, layout=l)
display(p)

savefig("figures/sharedair/havok/v1_with_forcing.png")
savefig("figures/sharedair/havok/v1_with_forcing.pdf")



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
prob = ODEProblem(f!, x₀, (ts[1], ts[end]), params)
sol = solve(prob, saveat=ts)# , abstol=1e-12, reltol=1e-12);
size(sol)

# split up solutions
X̂ = sol[:,L]'
X̂test = sol[:,Ltest]'


# 15. visualize results
Nmax=50000
p1 = plot(
    ts[L[1:Nmax]] ./ (60^2),
    X[L[1:Nmax],1],
    xlabel="time",
    ylabel="v₁",
    label="embedding",
    lw=2
)

plot!(
    ts[L[1:Nmax]] ./ (60^2),
    X̂[L[1:Nmax],1],
    label="fit",
    # ls=:dot,
    alpha=0.5,
    lw=2
)

p2 = plot(
    ts[L[1:Nmax]] ./ (60^2),
    X[L[1:Nmax],2],
    xlabel="time",
    ylabel="v₁",
    label="embedding",
    lw=2
)

plot!(
    ts[L[1:Nmax]] ./ (60^2),
    X̂[L[1:Nmax],2],
    label="fit",
    # ls=:dot,
    alpha=0.5,
    lw=2
)

p3 = plot(
    ts[L[1:Nmax]] ./ (60^2),
    X[L[1:Nmax],3],
    xlabel="time",
    ylabel="v₁",
    label="embedding",
    lw=2
)

plot!(
    ts[L[1:Nmax]] ./ (60^2),
    X̂[L[1:Nmax],3],
    label="fit",
    # ls=:dot,
    alpha=0.5,
    lw=2
)


plot(p1, p2, p3, layout=(3,1), size=(1600, 1000), margin=5*Plots.mm,
     plot_title= "r=$(r-n_control+1), n_control=$(n_control)"
     )

println(r-n_control+1)
savefig("figures/sharedair/havok/timeseries_reconstructed__r-18__c-10.png")
savefig("figures/sharedair/havok/timeseries_reconstructed__r-18__c-10.pdf")


xlims!(0, 2.5)
savefig("figures/sharedair/havok/timeseries_reconstructed_zoomed-in.png")
savefig("figures/sharedair/havok/timeseries_reconstructed_zoomed-in.pdf")


# 16. visualize the fitted attractor:
Nmax = 15000
p1 = scatter(
    X̂[L[1:Nmax],1], X̂[L[1:Nmax],2], X̂[L[1:Nmax],3],
    ms=2,
    msw=0,
    msa=0,
    marker_z = ts[L[1:Nmax]],
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

savefig("figures/sharedair/havok/attractor_reconstructed.png")
savefig("figures/sharedair/havok/attractor_reconstructed.pdf")


# 17. scatter plot and quantile quantile of fit

# p1 = scatterresult(
#     X[:,1], X̂[:,1],
#     Xtest[:,1], X̂test[:, 1],
#     xlabel="True v₁",
#     ylabel="Predicted v₁",
#     plot_title="HAVOK Fit for v₁",
# )

p1 = scatterresult(
    X[L[1:Nmax],1], X̂[L[1:Nmax],1],
    X[L[Nmax+1001:Nmax+2000],1], X̂[L[Nmax+1001:Nmax+2000],1],
#    Xtest[:,1], X̂test[:, 1],
    xlabel="True v₁",
    ylabel="Predicted v₁",
    plot_title="HAVOK Fit for v₁",
)


savefig("figures/sharedair/havok/scatterplot.png")
savefig("figures/sharedair/havok/scatterplot.pdf")

# p1 = quantilequantile(
#     X[:,1], X̂[:,1],
#     Xtest[:,1], X̂test[:, 1],
#     xlabel="True v₁",
#     ylabel="Predicted v₁",
#     title="HAVOK Fit for v₁",
# )

# savefig("figures/sharedair/havok/quantile-quantile.png")
# savefig("figures/sharedair/havok/quantile-quantile.pdf")



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

savefig("figures/sharedair/havok/forcing-stats.png")
savefig("figures/sharedair/havok/forcing-stats.pdf")



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

p1  = plot()
# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    plot!(
        p1,
        ts[idxs] .* tscale,
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
        ts[idxs] .* tscale,
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
        ts[idxs] .* tscale,
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
        ts[idxs] .* tscale,
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

xlims!(0, 2)

savefig("figures/sharedair/havok/v1_forcing_identified.png")
savefig("figures/sharedair/havok/v1_forcing_identified.pdf")




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
    title="Attractor with Intermittent Forcing",
)

i = 1
for idxs ∈ forcing_dict[:on]
    if i == 1
        label="active forcing"
    else
        label=""
    end

    plot!(
        p1,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_palette[2],
        label=label,
        lw=1
    )
    i+=1
end

# add plots for linear times
i=1
for idxs ∈ forcing_dict[:off]
    if i == 1
        label="approximately linear"
    else
        label=""
    end

    plot!(
        p1,
        X[idxs,1], X[idxs,2], X[idxs,3],
        color=mints_palette[1],
        label=label,
        lw=1
    )
    i+=1
end

display(p1)

savefig("figures/sharedair/havok/attractor_w_forcing.png")
savefig("figures/sharedair/havok/attractor_w_forcing.pdf")




# 22. add thresholding to original timeseries data
p1  = plot()
# add plots for forcing times
for idxs ∈ forcing_dict[:on]
    plot!(
        p1,
        ts[idxs] .* tscale,
        Data[3 .+ idxs],
        xlabel="",
        ylabel="PM 2.5 [μg/m^3]",
        label="",
        color=mints_palette[2],
    )
end
# add plots for linear times
for idxs ∈ forcing_dict[:off]
    plot!(
        p1,
        ts[idxs] .* tscale,
        Data[3 .+ idxs],
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
        ts[idxs] .* tscale,
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
        ts[idxs] .* tscale,
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

xlims!(0, 0.4)

savefig("figures/sharedair/havok/v1_forcing_identified.png")
savefig("figures/sharedair/havok/v1_forcing_identified.pdf")


# NOTE: need to think about how to visualize thresholding when B is a matrix and u(t) is a vector
#       perhaps just use the norm ||u(t)||^2 for the value we want to threshold
#       would also be interesting to look at the phase of the correction (perhaps a polar plot?)
