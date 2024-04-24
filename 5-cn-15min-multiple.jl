# HAVOK Model for Single Central Node
# Using 15-min data of joined time
# series with gaps


# HAVOK Model for Single Central Node
# Using 15-min data of joined time
# series with gaps

# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/

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
t_to_use = 3
datapath_cn = datapaths_cn[f_to_use]
fig_savepath = joinpath(fig_savepaths[f_to_use], "df-15-min-no-interp")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end


# load in a csv and it's associated summary
df = CSV.read(joinpath(datapath_cn, "df-"*time_types[t_to_use]*".csv"), DataFrame);
df_summary = CSV.read(joinpath(datapath_cn, "df-"*time_types[t_to_use]*"_summary.csv"), DataFrame);


# parse datetime to correct type
# df.datetime = String.(df.datetime);
# df.datetime = parse.(ZonedDateTime, df.datetime);

dt = (df.datetime[2]-df.datetime[1]).value / 1000


println(Second(df.datetime[2] - df.datetime[1]))


# t1 = Date(2023, 5, 29);
# t2 = Date(2023, 6, 8);


# idx_start = argmin([abs((Date(dt) - t1).value) for dt ∈ df.datetime])
# idx_end = argmin([abs((Date(dt) - t2).value) for dt ∈ df.datetime])


# df = df[idx_start:idx_end,:]


# set up parameters for integration

# n_embedding = 100
# n_derivative = 5
# r_cutoff = 18
# n_control = 10
# r = r_cutoff + n_control - 1

n_embedding = 100
n_derivative = 5
r_cutoff = 15
n_control = 10
r = r_cutoff + n_control - 1


# create a single dataset interpolated to every second
col_to_use = :pm2_5


t_start = df.datetime[1]
t_end = df.datetime[end]

df.dt = [(Second(dt - t_start)).value for dt ∈ df.datetime]


Zs = []
ts = []

gdf = groupby(df, :group)

for df_g ∈ gdf
    if nrow(df_g) .≥ n_embedding + n_derivative
        push!(Zs, df_g[:, col_to_use])
        push!(ts, df_g.dt)
    end
end


# visualize the time-series
#x_tick= t_start:Day(1):t_end
x_tick= (round(t_start, Month)-Month(1)):Month(2):(round(t_end, Month) + Month(1))
x_tick_pos = [Second(d - t_start).value for d ∈ x_tick]
x_tick_strings = [Dates.format.(d, "mm/yy") for d ∈ x_tick]

fig = Figure();
ax = Axis(fig[1,1], xlabel="time", ylabel="PM 2.5 (μg⋅m⁻3)", xticks=(x_tick_pos, x_tick_strings), xticklabelrotation=π/3);

for i ∈ 1:length(Zs)
    lines!(ax, ts[i], Zs[i], linewidth=3, color=mints_colors[1])
end

fig

save(joinpath(fig_savepath, "original-timeseries.png"), fig)



Hs = [TimeDelayEmbedding(Z, nrow=n_embedding; method=:backward) for Z ∈ Zs];
ts_s = [t[n_embedding:end] for t ∈ ts];
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
    lines!(ax1, Vrs[i][:,1], Vrs[i][:,2], Vrs[i][:,3], color=ts[i][n_embedding:end], colormap=:inferno, linewidth=3)
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

length(ts_s[1])
size(Hs[1])
size(Vrs[1])


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

xlims!(ax3, 0, 3)
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

ylims!(10^0.6, 10^2.3)
xlims!(-0.01, 0.01)
# axislegend(ax, [l1, l2], ["Gaussian Fit", "Actual PDF"])
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
