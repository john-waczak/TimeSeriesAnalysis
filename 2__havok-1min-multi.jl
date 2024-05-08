using CSV, DataFrames
using LinearAlgebra
using DataInterpolations
using DifferentialEquations
using Statistics, StatsBase, Distributions, KernelDensity
using Dates, TimeZones
using DataInterpolations
using ProgressMeter

using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")
include("./utils.jl")




# -----------------------------------------------------------------------
# 1. Load data and set up figure paths
# -----------------------------------------------------------------------


# load in data for central node
datapath = "./data/processed"
@assert ispath(datapath)
figpath = "./figures"
@assert ispath(figpath)

fig_savepath = joinpath(figpath, "2__havok-1min-multi")
if !ispath(fig_savepath)
    mkpath(fig_savepath)
end

c4_path = joinpath(datapath, "multi-sensor", "central-hub-4")
c7_path = joinpath(datapath, "multi-sensor", "central-hub-7")
c10_path = joinpath(datapath, "multi-sensor", "central-hub-10")


# load in CSVs and associated summaries
readdir(c4_path)

df4 = CSV.read(joinpath(c4_path, "df.csv"), DataFrame);
df4_summary = CSV.read(joinpath(c4_path, "df_summary.csv"), DataFrame);

df7 = CSV.read(joinpath(c7_path, "df.csv"), DataFrame);
df7_summary = CSV.read(joinpath(c7_path, "df_summary.csv"), DataFrame);

df10 = CSV.read(joinpath(c10_path, "df.csv"), DataFrame);
df10_summary = CSV.read(joinpath(c10_path, "df_summary.csv"), DataFrame);


extrema(df4_summary.nrow)   # (1, 720)
extrema(df7_summary.nrow)   # (1, 10606)   # <---- winner!
extrema(df10_summary.nrow)  # (1, 1439)



cols_to_use = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0, :temperature, :pressure, :humidity]
col_names = ["PM 0.1", "PM 0.3", "PM 0.5", "PM 1.0", "PM 2.5", "PM 5.0", "PM 10.0", "Temperature", "Pressure", "Relative Humidity"]
col_units = ["μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "μg/m³", "°C", "mbar", "%"]


# smooth with trialing 10 min average
trailing(Z, n) = [i < n ? mean(Z[1:i]) : mean(Z[i-n+1:i]) for i in 1:length(Z)]


# ----------------------------------------------------------------------
# Set up parameter values for HAVOK model
# ----------------------------------------------------------------------
method = :backward
n_embedding = 100
r_model = 5
n_control = 95
r = r_model + n_control


# ----------------------------------------------------------------------
# select same subset as in 1__havok-1min.jl
# ----------------------------------------------------------------------

idx_winner = argmax(df7_summary.nrow)
df = df7[df7.group .== idx_winner, :]
df.pressure .= 10 .* df.pressure

ts_single = df.dt .- df.dt[1]
dt_single = ts_single[2]-ts_single[1]
Zs_single = trailing(Array(df[:, :pm2_5]), 10)

# ----------------------------------------------------------------------
# form combined dataset to fit HAVOK model on full dataset
# ----------------------------------------------------------------------

Zs = []
ts = []

gdf = groupby(df7, :group)
for df_g ∈ gdf
    if nrow(df_g) .≥ (n_embedding + 20)
        push!(Zs, trailing(df_g[:, :pm2_5], 10))
        push!(ts, df_g.dt)
    end
end


Zs_joined = vcat(Zs...);
q_99 = quantile(Zs_joined, 0.99)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="PM 2.5 (μg/m³)", ylabel="Probability Distribution Function");
density!(
    ax,
    Zs_joined[Zs_joined .< q_99],
    strokewidth=3,
    strokecolor=mints_colors[3],
    color=(mints_colors[3], 0.5)
)
xlims!(ax, 0, nothing)
ylims!(ax, 0, nothing)

save(joinpath(fig_savepath, "1__joined-pdf.png"), fig)
save(joinpath(fig_savepath, "1__joined-pdf.pdf"), fig)

fig


fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="PM 2.5 (μg/m³)", ylabel="Cumulative Distribution Function");
ecdfplot!(
    ax,
    Zs_joined[Zs_joined .< q_99],
    strokewidth=3,
    color=mints_colors[3],
)

xlims!(ax, 0, nothing)
ylims!(ax, 0, nothing)

save(joinpath(fig_savepath, "2__joined-cdf.png"), fig)
save(joinpath(fig_savepath, "2__joined-cdf.pdf"), fig)

fig



# construct time delay embedding from joined dataset
Hs_joined = hcat([TimeDelayEmbedding(Z, n_embedding=n_embedding; method=:backward) for Z ∈ Zs]...);
ts_joined = [t[n_embedding:end] for t ∈ ts];


# generate indices of each component matrix so we can split
# after doing the SVD
idxs_H = []
i_now = 1
for i ∈ 1:length(ts_joined)
    tᵢ = ts_joined[i]
    push!(idxs_H, i_now:i_now + length(tᵢ) - 1)
    i_now += length(tᵢ)
end

idxs_H

# perform SVD of full data
U, σ, V = svd(Hs)

Vr = @view V[:,1:r]
Ur = @view U[:,1:r]
σr = @view σ[1:r]

# plot singular values
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="index", ylabel="Explained Variance");
lines!(ax, 1:length(σ), σ .^2 ./ sum(σr .^ 2), color=mints_colors[3], linewidth=3)
xlims!(ax, 0, 100)
ylims!(ax, -0.01, nothing)
save(joinpath(fig_savepath, "3__explained-variance.png"), fig)
save(joinpath(fig_savepath, "3__explained-variance.pdf"), fig)

fig


X = Vr
dX = zeros(size(Vr,1), r_model)

for j ∈ axes(dX, 2)
    for i ∈ 1:length(idxs_H)

        ts_i = ts_joined[i]
        idx_i = idxs_H[i]
        itp = CubicSpline(X[idx_i, j], ts_i)

        for k ∈ 1:length(ts_i)
            dX[idx_i[k], j] = DataInterpolations.derivative(itp, ts_i[k])
        end
    end
end


Ξ = (X\dX)'  # now Ξx = dx for a single column vector view
A = Ξ[:, 1:r-n_control]   # State matrix A
B = Ξ[:, r-n_control+1:end]      # Control matrix B


fig = Figure();
gl = fig[1,1] = GridLayout();
ax = Axis(
    gl[1,1];
    yreversed=true,
    title="A",
    xticklabelsvisible=false,
    yticklabelsvisible=false,
    xticksvisible=false,
    yticksvisible=false,
    aspect=DataAspect(),
);
cmap = cgrad(:inferno, 8, categorical=true)
heatmap!(ax, A, colormap=cmap, colorrange=(-0.1, 0.1))
cb = Colorbar(gl[1,2], colorrange=(-0.1, 0.1), colormap=cmap)
save(joinpath(fig_savepath, "4__A-heatmap.png"), fig)
save(joinpath(fig_savepath, "4__A-heatmap.pdf"), fig)
fig


# evaluate if global matrices are still effective on the dataframe from 1

Hsingle = TimeDelayEmbedding(Zs_single, n_embedding=n_embedding; method=:backward);
Vr_single = (inv(diagm(σr)) * Ur' * Hsingle)'

Zs_x = Zs_single[n_embedding:end]
dt_x = ts_single[2] - ts_single[1]
ts_x = range(ts_single[n_embedding], step=dt_x, length=length(Zs_x))

itps = [DataInterpolations.LinearInterpolation(Vr_single[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:r];
forcing(t) = [itp(t) for itp ∈ itps]

# define function and integrate to get model predictions
function f!(dx, x, p, t)
    A,B = p
    dx .= A*x + B*forcing(t)
end

ps = (A, B)
x₀ = Vr_single[1,1:r_model]
dx = copy(x₀)

f!(dx, x₀, ps, ts_x[1])

# define ODE problem and solve
prob = ODEProblem(f!, x₀, (ts_x[1], ts_x[end]), ps);
sol = solve(prob, saveat=ts_x);
X̂ = Array(sol)'
Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]

# compute rmse
rmse = sqrt(mean(Zs_x .- Ẑs_x) .^2)

fig = Figure();
ax = CairoMakie.Axis(fig[2,1], xlabel="time (days)", ylabel="PM 2.5 (μg/m³)",)
l_orig = lines!(ax, ts_x ./(24*60), Zs_x)
l_havok = lines!(ax, ts_x ./(24*60), Ẑs_x)
text!(fig.scene, 0.175, 0.915, text="RMSE = $(round(rmse, digits=3))", space=:relative)
xlims!(ax, ts_x[1]/(24*60), ts_x[end]/(24*60))
ylims!(ax, 0, nothing)
fig[1,1] = Legend(fig, [l_orig, l_havok], ["Original", "HAVOK"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)
save(joinpath(fig_savepath, "5__havok-reconstruction.png"), fig)
save(joinpath(fig_savepath, "5__havok-reconstruction.pdf"), fig)
fig




# --------------------------------------------------------------
# Visualize forcing statistics
# --------------------------------------------------------------


F = (B*Vr[:, r_model+1:end]')
Fsingle = (B*Vr_single[:, r_model+1:end]')
X_nn = Vr[:, 1:r_model]'


# visualize the forcing function and it's statistics
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="time (days)", ylabel="External Forcing");
ls = []
for i ∈ axes(F,1)
    l_i = lines!(ax, ts_x ./ (24*60), Fsingle[i,:], color=(mints_colors[2], 1-0.5*(i-1)/5))
end

save(joinpath(fig_savepath, "6__forcing-timeseries.png"), fig)
save(joinpath(fig_savepath, "6__forcing-timeseries.pdf"), fig)

fig


# Statistics of forcing function
forcing_pdf = kde(X[:,  - n_control + 1] .- mean(X[:, r - n_control + 1]), npoints=256)

# create gaussian using standard deviation of

fig = Figure();
ax = Axis(fig[2,1], yscale=log10, xlabel="vᵣ", ylabel="Forcing Statistics");

forcing_pdf = kde(Y[1,:] .- mean(Y[1,:]), npoints=2*2048)
idxs_nozero = forcing_pdf.density .> 0
gauss = Normal(0.0, std(Y[1,:]))

l1 = lines!(ax, gauss, linestyle=:dash, linewidth=3)
l2 = lines!(ax, forcing_pdf.x[idxs_nozero], forcing_pdf.density[idxs_nozero], linewidth=3)
ylims!(1e1, nothing)
xlims!(-0.00005, 0.00005)
fig[1,1] = Legend(fig, [l1, l2], ["Gaussian", "Estimated PDF"], framevisible=false, orientation=:horizontal, padding=(0,0,0,0), labelsize=17, height=-5)

save(joinpath(fig_savepath, "7__forcing-statistics.png"), fig)
save(joinpath(fig_savepath, "7__forcing-statistics.pdf"), fig)

fig





# --------------------------------------------------------------
# Construct NN to fit the forcing
# --------------------------------------------------------------

