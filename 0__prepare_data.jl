using Pkg
#Pkg.add(url="https://github.com/john-waczak/TimeSeriesTools.jl")
Pkg.add(url="https://github.com/john-waczak/TimeSeriesTools.jl.git")


using TimeSeriesTools
using CSV, DataFrames
using DifferentialEquations
using Dates, TimeZones

using DataInterpolations
#using Interpolations
using StaticArrays
using Statistics
using Plots

include("utils.jl")

# # Download Data from OSN

ips_links = [
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/01/MINTS_001e06373996_IPS7100_2023_06_01.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/02/MINTS_001e06373996_IPS7100_2023_06_02.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/03/MINTS_001e06373996_IPS7100_2023_06_03.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/04/MINTS_001e06373996_IPS7100_2023_06_04.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/05/MINTS_001e06373996_IPS7100_2023_06_05.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/06/MINTS_001e06373996_IPS7100_2023_06_06.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/07/MINTS_001e06373996_IPS7100_2023_06_07.csv",
]

bme_links = [
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/01/MINTS_001e06373996_BME680_2023_06_01.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/02/MINTS_001e06373996_BME680_2023_06_02.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/03/MINTS_001e06373996_BME680_2023_06_03.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/04/MINTS_001e06373996_BME680_2023_06_04.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/05/MINTS_001e06373996_BME680_2023_06_05.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/06/MINTS_001e06373996_BME680_2023_06_06.csv",
    "https://ncsa.osn.xsede.org/ees230012-bucket01/AirQualityNetwork/data/raw/Central_Hub_4/2023/06/07/MINTS_001e06373996_BME680_2023_06_07.csv",
]


# test_df = CSV.File(download(ips_links[1])) |> DataFrame

dfs = [CSV.File(download(url)) |> DataFrame for url ∈ ips_links];

println("---")
println("IPS7100")
println("---")

for i ∈ 1:length(dfs)
    dfs[i].dateTime .= date2datetime.(dfs[i].dateTime)
    println(dfs[i].dateTime[1], "\t", dfs[i].dateTime[end])
end

df_ips7100 = vcat(dfs...);


dfs = [CSV.File(download(url)) |> DataFrame for url ∈ bme_links];

println("---")
println("BME680")
println("---")

for i ∈ 1:length(dfs)
    dfs[i].dateTime .= date2datetime.(dfs[i].dateTime)
    println(dfs[i].dateTime[1], "\t", dfs[i].dateTime[end])
end

df_bme680 = vcat(dfs...);

# clean up the dfs list
dfs = nothing
GC.gc()

# find indices of rows in IP7100 between start and end time of BME680
idx₀ = findfirst((df_ips7100.dateTime .- df_bme680.dateTime[1]) .> Millisecond(0))
idx_end = findfirst((df_ips7100.dateTime .- df_bme680.dateTime[end]) .> Millisecond(0))

# chop IPS7100 to those values
names_to_drop = [n for n ∈ names(df_ips7100) if occursin("pc", n)]
df_ips7100 = df_ips7100[idx₀:idx_end, Not(names_to_drop)]
df_bme680 = df_bme680[:, Not(["gas"])]

# add new column which is time rounded to nearest seconds since start of IPS7100 data
t₀ = df_ips7100.dateTime[1]

df_ips7100.t = round.(Int, Dates.value.(df_ips7100.dateTime .- t₀) ./ 1000)
df_bme680.t = round.(Int, Dates.value.(df_bme680.dateTime .- t₀) ./ 1000)


maximum(df_ips7100.t[2:end] .- df_ips7100.t[1:end-1])
# so clearly we should impute to a regular grid

df_out = DataFrame()
df_out.t = df_ips7100.t[1]:df_ips7100.t[end]



for col_name ∈ names(df_ips7100)
    if col_name ∉ ["dateTime", "t"]
        f = LinearInterpolation(df_ips7100[:, col_name], df_ips7100.t)
        df_out[!, col_name] = f.(df_out.t)
    end
end

for col_name ∈ ["temperature", "pressure", "humidity"]
    f = CubicSpline(df_bme680[:, col_name], df_bme680.t)
    df_out[!, col_name] = f.(df_out.t)
end

names(df_out)


# plot(df_out.t ./ (60^2*24), df_out.pm2_5, label="")
# plot!(
#     twinx(),
#     df_out.t ./ (60^2 * 24),
#     df_out.humidity,
#     color=:red,
#     alpha=0.5,
#     lw=2,
#     label=""
# )
# #xlims!(2,2.4)


# save the file
if !ispath("data/sharedair")
    mkpath("data/sharedair")
end
CSV.write("data/sharedair/data.csv", df_out)


# generate simple Lorenz system data to match that used in paper's example

# set up parameters
σ=10.0
β=8/3
ρ=28.0

p = [σ, ρ, β]

u0 = [-8, 8, 27]
dt = 0.001
tspan = dt:dt:200

function lorenz!(du, u, p, t)
    x,y,z=u
    σ,ρ,β=p

    du[1] = dx = σ * (y - x)
    du[2] = dy = x * (ρ - z) - y
    du[3] = dz = x * y - β * z
end

prob = ODEProblem(lorenz!, u0, (tspan[1], tspan[end]), p)
sol = solve(prob, saveat=tspan, abstol=1e-12, reltol=1e-12);

df_out = DataFrame()
df_out.t = sol.t
df_out.x = sol[1,:]
df_out.y = sol[2,:]
df_out.z = sol[3,:]

if !ispath("data/lorenz")
    mkpath("data/lorenz")
end
CSV.write("data/lorenz/data.csv", df_out)


