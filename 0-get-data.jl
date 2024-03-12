using CondaPkg
CondaPkg.add("influxdb-client")
CondaPkg.add("pandas")
using PythonCall
using JSON
using ProgressMeter
using Dates, TimeZones
using CSV, DataFrames
using Statistics, LinearAlgebra
using DifferentialEquations

include("./utils.jl")


creds = JSON.parsefile("./credentials.json")
influx_client = pyimport("influxdb_client")
client = influx_client.InfluxDBClient(url="http://mdash.circ.utdallas.edu:8086", token=creds["token"], org=creds["orgname"], bucket=creds["bucket"])
query_api = client.query_api()


out_path = "./data/raw/central-nodes/"
if !ispath(out_path)
    mkpath(out_path)
end


d_start = Date(2021, 1, 1)
d_end = Date(2024, 1, 1)

nodes = [
    "Central Hub 4",
    "Central Hub 7",
    "Central Hub 10",
]


# Note: we previously used central hub data from 05/29/23 - 06/07/23 for Central Hub 4

for node ∈ nodes
    println("Working on $(node)")
    @showprogress for d ∈ d_start:d_end
        csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
        if !ispath(csv_path)
            mkpath(csv_path)
        end
        out_name = joinpath(csv_path, "$(d).csv")

        dend = d + Day(1)

#         query = """
# from(bucket: "SharedAirDFW")
#   |> range(start: $(d), stop: $(dend))
#   |> filter(fn: (r) => r["device_name"] == "$(node)")
#   |> filter(fn: (r) => r["_measurement"] == "IPS7100")
#   |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
#   |> aggregateWindow(every: 1m, fn: mean)
#   |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
#   |> keep(columns: ["_time", "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0"])
# """

        query = """
from(bucket: "SharedAirDFW")
  |> range(start: $(d), stop: $(dend))
  |> filter(fn: (r) => r["device_name"] == "$(node)")
  |> filter(fn: (r) => r["_measurement"] == "IPS7100")
  |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0"])
"""



        try
            df = query_api.query_data_frame(query)
            df = df.drop(["result", "table"], axis=1)
            df.to_csv(out_name)
        catch e
            println("Falure for $(node) on $(d)")
        end
    end
end





cols_to_keep = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0]


function process_df(df_path)
    # 1. load data
    df = CSV.read(df_path, DataFrame)
    df = df[:, Not(:Column1)]
    # 2. rename _time colum
    rename!(df, :_time => :datetime)
    # 3. parse datetime to ZonedDateTime
    df.datetime = parse_datetime.(df.datetime)

    # 4. round to nearest second
    df.datetime = round.(df.datetime, Second(1))

    # 5. average to 1 minute
    df.min = round.(df.datetime, Minute(1))
    gdf = groupby(df, :min)
    df = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df, :min=> :datetime);
    return df
end

for node ∈ nodes
    println("Working on $(node)")

    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))

    dfs = DataFrame[]
    for (root, dirs, files) ∈ walkdir(csv_path)
        @showprogress for file ∈ files
            if endswith(file, ".csv")
                df = process_df(joinpath(root, file))
                push!(dfs, df)
            end
        end
    end


    println("combining dataframes...")
    dfs = vcat(dfs...)
    dropmissing!(dfs)

    gdf = groupby(dfs, :datetime)
    df = combine(gdf, cols_to_keep .=> mean; renamecols = false)

    println("gouping by Δt jump...")

    dfs.dt = [Minute(q .- dfs.datetime[1]).value for q in dfs.datetime]
    dfs.t_skip = vcat(0, Minute.(dfs.datetime[2:end] .- dfs.datetime[1:end-1]) .!= Minute(1))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(dfs)
        if dfs.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    dfs.group = idx_group
    dfs_summary = combine(groupby(dfs, :group), nrow)


    save_path = "./data/processed/central-nodes/"
    if !ispath(save_path)
        mkpath(save_path)
    end

    csv_path = joinpath(save_path, replace(lowercase(node), " " => "-"))
    CSV.write(joinpath(csv_path, "df.csv"), dfs)
    CSV.write(joinpath(csv_path, "df-summary.csv"), dfs_summary)



    # do the same but for a 15-min average and a 1-hour average
end





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
# sol = solve(prob, DP5(), saveat=tspan, abstol=1e-12, reltol=1e-12);
sol = solve(prob; saveat=tspan, abstol=1e-12, reltol=1e-12);


df_out = DataFrame()
df_out.t = sol.t
df_out.x = sol[1,:]
df_out.y = sol[2,:]
df_out.z = sol[3,:]

if !ispath("./data/processed/lorenz")
    mkpath("./data/processed/lorenz")
end
CSV.write("data/processed/lorenz/data.csv", df_out)



