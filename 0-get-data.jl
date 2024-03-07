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


creds = JSON.parsefile("./credentials.json")
influx_client = pyimport("influxdb_client")
client = influx_client.InfluxDBClient(url="http://mdash.circ.utdallas.edu:8086", token=creds["token"], org=creds["orgname"], bucket=creds["bucket"])
query_api = client.query_api()


out_path = "./data/raw/central-nodes/"
if !ispath(out_path)
    mkpath(out_path)
end


d_start = Date(2023, 1, 1)
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

        query = """
from(bucket: "SharedAirDFW")
  |> range(start: $(d), stop: $(dend))
  |> filter(fn: (r) => r["device_name"] == "$(node)")
  |> filter(fn: (r) => r["_measurement"] == "IPS7100")
  |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
  |> aggregateWindow(every: 1m, fn: mean)
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




# combine raw into single dataframe
dfs = Dict()

for node ∈ nodes
    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
    dfs[node] = DataFrame[]
    for (root, dirs, files) ∈ walkdir(csv_path)
        for file ∈ files
            if endswith(file, ".csv")
                df = CSV.read(joinpath(root, file), DataFrame)
                push!(dfs[node], df)
            end
        end
    end
end



for node ∈ keys(dfs)
    dfs[node] = vcat(dfs[node]...)
    dfs[node] = dfs[node][:, Not([:Column1])]
    rename!(dfs[node], :_time => :datetime)
    dfs[node].datetime = [ZonedDateTime(String(dt), "yyyy-mm-dd HH:MM:SSzzzz") for dt in dfs[node].datetime];
end


out_path = "./data/processed/central-nodes/"
if !ispath(out_path)
    mkpath(out_path)
end



for (node, df) in pairs(dfs)
    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
    if !ispath(csv_path)
        mkpath(csv_path)
    end

    cols_to_keep = [:pm0_1, :pm0_3, :pm0_5, :pm1_0, :pm2_5, :pm5_0, :pm10_0]
    dropmissing!(df)

    df.min = round.(df.datetime, Dates.Minute(1));
    df.five_min = round.(df.datetime, Dates.Minute(5));
    df.quarter_hour = round.(df.datetime, Dates.Minute(15));
    df.hour = round.(df.datetime, Dates.Hour);
    df.eight_hour = round.(df.datetime, Dates.Hour(8));
    df.day = round.(df.datetime, Dates.Day);


    # 1 minute data
    gdf = groupby(df, :min)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :min => :datetime)
    df_sub.dt = [Minute(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Minute.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Minute(1))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)


    CSV.write(joinpath(csv_path, "df-1-min.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-1-min_summary.csv"), df_sub_summary)



    # 5 minute data
    gdf = groupby(df, :five_min)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :five_min => :datetime)
    df_sub.dt = [Minute(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Minute.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Minute(5))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)


    CSV.write(joinpath(csv_path, "df-5-min.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-5-min_summary.csv"), df_sub_summary)


    # 15 minute data
    gdf = groupby(df, :quarter_hour)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :quarter_hour=> :datetime)
    df_sub.dt = [Minute(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Minute.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Minute(15))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)

    CSV.write(joinpath(csv_path, "df-15-min.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-15-min_summary.csv"), df_sub_summary)


    # 1 hour data
    gdf = groupby(df, :hour)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :hour => :datetime)
    df_sub.dt = [Hour(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Hour.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Hour(1))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)

    CSV.write(joinpath(csv_path, "df-hour.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-hour_summary.csv"), df_sub_summary)


    # 8 hour data
    gdf = groupby(df, :eight_hour)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :eight_hour => :datetime)
    df_sub.dt = [Hour(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Hour.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Hour(8))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)

    CSV.write(joinpath(csv_path, "df-eight-hour.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-eight-hour_summary.csv"), df_sub_summary)


    # 24 hour data
    gdf = groupby(df, :day)
    df_sub = combine(gdf, cols_to_keep .=> mean; renamecols = false)
    rename!(df_sub, :day => :datetime)
    df_sub.dt = [Day(q .- df_sub.datetime[1]).value for q in df_sub.datetime]
    df_sub.t_skip = vcat(0, Day.(df_sub.datetime[2:end] .- df_sub.datetime[1:end-1]) .!= Day(1))

    idx_group = Int[]
    i = 1
    for j ∈ 1:nrow(df_sub)
        if df_sub.t_skip[j] == 1
            i += 1
        end
        push!(idx_group, i)
    end
    df_sub.group = idx_group
    df_sub_summary = combine(groupby(df_sub, :group), nrow)

    CSV.write(joinpath(csv_path, "df-day.csv"), df_sub)
    CSV.write(joinpath(csv_path, "df-day_summary.csv"), df_sub_summary)
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


