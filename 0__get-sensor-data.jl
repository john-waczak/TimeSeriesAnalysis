using ProgressMeter
using Dates, TimeZones, TimeZoneFinder
using CSV, DataFrames, JSON
using Statistics, LinearAlgebra
using HTTP


include("./utils.jl")
creds = JSON.parsefile("./credentials.json")
token = creds["token"]
org = creds["orgname"]
bucket = creds["bucket"]
url = "http://mdash.circ.utdallas.edu:8086/api/v2/query?org=$(org)"

headers = Dict(
    "Authorization" => "Token $(token)",
    "Accept" => "application/csv",
    "Content-type" => "application/vnd.flux",
)


out_path = "./data/raw/"
if !ispath(out_path)
    mkpath(out_path)
end

d_start = DateTime(2024, 1, 1, 0, 0, 0)
d_end = DateTime(2024, 8, 12, 0, 0, 0)
node = "vaLo Node 01"


# get the Lat/Lon for the sensor location
d= DateTime(2024, 7, 1, 0, 0, 0)
dstart = d - Minute(25)
dend = d + Day(1) + Minute(25)
query_gps = """
from(bucket: "SharedAirDFW")
  |> range(start: time(v: "$(d)Z"), stop: time(v: "$(d + Day(1))Z"))
  |> filter(fn: (r) => r["device_name"] == "$(node)")
  |> filter(fn: (r) => r["_measurement"] == "GPSGPGGA2")
  |> filter(fn: (r) => r["_field"] == "latitudeCoordinate" or r["_field"] == "longitudeCoordinate" or r["_field"] == "altitude")
  |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time", "latitudeCoordinate", "longitudeCoordinate", "altitude",])
"""

resp = HTTP.post(url, headers=headers, body=query_gps)
df = CSV.read(IOBuffer(String(resp.body)), DataFrame)
df = df[:, Not([:Column1, :result, :table, :_time])]
dropmissing!(df)

# get position of sensor
pos = (;lat=mean(df.latitudeCoordinate), lon=mean(df.longitudeCoordinate), alt=mean(df.altitude))

# get the time zone:
tzone = timezone_at(pos.lat, pos.lon)

# save to file in case we need it later
summary = Dict(
    "node" => node,
    "time_zone" => tzone,
    "latitude" => pos.lat,
    "longitude" => pos.lon,
)

nname = replace(node, " " => "-")
open("./data/$(nname)-summary.json", "w") do f
    JSON.print(f, summary)
end



d = ZonedDateTime(d, tzone)
dstart = d - Minute(25)
dend = d + Day(1) + Minute(25)


function process_df(resp, tzone, d)
    df = CSV.read(IOBuffer(String(resp.body)), DataFrame)

    df = df[:, Not([:Column1, :result, :table])]
    rename!(df, :_time => :datetime)
    df.datetime = parse_datetime.(df.datetime, tzone)
    idx_keep = [Date(di) == Date(d) for di in df.datetime]
    df = df[idx_keep, :]

    dropmissing!(df)

    return df
end



@showprogress for d ∈ d_start:Day(1):d_end
    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
    if !ispath(csv_path)
        mkpath(csv_path)
    end


    out_ips = joinpath(csv_path, "$(Date(d))_ips.csv")
    out_bme = joinpath(csv_path, "$(Date(d))_bme.csv")
    out_rg15 = joinpath(csv_path, "$(Date(d))_rg15.csv")
    out_scd = joinpath(csv_path, "$(Date(d))_scd.csv")

    d = ZonedDateTime(d, tzone)
    dstart = d - Minute(25)
    dend = d + Day(1) + Minute(25)

    query_ips = """
    from(bucket: "SharedAirDFW")
      |> range(start: time(v: "$(dstart)"), stop: time(v: "$(dend)"))
      |> filter(fn: (r) => r["device_name"] == "$(node)")
      |> filter(fn: (r) => r["_measurement"] == "IPS7100")
      |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
      |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0"])
    """

    query_bme = """
    from(bucket: "SharedAirDFW")
      |> range(start: time(v: "$(dstart)"), stop: time(v: "$(dend)"))
      |> filter(fn: (r) => r["device_name"] == "$(node)")
      |> filter(fn: (r) => r["_measurement"] == "BME280V2")
      |> filter(fn: (r) => r["_field"] == "temperature" or r["_field"] == "pressure" or r["_field"] == "humidity" or r["_field"] == "dewPoint")
      |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "temperature", "pressure", "humidity", "dewPoint"])
    """

    query_rg15 = """
    from(bucket: "SharedAirDFW")
      |> range(start: time(v: "$(dstart)"), stop: time(v: "$(dend)"))
      |> filter(fn: (r) => r["device_name"] == "$(node)")
      |> filter(fn: (r) => r["_measurement"] == "RG15")
      |> filter(fn: (r) => r["_field"] == "rainPerInterval" or r["_field"] == "accumulation")
      |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "rainPerInterval"])
    """

    query_scd30 = """
    from(bucket: "SharedAirDFW")
      |> range(start: time(v: "$(dstart)"), stop: time(v: "$(dend)"))
      |> filter(fn: (r) => r["device_name"] == "$(node)")
      |> filter(fn: (r) => r["_measurement"] == "SCD30V2")
      |> filter(fn: (r) => r["_field"] == "co2" or r["_field"] == "temperature")
      |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
      |> keep(columns: ["_time", "co2"])
    """

    try
        # get IPS data
        resp = HTTP.post(url, headers=headers, body=query_ips)
        df_ips = process_df(resp, tzone, d)
        CSV.write(out_ips, df_ips)

    catch e
        println("$(node) failed to ingest IPS on $(d)")
        println(e)
    end

    try
        # get BME data
        resp = HTTP.post(url, headers=headers, body=query_bme)
        df_bme = process_df(resp, tzone, d)
        CSV.write(out_bme, df_bme)
    catch e
        println("$(node) failed to ingest BME on $(d)")
        println(e)
    end

    try
        # get RG15 data
        resp = HTTP.post(url, headers=headers, body=query_rg15)
        df_rg = process_df(resp, tzone, d)
        CSV.write(out_rg15, df_rg)
    catch e
        println("$(node) failed to ingest RG15 on $(d)")
        println(e)
    end

    try
        # get SCD data
        resp = HTTP.post(url, headers=headers, body=query_scd30)
        df_scd = process_df(resp, tzone, d)
        CSV.write(out_scd, df_scd)
    catch e
        println("$(node) failed to ingest SCD on $(d)")
        println(e)
    end
end




# loop over each node and concatentate all datasets
println("Working on $(node)")

csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))

df_ips = []
df_bme = []
df_rg =  []
df_scd = []
for (root, dirs, files) ∈ walkdir(csv_path)
    for file ∈ files
        if endswith(file, "ips.csv")
            df = CSV.read(joinpath(root, file), DataFrame)
            df.datetime .= ZonedDateTime.(String.(df.datetime))
            push!(df_ips, df)
        elseif endswith(file, "bme.csv")
            df = CSV.read(joinpath(root, file), DataFrame)
            df.datetime .= ZonedDateTime.(String.(df.datetime))
            push!(df_bme, df)
        elseif endswith(file, "rg15.csv")
            df = CSV.read(joinpath(root, file), DataFrame)
            df.datetime .= ZonedDateTime.(String.(df.datetime))
            push!(df_rg, df)
        elseif endswith(file, "scd.csv")
            df = CSV.read(joinpath(root, file), DataFrame)
            df.datetime .= ZonedDateTime.(String.(df.datetime))
            push!(df_scd, df)
        else
            continue
        end
    end
end

df_ips = vcat(df_ips...);
df_bme = vcat(df_bme...);
df_rg = vcat(df_rg...);
df_scd = vcat(df_scd...);


# only care about data where we have IPS values
df = df_ips

# generate group index to identify breaks in continuity
df.dt = [Minute(q .- df.datetime[1]).value for q in df.datetime]
df.t_skip = vcat(0, Minute.(df.datetime[2:end] .- df.datetime[1:end-1]) .!= Minute(1))

idx_group = Int[]
i = 1
for j ∈ 1:nrow(df)
    if df.t_skip[j] == 1
        i += 1
    end
    push!(idx_group, i)
end
df.group = idx_group


df = leftjoin(df, df_bme, on=:datetime);
@assert nrow(df) == nrow(df_ips)

df = leftjoin(df, df_scd, on=:datetime);
@assert nrow(df) == nrow(df_ips)

df = leftjoin(df, df_rg, on=:datetime);
@assert nrow(df) == nrow(df_ips)

df_summary = combine(groupby(df, :group), nrow)

println("Ndays: ", maximum(df_summary.nrow) / (24*60))

save_path = "./data/processed/"
csv_path = joinpath(save_path, replace(lowercase(node), " " => "-"))

if !ispath(csv_path)
    mkpath(csv_path)
end

CSV.write(joinpath(csv_path, "df.csv"), df)
CSV.write(joinpath(csv_path, "df_summary.csv"), df_summary)



