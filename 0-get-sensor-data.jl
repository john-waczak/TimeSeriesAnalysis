using CondaPkg
CondaPkg.add("influxdb-client")
CondaPkg.add("pandas")
using PythonCall
using ProgressMeter
using Dates, TimeZones
using CSV, DataFrames, JSON
using Statistics, LinearAlgebra


include("./utils.jl")

creds = JSON.parsefile("./credentials.json")
influx_client = pyimport("influxdb_client")
pandas = pyimport("pandas")
client = influx_client.InfluxDBClient(url="http://mdash.circ.utdallas.edu:8086", token=creds["token"], org=creds["orgname"], bucket=creds["bucket"])
query_api = client.query_api()


out_path = "./data/raw/"
if !ispath(out_path)
    mkpath(out_path)
end


# d_start = DateTime(2021, 1, 1, 0, 0, 0)
# d_end = DateTime(2024, 7, 1, 0, 0, 0)

d_start = DateTime(2024, 1, 1, 0, 0, 0)
d_end = DateTime(2024, 7, 10, 0, 0, 0)



# nodes = [
#     "Central Hub 4",
#     "Central Hub 7",  <---- This one is Joppa
#     "Central Hub 10",
# ]

nodes = [
    "vaLo Node 01"
]



# Download data
for node ∈ nodes
    println("Working on $(node)")
    @showprogress for d ∈ d_start:Day(1):d_end
        csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
        if !ispath(csv_path)
            mkpath(csv_path)
        end
        out_ips = joinpath(csv_path, "$(Date(d))_ips.csv")
        out_bme = joinpath(csv_path, "$(Date(d))_bme.csv")
        out_rg15 = joinpath(csv_path, "$(Date(d))_rg15.csv")
        out_scd = joinpath(csv_path, "$(Date(d))_scd.csv")

        out_ips_std = joinpath(csv_path, "$(Date(d))_ips-std.csv")
        out_bme_std = joinpath(csv_path, "$(Date(d))_bme-std.csv")
        out_rg15_std = joinpath(csv_path, "$(Date(d))_rg15-std.csv")
        out_scd_std = joinpath(csv_path, "$(Date(d))_scd-std.csv")

        dstart = d - Minute(25)
        dend = d + Day(1) + Minute(25)

        query_ips = """
        from(bucket: "SharedAirDFW")
          |> range(start: time(v: "$(dstart)Z"), stop: time(v: "$(dend)Z"))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "IPS7100")
          |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
          |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0"])
        """

        query_bme = """
        from(bucket: "SharedAirDFW")
          |> range(start: time(v: "$(dstart)Z"), stop: time(v: "$(dend)Z"))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "BME280V2")
          |> filter(fn: (r) => r["_field"] == "temperature" or r["_field"] == "pressure" or r["_field"] == "humidity" or r["_field"] == "dewPoint")
          |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "temperature", "pressure", "humidity", "dewPoint"])
        """

        query_rg15 = """
        from(bucket: "SharedAirDFW")
          |> range(start: time(v: "$(dstart)Z"), stop: time(v: "$(dend)Z"))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "RG15")
          |> filter(fn: (r) => r["_field"] == "rainPerInterval" or r["_field"] == "accumulation")
          |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "rainPerInterval"])
        """

        query_scd30 = """
        from(bucket: "SharedAirDFW")
          |> range(start: time(v: "$(dstart)Z"), stop: time(v: "$(dend)Z"))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "SCD30V2")
          |> filter(fn: (r) => r["_field"] == "co2" or r["_field"] == "temperature")
          |> aggregateWindow(every: 1m, period: 10m, offset:-5m,  fn: mean, createEmpty: true)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "co2"])
        """

        try
            # get IPS data
            df_ips = query_api.query_data_frame(query_ips)
            df_ips = df_ips.drop(["result", "table"], axis=1)
            df_ips.to_csv(out_ips)
        catch e
            println("$(node) failed to ingest IPS on $(d)")
        end

        try
            # get BME data
            df_bme = query_api.query_data_frame(query_bme)
            df_bme = df_bme.drop(["result", "table"], axis=1)
            df_bme.to_csv(out_bme)
        catch e
            println("$(node) failed to ingest BME on $(d)")
        end

        try
            # get RG15 data
            df_rg = query_api.query_data_frame(query_rg15)
            df_rg = df_rg.drop(["result", "table"], axis=1)
            df_rg.to_csv(out_rg15)
        catch e
            println("$(node) failed to ingest RG15 on $(d)")
        end

        try
            # get SCD data
            df_scd = query_api.query_data_frame(query_scd30)
            df_scd = df_scd.drop(["result", "table"], axis=1)
            df_scd.to_csv(out_scd)
        catch e
            println("$(node) failed to ingest SCD on $(d)")
        end

            # join on _time column
            # df_out = pandas.merge(df_ips, df_bme, how="inner", on="_time")
            # df_out = pandas.merge(df_out, df_rg, how="inner", on="_time")
            # df_out = pandas.merge(df_out, df_scd, how="inner", on="_time")
            # df_out = pandas.merge(df_out, df_as, how="inner", on="_time")
    end
end









function process_df(df_path)
    #day = parse(Date, splitext(splitpath(df_path)[end])[1])
    day = parse(Date, split(splitpath(df_path)[end], "_")[1])

    df = CSV.read(df_path, DataFrame)
    df = df[:, Not(:Column1)]
    rename!(df, :_time => :datetime)
    df.datetime = parse_datetime.(df.datetime)

    # keep only relevant days since we padded the data
    idx_keep = [Date(di) == day for di ∈ df.datetime]
    df = df[idx_keep, :]

    dropmissing!(df)

    return df
end





# loop over each node and concatentate all datasets
for node ∈ nodes
    println("Working on $(node)")

    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))

    df_ips = []
    df_bme = []
    df_rg =  []
    df_scd = []
    for (root, dirs, files) ∈ walkdir(csv_path)
        for file ∈ files
            if endswith(file, "ips.csv")
                df = process_df(joinpath(root, file))
                push!(df_ips, df)
            elseif endswith(file, "bme.csv")
                df = process_df(joinpath(root, file))
                push!(df_bme, df)
            elseif endswith(file, "rg15.csv")
                df = process_df(joinpath(root, file))
                push!(df_rg, df)
            elseif endswith(file, "scd.csv")
                df = process_df(joinpath(root, file))
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
end
