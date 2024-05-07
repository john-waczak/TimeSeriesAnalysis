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


out_path = "./data/raw/multi-sensor/"
if !ispath(out_path)
    mkpath(out_path)
end


d_start = Date(2021, 1, 1)
d_end = Date(2024, 4, 18)

nodes = [
    "Central Hub 4",
    "Central Hub 7",
    "Central Hub 10",
]



# Download data
for node ∈ nodes
    println("Working on $(node)")
    @showprogress for d ∈ d_start:d_end
        csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))
        if !ispath(csv_path)
            mkpath(csv_path)
        end
        out_name = joinpath(csv_path, "$(d).csv")

        dend = d + Day(1)

        query_ips7100 = """
        from(bucket: "SharedAirDFW")
          |> range(start: $(d), stop: $(dend))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "IPS7100")
          |> filter(fn: (r) => r["_field"] == "pm0_1" or r["_field"] == "pm0_3" or r["_field"] == "pm0_5" or r["_field"] == "pm1_0" or r["_field"] == "pm2_5" or r["_field"] == "pm5_0"  or r["_field"] == "pm10_0")
          |> aggregateWindow(every: 1m, fn: mean)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0"])
        """

        query_bme680 = """
        from(bucket: "SharedAirDFW")
          |> range(start: $(d), stop: $(dend))
          |> filter(fn: (r) => r["device_name"] == "$(node)")
          |> filter(fn: (r) => r["_measurement"] == "BME680")
          |> filter(fn: (r) => r["_field"] == "temperature" or r["_field"] == "pressure" or r["_field"] == "humidity" or r["_field"] == "dewPoint")
          |> aggregateWindow(every: 1m, fn: mean)
          |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
          |> keep(columns: ["_time", "temperature", "pressure", "humidity"])
        """

        try
            # get IPS data
            df_ips7100 = query_api.query_data_frame(query_ips7100)
            df_ips7100 = df_ips7100.drop(["result", "table"], axis=1)

            # get BME data
            df_bme680 = query_api.query_data_frame(query_bme680)
            df_bme680 = df_bme680.drop(["result", "table"], axis=1)

            # join on _time column
            df_out = pandas.merge(df_ips7100, df_bme680, how="inner", on="_time")

            # save result
            df_out.to_csv(out_name)
        catch e
            println("Falure for $(node) on $(d)")
        end
    end
end




function process_df(df_path)
    df = CSV.read(df_path, DataFrame)
    df = df[:, Not(:Column1)]
    rename!(df, :_time => :datetime)
    df.datetime = parse_datetime.(df.datetime)

    return df
end


# df_path = "./data/raw/multi-sensor/central-hub-4/2024-01-01.csv"
# process_df(df_path)

# loop over each node and concatentate all datasets
for node ∈ nodes
    println("Working on $(node)")


    csv_path = joinpath(out_path, replace(lowercase(node), " " => "-"))

    # set up array for storing processed frameds
    dfs = DataFrame[]

    for (root, dirs, files) ∈ walkdir(csv_path)
        @showprogress for file ∈ files
            if endswith(file, ".csv")
                df = process_df(joinpath(root, file))
                push!(dfs, df)
            end
        end
    end

    println("There are $(length(dfs)) dataframes")
    println("\tcombining...")
    df = vcat(dfs...)
    dropmissing!(df)

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
    df_summary = combine(groupby(df, :group), nrow)


    save_path = "./data/processed/multi-sensor/"
    csv_path = joinpath(save_path, replace(lowercase(node), " " => "-"))

    if !ispath(csv_path)
        mkpath(csv_path)
    end

    CSV.write(joinpath(csv_path, "df.csv"), df)
    CSV.write(joinpath(csv_path, "df_summary.csv"), df_summary)
end
