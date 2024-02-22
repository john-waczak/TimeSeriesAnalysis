using CSV, DataFrames
using DifferentialEquations
using Dates, TimeZones
using DataInterpolations
using Statistics
using CairoMakie, MintsMakieRecipes

# Set some plotting defaults
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


df = CSV.read("./data/fort-worth-pm.csv", DataFrame);


unnamed = [n for n in names(df) if occursin("Unnamed", n)]
df = df[:, Not(unnamed)];


df.dateTime = [ZonedDateTime(String(dt), "yyyy-mm-dd HH:MM:SSzzzz") for dt in df.dateTime];

gdf = groupby(df, :dateTime)
df = combine(gdf, valuecols(gdf) .=> mean; renamecols=false);


idx_bad = findall((df.dateTime[2:end] .- df.dateTime[1:end-1]) .!= Millisecond(300000))

df.dateTime[idx_bad[3]+1] - df.dateTime[idx_bad[3]]


cor_cols = [n for n ∈ names(df) if occursin("cor", n)]
epa_cols = [n for n ∈ names(df) if occursin("epa", n)]
atmos_cols = ["temperature", "pressure", "humidity", "dewPoint", "altitude"]
ips_cols = [
    "pc0_1", "pc0_3", "pc0_5", "pc1_0", "pc2_5", "pc5_0", "pc10_0",
    "pm0_1", "pm0_3", "pm0_5", "pm1_0", "pm2_5", "pm5_0", "pm10_0",
]

push!(cor_cols, "dateTime")
push!(epa_cols, "dateTime")
push!(atmos_cols, "dateTime")
push!(ips_cols, "dateTime")

df_cor = df[:, cor_cols];
df_epa = df[:, epa_cols];
df_atmos = df[:, atmos_cols];
df_ips = df[:, ips_cols];
