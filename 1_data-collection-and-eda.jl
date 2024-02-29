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

outpath = joinpath(abspath("./data"), "processed")
if !ispath(outpath)
    mkpath(outpath)
end

figpath = joinpath(abspath("./"), "figures", "time-series")
if !ispath(figpath)
    mkpath(figpath)
end


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

open("config.jl", "w") do f
    println(f, "cor_cols = [")
    for col ∈ cor_cols
        println(f, "\t\"$(col)\",")
    end
    println(f, "]")

    println(f, "\n")
    println(f, "epa_cols = [")
    for col ∈ epa_cols
        println(f, "\t\"$(col)\",")
    end
    println(f, "]")

    println(f, "\n")
    println(f, "atmos_cols = [")
    for col ∈ atmos_cols
        println(f, "\t\"$(col)\",")
    end
    println(f, "]")

    println(f, "\n")
    println(f, "ips_cols = [")
    for col ∈ ips_cols
        println(f, "\t\"$(col)\",")
    end
    println(f, "]")
end


unnamed = [n for n in names(df) if occursin("Unnamed", n)]
df = df[:, Not(unnamed)];


df.dateTime = [ZonedDateTime(String(dt), "yyyy-mm-dd HH:MM:SSzzzz") for dt in df.dateTime];


# create new column for date time rounded to nearest quarter hour, hour, 8 hours, and 24 hours
df.quarter_hour = round.(df.dateTime, Dates.Minute(15))
df.hour = round.(df.dateTime, Dates.Hour)
df.eight_hour = round.(df.dateTime, Dates.Hour(8))
df.day = round.(df.dateTime, Dates.Day)


# now let's process each group going backwards and save the final dataframes
gdf_day = groupby(df, :day)
df_day = combine(gdf_day, valuecols(gdf) .=> mean; renamecols = false)
# create dt column
df_day.dt = [Day(day .- df_day.day[1]).value for day in df_day.day]
df_day.t_skip = vcat(0, Day.(df_day.day[2:end] .- df_day.day[1:end-1]) .!= Day(1))
@assert all(df_day.t_skip .== 0)

train_frac = 0.8
idx_test = round(Int, nrow(df_day)*train_frac)
df_day.group = [i<idx_test ? 1 : 2 for i ∈ 1:nrow(df_day)]

df_day_summary = combine(groupby(df_day, :group), nrow)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Days since $(Date(df_day.day[1]))", ylabel="PM 2.5", title="24-hour Averaged Data");
l1 = lines!(ax, df_day.dt, df_day.pm2_5, linewidth=3)
l2 = lines!(ax, df_day.dt, df_day.epa_pm2_5, linewidth=3)
l3 = lines!(ax, df_day.dt, df_day.cor_pm2_5, linewidth=3)
axislegend(ax, [l1, l2, l3], ["Raw", "Corrected", "EPA"]; position=:lt)

save(joinpath(figpath, "24-hour.png"), fig)
save(joinpath(figpath, "24-hour.pdf"), fig)

fig
CSV.write(joinpath(outpath, "df_day.csv"), df_day)
CSV.write(joinpath(outpath, "df_day-summary.csv"), df_day_summary)




# 8 hour
gdf_eight = groupby(df, :eight_hour)
df_eight = combine(gdf_eight, valuecols(gdf) .=> mean; renamecols = false)
df_eight.dt = [Hour(e .- df_eight.eight_hour[1]).value for e in df_eight.eight_hour]
df_eight.t_skip = vcat(0, Hour.(df_eight.eight_hour[2:end] .- df_eight.eight_hour[1:end-1]) .!= Hour(8))

idx_group = Int[]
i = 1
for j ∈ 1:nrow(df_eight)
    if df_eight.t_skip[j] == 1
        i += 1
    end
    push!(idx_group, i)
end
df_eight.group = idx_group
df_eight_summary = combine(groupby(df_eight, :group), nrow)

fig = Figure();
ax = Axis(fig[1,1], xlabel="Days since $(Date(df_eight.eight_hour[1]))", ylabel="PM 2.5", title="8-hour Averaged Data");
l1 = lines!(ax, df_eight.dt ./ 24, df_eight.pm2_5, linewidth=3)
l2 = lines!(ax, df_eight.dt ./ 24, df_eight.epa_pm2_5, linewidth=3)
l3 = lines!(ax, df_eight.dt ./ 24, df_eight.cor_pm2_5, linewidth=3)
axislegend(ax, [l1, l2, l3], ["Raw", "Corrected", "EPA"]; position=:lt)
fig

save(joinpath(figpath, "8-hour.png"), fig)
save(joinpath(figpath, "8-hour.pdf"), fig)

CSV.write(joinpath(outpath, "df_eight_hour.csv"), df_eight)
CSV.write(joinpath(outpath, "df_eight_hour-summary.csv"), df_eight_summary)




# quarter hour
gdf_quarter = groupby(df, :quarter_hour)
df_quarter = combine(gdf_quarter, valuecols(gdf) .=> mean; renamecols = false)
df_quarter.dt = [Minute(q .- df_quarter.quarter_hour[1]).value for q in df_quarter.quarter_hour]
df_quarter.t_skip = vcat(0, Minute.(df_quarter.quarter_hour[2:end] .- df_quarter.quarter_hour[1:end-1]) .!= Minute(15))

idx_group = Int[]
i = 1
for j ∈ 1:nrow(df_quarter)
    if df_quarter.t_skip[j] == 1
        i += 1
    end
    push!(idx_group, i)
end
df_quarter.group = idx_group
df_quarter_summary = combine(groupby(df_quarter, :group), nrow)


fig = Figure();
ax = Axis(fig[1,1], xlabel="Days since $(Date(df_quarter.quarter_hour[1]))", ylabel="PM 2.5", title="15-minute Averaged Data");
l1 = lines!(ax, df_quarter.dt ./ (60*24), df_quarter.pm2_5, linewidth=3, alpha=0.8)
l3 = lines!(ax, df_quarter.dt ./ (60*24), df_quarter.cor_pm2_5, linewidth=3, alpha=0.8)
l2 = lines!(ax, df_quarter.dt ./ (60*24), df_quarter.epa_pm2_5, linewidth=3, alpha=0.8)
axislegend(ax, [l1, l2, l3], ["Raw", "Corrected", "EPA"]; position=:rt)
xlims!(ax, 0, nothing)
fig

save(joinpath(figpath, "15-minute.png"), fig)
save(joinpath(figpath, "15-minute.pdf"), fig)


CSV.write(joinpath(outpath, "df_quarter_hour.csv"), df_quarter)
CSV.write(joinpath(outpath, "df_quarter_hour-summary.csv"), df_quarter_summary)



# 5-min
df = df[:, Not([:quarter_hour, :hour, :eight_hour, :day])]
df.dt = [Minute(m .- df.dateTime[1]).value for m in df.dateTime]
df.t_skip = vcat(0, Minute.(df.dateTime[2:end] .- df.dateTime[1:end-1]) .!= Minute(5))

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

fig = Figure();
ax = Axis(fig[1,1], xlabel="Days since $(Date(df.dateTime[1]))", ylabel="PM 2.5", title="5-minute Data");
l1 = lines!(ax, df.dt ./ (60*24), df.pm2_5, linewidth=3, alpha=0.8)
l3 = lines!(ax, df.dt ./ (60*24), df.cor_pm2_5, linewidth=3, alpha=0.8)
l2 = lines!(ax, df.dt ./ (60*24), df.epa_pm2_5, linewidth=3, alpha=0.8)
axislegend(ax, [l1, l2, l3], ["Raw", "Corrected", "EPA"]; position=:rt)
xlims!(ax, 0, df.dt[end]./(60*24))
fig

save(joinpath(figpath, "5-minute.png"), fig)
save(joinpath(figpath, "5-minute.pdf"), fig)


CSV.write(joinpath(outpath, "df_5_min.csv"), df)
CSV.write(joinpath(outpath, "df_5_min-summary.csv"), df_summary)
