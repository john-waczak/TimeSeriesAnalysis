using CSV, DataFrames
using Plots
using TimeSeriesTools
using ParameterHandling
using Dates, TimeZones
using Unitful
using Markdown, LaTeXStrings
using Statistics, StatsBase
using BenchmarkTools
using LinearAlgebra
using DifferentialEquations
using Interpolations
using StaticArrays


include("utils.jl")

if !ispath("figues/sharedair/variograms")
    mkpath("figures/sharedair/variograms")
end


df = CSV.File("data/sharedair/data.csv") |> DataFrame

nrow(df)

# verify smooth transition between files"
println(df1.dateTime[end], "\t", df2.dateTime[1])
println(df2.dateTime[end], "\t", df3.dateTime[1])


# First, let's take our donloaded data and construct some timeseries to analyze:
describe(df)

# generate a timeseries object for each
pm2_5 = RegularTimeSeries(
    df.pm2_5,
    1.0,
    u"μg/m^3",
    u"s",
    ZonedDateTime(2023, 3, 4, 0, 0, 0, 035, tz"UTC")
)

pm10_0 = RegularTimeSeries(
    df.pm10_0,
    1.0,
    u"μg/m^3",
    u"s",
    ZonedDateTime(2023, 3, 4, 0, 0, 0, 035, tz"UTC")
)

pm1_0 = RegularTimeSeries(
    df.pm1_0,
    1.0,
    u"μg/m^3",
    u"s",
    ZonedDateTime(2023, 3, 4, 0, 0, 0, 035, tz"UTC")
)

pmtemp = RegularTimeSeries(
    df.temperature,
    1.0,
    u"μg/m^3",
    u"s",
    ZonedDateTime(2023, 3, 4, 0, 0, 0, 035, tz"UTC")
)



p1 = plot(
    times(pm10_0) * pm10_0.t_units .|> u"hr",
    pm10_0.z * pm10_0.z_units,
    xlabel="time",
    ylabel="Concentration",
    title="Time series for PM 10.0 starting on $(Date(pm10_0.start_time))",
    label="PM 10.0",
#    xlims=(0, Inf),  # lims only considers numbers and will autoselect for any Inf supplied.
#    ylims=(0, Inf),
    lw=2,
    alpha=0.5,
)

plot!(
    times(pm2_5) * pm2_5.t_units .|> u"hr",
    pm2_5.z * pm2_5.z_units,
    xlabel="time",
    ylabel="Concentration",
    title="Time series for PM 10 starting on $(Date(pm2_5.start_time))",
    label="PM 2.5",
#    xlims=(0, Inf),  # lims only considers numbers and will autoselect for any Inf supplied.
#    ylims=(0, Inf),
    lw=2,
    alpha=0.5
)

plot!(
    times(pm1_0) * pm1_0.t_units .|> u"hr",
    pm1_0.z * pm1_0.z_units,
    xlabel="time",
    ylabel="Concentration",
    title="Time series for PM 10 starting on $(Date(pm1_0.start_time))",
    label="PM 1.0",
#    xlims=(0, Inf),  # lims only considers numbers and will autoselect for any Inf supplied.
#    ylims=(0, Inf),
    lw=2,
    alpha=0.5
)

xlims!(0, 0.25)
ylims!(15, 60)


savefig("figures/sharedair/variograms/time-series.png")
savefig("figures/sharedair/variograms/time-series.pdf")
savefig("figures/sharedair/variograms/time-series.svg")


# evaluate mean absolute deviation on sliding window to estimate uncertainty
ribbon = 2*rolling_deviation(pm2_5, 5*60+1; func=mean_dev)  # 5 minute window

p1 = plot(
    times(pm2_5)* pm2_5.t_units .|> u"hr",
    pm2_5.z * pm2_5.z_units,
    ribbon=ribbon,
    xlabel="time",
    ylabel="Concentration",
    title="Time series for PM 10 on $(Date(pm2_5.start_time))",
    label="",
    xlims=(0, 0.5),  # lims only considers numbers and will autoselect for any Inf supplied.
    ylims=(0, 100.0),
    lw=3
)

savefig("figures/sharedair/variograms/time-series_unc.png")
savefig("figures/sharedair/variograms/time-series_unc.pdf")





# ## Model 2: Uncertainty via Semivariogram
γ, h = semivariogram(pm2_5; lag_max=0.5*60*60)

γ_params = (
    nugget=positive(0.01),
    sill=positive(0.1),
    range=positive(100.0)
)

γ_fit = fit_spherical_γ(h, γ, γ_params)

scatter(
    h * pm2_5.t_units .|> u"minute",
    γ,
    xlabel=L"\Delta t",
    ylabel=L"\gamma(\Delta t)",
    title="IPS7100 semivariogram for $(Date(pm2_5.start_time))",
    label=L"Empirical $\gamma$",
    ms=2,
    msw=0,
    xlims=(-0.075, Inf),
    ylims=(0, Inf)
)

h_fit = 0.0:1.0:h[end]
γ_out = γ_fit.(h_fit)

plot!(
    h_fit * pm2_5.t_units .|> u"minute",
    γ_out,
    label="Spherical Model",
    lw=3,
)

scatter!([0], [nugget(γ_fit)], label="nugget", color=:grey)
vline!([γ_range(γ_fit) * pm2_5.t_units |> u"minute"], ls=:dash, color=:grey, label="range")
hline!([sill(γ_fit) + nugget(γ_fit)], ls=:dash, color=:brown, label="sill")

savefig("figures/sharedair/variograms/variogram.png")
savefig("figures/sharedair/variograms/variogram.pdf")


# now we can reevaluate the uncertainty on a window determined by the range of our variogram
Nwindow = round(Int, γ_range(γ_fit)/pm2_5.Δt)
if Nwindow % 2 == 0
    Nwindow += 1
end

p1 = plot(
    times(pm2_5)* pm2_5.t_units .|> u"hr",
    pm2_5.z * pm2_5.t_units,
    ribbon = 2*rolling_deviation(pm2_5, Nwindow; func=mean_dev),  # 5 minute window
    xlabel="time",
    ylabel="Concentration",
    title="Time series for PM 10 on $(Date(pm2_5.start_time))",
    label="",
    xlims=(0, 0.5),  # lims only considers numbers and will autoselect for any Inf supplied.
    ylims=(0, 100), 
    lw=3
)

savefig("figures/sharedair/variograms/time-series_variogram-unc.png")
savefig("figures/sharedair/variograms/time-series_variogram-unc.pdf")







