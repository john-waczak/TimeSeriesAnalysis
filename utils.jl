using Dates
using TimeZones

function date2datetime(dt::AbstractString;  df = dateformat"yyyy-mm-dd HH:MM:SS.sss", timezone=tz"UTC")
    dt_string = String(dt)
    dt_split = split(dt_string, ".")
    dt_millisecond = join([dt_split[1], dt_split[2][1:end-3]], ".")
    return ZonedDateTime(DateTime(dt_millisecond, df), timezone)
end


function window_idxs(i, N, Z::AbstractVector)
    if N > length(Z)
        thow(ArgumentError("N must be less than length of data"))
    end

    n = Int((N-1)/2)

    idx_out = i-n:i+n

    # handle edge cases
    if idx_out[1] < 1
        offset = 1-idx_out[1]
        idx_out = idx_out .+ offset
    end

    if idx_out[end] > length(Z)
        offset = idx_out[end] - length(Z)
        idx_out = idx_out .- offset
    end

    return idx_out
end

window_idxs(i, N, Z::AbstractRegularTimeSeries) = window_idxs(i, N, Z.z)


# Now we should be able to easily compute the representativeness
function rolling_deviation(Z::AbstractRegularTimeSeries, Nwindow::Int; func=std)
    Δz = zeros(length(Z))
    for i ∈ 1:length(Z)
        Δz[i] = func(Z.z[window_idxs(i,Nwindow,Z)])
    end
    return Δz
end

function mean_dev(Z::AbstractVector)
    μ = mean(Z)
    return sum(abs.(Z .- μ))/length(Z)
end



function r_cutoff(σ; ratio=0.01, rmax=15)
    return min(sum(σ ./ sum(σ) .> ratio) + 1, rmax)
end


# see this paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e2428512fcfe5d907c0db26cae4546872a19a954
function r_optimal_approx(σ, m, n)
    β = m/n
    ω = 0.56*β^3 - 0.95*β^2 + 1.82β + 1.43

    r = length(σ[σ .< ω * median(σ)])
end

