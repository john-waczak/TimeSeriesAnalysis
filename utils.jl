# Function for obtain Hankel Matrix of time-delay embeddings
function TimeDelayEmbedding(z; nrow=100, method=:forward)
    ncol = length(z) - nrow + 1

    # allocate output
    H = zeros(eltype(z), nrow, ncol)

    for k ∈ 1:ncol
        H[:,k] = z[k:k+nrow-1]
    end

    if method == :backward
        H .= H[end:-1:1, :]
    end

    return H
end



function parse_datetime(dt)
    dt_out = String(dt)

    if occursin(".", dt)
        try
            # return ZonedDateTime(dt, "yyyy-mm-dd HH:MM:SS.sssssszzzz")
            return DateTime(dt[1:end-6], dateformat"yyyy-mm-dd HH:MM:SS.ssssss")
        catch e
            return missing
        end
    else
        try
            # return ZonedDateTime(dt, "yyyy-mm-dd HH:MM:SSzzzz")
            return DateTime(dt[1:end-6], dateformat"yyyy-mm-dd HH:MM:SS")
        catch e
            return missing
        end
    end
end



function r_cut(σ; ratio=0.01, rmax=15)
    return min(sum(σ ./ sum(σ) .> ratio) + 1, rmax)
end


# see this paper: https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e2428512fcfe5d907c0db26cae4546872a19a954
function r_optimal_approx(σ, m, n)
    β = m/n
    ω = 0.56*β^3 - 0.95*β^2 + 1.82β + 1.43

    r = length(σ[σ .< ω * median(σ)])
end



