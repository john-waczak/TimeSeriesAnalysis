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

