# Function for obtain Hankel Matrix of time-delay embeddings
function TimeDelayEmbedding(z; n_embedding=100, method=:forward)
    ncol = length(z) - n_embedding + 1

    # allocate output
    H = zeros(eltype(z), n_embedding, ncol)

    for k ∈ 1:ncol
        H[:,k] = z[k:k+n_embedding-1]
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



function r_expvar(σ; cutoff=0.9)
    expvar = cumsum(σ ./ sum(σ))
    return findfirst(expvar .> cutoff)
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



function load_data(df_path, t1, t2, col_to_use; interp=true)
    println("Loading data...")
    df = CSV.read(df_path, DataFrame, select=[:datetime, col_to_use]);

    # find indices closest to t1 and t2
    idx_start = argmin([abs((Date(dt) - t1).value) for dt ∈ df.datetime])
    idx_end = argmin([abs((Date(dt) - t2).value) for dt ∈ df.datetime])

    # clip to t1 and t2
    println("Clipping to $(t1)-$(t2)")
    df = df[idx_start:idx_end,:]

    # create a single dataset interpolated to every second
    if interp
        println("Interpolating to 1s")
        zs_df = df[:, col_to_use];
        ts_df = [(dt_f .- df.datetime[1]).value ./ 1000 for dt_f ∈ df.datetime];
        z_itp = LinearInterpolation(zs_df, ts_df)

        ts = ts_df[1]:ts_df[end]
        Zs = z_itp.(ts)
    else
        ts = td_df
        Zs = zd_df
    end

    return Zs, ts
end


