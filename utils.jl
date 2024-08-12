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


function parse_datetime(dt, tz)
    dt_out = String(dt)
    dt_out = ZonedDateTime(DateTime(split(dt, "Z")[1]), tz"UTC")
    return  astimezone(dt_out, tz)
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





function eval_havok(Zs, ts, n_embedding, r_model, n_control; method=:backward)
    r = r_model + n_control

    # cutoff time for training vs testing partition
    Zs_x = Zs[n_embedding:end]
    ts_x = range(ts[n_embedding], step=dt, length=length(Zs_x))

    # construct Hankel Matrix
    H = TimeDelayEmbedding(Zs; n_embedding=n_embedding, method=method);

    # Decompose via SVD
    U, σ, V = svd(H)

    # truncate the matrices
    Vr = @view V[:,1:r]
    Ur = @view U[:,1:r]
    σr = @view σ[1:r]


    X = Vr
    dX = zeros(size(Vr,1), r_model)

    for j ∈ axes(dX, 2)
        itp = CubicSpline(X[:,j], ts_x)
        for i ∈ axes(dX, 1)
            dX[i,j] = DataInterpolations.derivative(itp, ts_x[i])
        end
    end

    # Compute model matrix via least squares
    Ξ = (X\dX)'  # now Ξx = dx for a single column vector view
    A = Ξ[:, 1:r_model]   # State matrix A
    B = Ξ[:, r_model+1:end]      # Control matrix B


    # define interpolation function for forcing coordinate(s)
    itps = [DataInterpolations.LinearInterpolation(Vr[:,j], ts_x; extrapolate=true) for j ∈ r_model+1:r];
    forcing(t) = [itp(t) for itp ∈ itps]

    params = (A, B)
    x₀ = X[1,1:r_model]

    # define function and integrate to get model predictions
    function f!(dx, x, p, t)
        A,B = p
        dx .= A*x + B*forcing(t)
    end

    prob = ODEProblem(f!, x₀, (ts_x[1], ts_x[end]), params);
    sol = solve(prob, saveat=ts_x);
    X̂ = Array(sol)'

    # reconstruct original time series
    Ẑs_x = X̂ * diagm(σr[1:r_model]) * Ur[1,1:r_model]

    return Zs_x, Ẑs_x, ts_x, (ts_x[1]:ts_x[end])
end

