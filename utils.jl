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

