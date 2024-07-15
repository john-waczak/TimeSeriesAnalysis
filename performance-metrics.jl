using Statistics
using StatsBase

# set up functions for time series evaluations
mean_bias(ŷ, y) = mean(ŷ .- y)
mean_absolute_error(ŷ, y) = mean(abs.(ŷ .- y))
normalized_mean_bias(ŷ, y) = sum(ŷ .- y) / sum(y)
normalized_mae(ŷ, y) = sum(abs.(ŷ .- y)) / sum(y)
rmse(ŷ, y) = sqrt(mean(abs2.(ŷ .- y)))
corr_coef(ŷ, y) = cor(ŷ, y)

function coefficient_of_efficiency(ŷ, y)
    μ_o = mean(y)
    return 1.0 - sum(abs.(ŷ .- y))/sum(abs.(y .- μ_o))
end

r_squared(ŷ, y) =  1 - sum(abs2.(ŷ .- y))/sum(abs2.(mean(y) .- y))

