# SciML Tutorial:
# https://docs.sciml.ai/Overview/stable/showcase/missing_physics/


# SciML Tools
using DifferentialEquations, ModelingToolkit, DataDrivenDiffEq, SciMLSensitivity, DataDrivenSparse
using Optimization, OptimizationOptimisers, OptimizationOptimJL

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote, StableRNGs



using CairoMakie
include("./makie-defaults.jl")
include("./viz.jl")


# Seed stable random number generator
rng = StableRNG(1111)


# Generate Training Data
function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end


# define parameter values
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)  # need Float32 for Neural Network Libraries to be happy
p_ = [1.3, 0.9, 0.8, 1.8]


# define ODE problem
prob = ODEProblem(lotka!, u0, tspan, p_)
sol = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# add Noise via the mean
X = Array(sol)
t = sol.t

x̄ = mean(X, dims=2)
noise_mag = 5e-3

Xn = X .+ (noise_mag * x̄) .* randn(rng, eltype(X), size(X))


# visualize soln
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t")

lx = lines!(ax, t, X[1,:])
ly = lines!(ax, t, X[2,:])
sx = scatter!(ax, t, Xn[1,:])
sy = scatter!(ax, t, Xn[2,:])

fig[1,2] = Legend(fig, [lx, ly, sx, sy], ["true x", "true y", "noisy x", "noisy y"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)
fig



# Set up Universal Differential Equation

rbf(x) = exp.(-(x .^2))  # rbf kernel for activation function

# Multilayer Feed-Forward Network
const U = Lux.Chain(
    Lux.Dense(2, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 5, rbf),
    Lux.Dense(5, 2)
)

# Get initial parameters and state of NN
p, st = Lux.setup(rng, U)
const _st = st


# Define UDE as dynamical system u̇ = known(u) + NN(u)
# p are NN paramters, p_true are known model coefficients, i.e. α, δ
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1]  # NN prediction
    du[1] =  p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end


# Set up closure to handle extra arguments
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)


# set up ODE problem
prob_nn = ODEProblem(nn_dynamics!, Xn[:,1], tspan, p)



# set up function for generating solution with new parameters
function predict(θ, u0_new = Xn[:,1], T = t)
    _prob = remake(prob_nn, u0=u0_new, tspan = (T[1], T[end]), p=θ)

    Array(solve(_prob, Vern7(), saveat=T, abstol=1e-6, reltol=1e-6, sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end


# set up loss function to optimize
function loss(θ)
    X̂ = predict(θ)
    mean(abs2, Xn .- X̂)
end


# test out the loss function:
loss(p)


# set up array to track loss function with callback
losses = Float64[]


function callback(p, l)
    push!(losses, l)
    if length(losses) % 50 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end

    return false  # true signals early stopping
end


# Build Optimization Problem for Training
adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x,p) -> loss(x), adtype)
# ComponentVector is used here to flatten the parameters in an easy way with views
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))


# Train first using ADAM for rapid convergence to local minimum
res1 = Optimization.solve(optprob, ADAM(), callback=callback, maxiters=5000)

# Train second round using LBFGS to get better minimum using derivative information
optprob2 = Optimization.OptimizationProblem(optf, res1.u)
res2 = Optimization.solve(optprob2, Optim.LBFGS(), callback=callback, maxiters=1000)

# Rename the best candidate
p_trained = res2.u



# Visualize the Trained UDE
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="iteration", ylabel="loss", title="UDE Training", xscale=log10, yscale=log10);
l_adam = lines!(ax, 1:5000, losses[1:5000], linewidth=2)
l_lbfgs = lines!(ax, 5001:length(losses), losses[5001:end], linewidth=2)
fig[1,2] = Legend(fig, [l_adam, l_lbfgs], ["ADAM", "LBFGS"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)
# ylims!(ax, 1e-2, 1e6)
fig



# compare original data to output of UDE
ts = first(sol.t):(mean(diff(sol.t))/2):last(sol.t)  # use a finner time step for comparison
X̂ = predict(p_trained, Xn[:,1], ts)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="x(t), y(t)");

lx = lines!(ax, ts, X̂[1,:])
ly = lines!(ax, ts, X̂[2,:])

sx = scatter!(ax, sol.t, Xn[1,:])
sy = scatter!(ax, sol.t, Xn[2,:])

fig[1,2] = Legend(fig, [lx, ly, sx, sy], ["UDE x(t)", "UDE y(t)", "data x", "data y"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)

fig


# Visualize how NN has fit the Residual
Ȳ = [-p_[2] * (X̂[1, :] .* X̂[2, :])'; p_[3] * (X̂[1, :] .* X̂[2, :])']
Ŷ = U(X̂, p_trained, st)[1]

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t", ylabel="U(x,y)");

lx = lines!(ax, ts, Ŷ[1,:])
ly = lines!(ax, ts, Ŷ[2,:])
llx = lines!(ax, ts, Ȳ[1,:], linestyle=:dash)
lly = lines!(ax, ts, Ȳ[2,:], linestyle=:dash)

fig[1,2] = Legend(fig, [lx, ly, llx, lly], ["UDE Term 1", "UDE Term 2", "True Term 1", "True Term 2"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)

fig



# Set up variables for symbolic regression
@variables u[1:2]
b = polynomial_basis(u, 4)  # polynomial combinations up to order 4
basis = Basis(b, u);


# SINDy Problem i.e. Ẋ = NN
sindy_problem = ContinuousDataDrivenProblem(Xn, t)

# Ideal Problem
ideal_problem = DirectDataDrivenProblem(X̂, Ȳ)

# NN Problem
nn_problem = DirectDataDrivenProblem(X̂, Ŷ)


# solve using ADMM method
λ = exp10.(-3:0.01:3)  # learning schedule for λ
opt = ADMM(λ)


# solve problems
options = DataDrivenCommonOptions(maxiters = 10_000,
                                  normalize = DataNormalization(ZScoreTransform),
                                  selector = bic, digits = 1,
                                  data_processing = DataProcessing(split = 0.9,
                                                                   batchsize = 30,
                                                                   shuffle = true,
                                                                   rng = StableRNG(1111)))

sindy_res = solve(sindy_problem, basis, opt, options=options)
sindy_eqs = get_basis(sindy_res)
println(sindy_res)


options = DataDrivenCommonOptions(
    maxiters = 10_000,
    normalize = DataNormalization(ZScoreTransform),
    selector = bic,
    digits = 1,
    data_processing = DataProcessing(
        split = 0.9,
        batchsize = 30,
        shuffle = true,
        rng = StableRNG(1111)
    )
)

ideal_res = solve(ideal_problem, basis, opt, options=options)
ideal_eqs = get_basis(ideal_res)
println(ideal_res)



options = DataDrivenCommonOptions(maxiters = 10_000,
                                  normalize = DataNormalization(ZScoreTransform),
                                  selector = bic, digits = 1,
                                  data_processing = DataProcessing(split = 0.9,
                                                                   batchsize = 30,
                                                                   shuffle = true,
                                                                   rng = StableRNG(1111)))

nn_res = solve(nn_problem, basis, opt, options = options)
nn_eqs = get_basis(nn_res)


println(sindy_eqs)
println(ideal_eqs)
println(nn_eqs)


# Now that we have equation candidates, learn new parameters to make this work:
function recovered_dynamics!(du, u, p, t)
    û = nn_eqs(u, p)  # extract recovered equations
    du[1] =  p_[1]*u[1] + û[1]
    du[2] = -p_[4]*u[2] + û[2]
end

estimation_prob = ODEProblem(recovered_dynamics!, u0, tspan, get_parameter_values(nn_eqs))
estimate = solve(estimation_prob, Tsit5(), saveat = sol.t)

# visualize the solution
fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t");
le1 = lines!(ax, sol.t, estimate[1,:])
le2 = lines!(ax, sol.t, estimate[2,:])
ls1 = lines!(ax, sol.t, X[1,:])
ls2 = lines!(ax, sol.t, X[2,:])

fig[1,2] = Legend(fig, [le1, le2, ls1, ls2], ["UDE x(t)", "UDE y(t)", "True x(t)", "True y(t)"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)


# tune the solution now that we have a funcitonal form
# by minimizing gap between UDE term and our functional terms
function parameter_loss(p)
    Y = reduce(hcat, map(Base.Fix2(nn_eqs, p), eachcol(X̂)))
    sum(abs2, Ŷ .- Y)
end

optf = Optimization.OptimizationFunction((x,p) -> parameter_loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, get_parameter_values(nn_eqs))
parameter_res = Optimization.solve(optprob, Optim.LBFGS(), maxiters=1000)


# simulate the full problem now
t_long = (0.0, 50.0)
estimation_prob = ODEProblem(recovered_dynamics!, u0, t_long, parameter_res)
estimate_long = solve(estimation_prob, Tsit5(), saveat = 0.1)

true_prob = ODEProblem(lotka!, u0, t_long, p_)
true_solution_long = solve(true_prob, Tsit5(), saveat=estimate_long.t)

fig = Figure();
ax = CairoMakie.Axis(fig[1,1], xlabel="t");

lx_e = lines!(ax, estimate_long.t, estimate_long[1,:])
ly_e = lines!(ax, estimate_long.t, estimate_long[2,:])

lx_t = lines!(ax, true_solution_long.t, true_solution_long[1,:], linestyle=:dash)
ly_t = lines!(ax, true_solution_long.t, true_solution_long[2,:], linestyle=:dash)

fig[1,2] = Legend(fig, [lx_e, ly_e, lx_t, ly_t], ["Estimated x(t)", "Estimated y(t)", "Estimated x(t)", "Estimated y(t)"], framevisible=false, orientation=:vertical, padding=(0,0,0,0), labelsize=17) # , height=-5)

fig





