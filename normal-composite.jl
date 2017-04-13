# normal with conjugate prior and known variance using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

# true values
μ = 10.0
σ = 1.5

# simulate data
srand(1)
n = 1000
D = rand(Normal(μ, σ), n)

# create composite for model
type NormalModel
    # data
    D::Vector{Float64}

    # prior distributions
    prior_mu::Distributions.Cauchy
    prior_sigma::Distributions.Gamma

    # mcmc parameters
    burnin::Int
    S::Int

    # chains
    chain::Matrix{Float64}

    NormalModel(D, prior_mu, prior_sigma, burnin, S, chain) = burnin >= S ?
        error("S must be greater than burnin") :
        new(D, prior_mu, prior_sigma, burnin, S, chain)
end

# create outer constructor
function NormalModel(D; μ₀=0.0, σ₀=5.0, A=0.1, B=0.1, burnin=100, S=1100)
    prior_mu = Cauchy(μ₀, σ₀)
    prior_sigma = Gamma(A, 1. / B)
    NormalModel(D, prior_mu, prior_sigma, burnin, S, zeros(S, 2))
end

# potential energy
function U(model::NormalModel, x::Vector)
    return -logpdf(model.prior_mu, x[1]) - logpdf(model.prior_sigma, x[2]) -
            loglikelihood(Normal(x[1], x[2]), model.D)
end

# simulate hamiltonian dynamics for L steps
# and accept/reject based on metropolis using
# canonical distribution
function hmc_step(model::NormalModel, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

    # half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> U(model, x), q)

    # take L full steps for position and L-1 for momentum
    for i in 1:L
        q += ϵ * p

        # constrain τ variable
        if q[2] < 0.001
            q[2] = 0.001 + (0.001 - q[2])
            p[2] *= -1.
        end

        if i != L
            p -= ϵ * ForwardDiff.gradient(x -> U(model, x), q)
        end
    end

    # finish with a half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> U(model, x), q)

    # negate momentum so metropolis proposal is symmetric
    p = -p

    # calculate hamiltonian for new and old momentum and position
    current_U = U(model, current_q)
    current_K = sum(current_p .^ 2) / 2
    proposed_U = U(model, q)
    proposed_K = sum(p .^ 2) / 2
    accept_prob = exp(current_U - proposed_U + current_K - proposed_K)

    # accept or reject proposal with metropolis
    if rand(Uniform(), 1)[1] < accept_prob
        return q # accept
    else
        return current_q # reject
    end
end

function post(model::NormalModel; orig=[0.0, 1.0], L=10, ϵₐ=0.01, ϵᵦ=0.05)
    # starting point
    model.chain[1, 1] = orig[1]
    model.chain[1, 2] = orig[2]

    for s in 2:model.S
        ϵ = rand(Uniform(ϵₐ, ϵᵦ), 1)[1]
        model.chain[s, :] = hmc_step(model, ϵ, L, model.chain[s-1, :])
    end

    return nothing
end

function acceptance_rate(model::NormalModel)
    # check acceptance ratio
    rejects = 0
    for i in 2:model.S
        if model.chain[i, :] == model.chain[(i-1), :]
            rejects += 1
        end
    end

    (model.S - rejects) / model.S
end

model = NormalModel(D)
post(model, orig=[-10.0, 4.0])
acceptance_rate(model)
mean(model.chain[(model.burnin+1):model.S, :], 1)
std(model.chain[(model.burnin+1):model.S, :], 1)

using Gadfly
plot(y=model.chain[:, 1])
plot(y=model.chain[:, 2])
