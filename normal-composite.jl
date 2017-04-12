# normal with conjugate prior and known variance using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

# true values
μ = 10.0
σ = 1.5

# priors
μ₀ = 0.
σ₀ = 5.0
A = 0.1
B = 0.1

# simulate data
srand(1)
n = 1000
D = rand(Normal(μ, σ), n)

# mcmc parameters
burnin = 100 # number of burnin samples
iter = 1000 + burnin # total samples
leapfrog = 10 # number of leapfrog steps
ϵₐ=0.01 # lower bound of ϵ
ϵᵦ=0.05 # upper bound of ϵ

# create composite for model
type NormalModel
    # data
    D::Vector{Float64}

    # prior distributions
    μ₀::Float64
    σ₀::Float64
    A::Float64
    B::Float64

    # mcmc parameters
    burnin::Int
    S::Int

    # chains
    chain::Matrix{Float64}

    NormalModel(D, μ₀, σ₀, A, B, burnin, S, chain) = burnin >= S ?
        error("S must be greater than burnin") :
        new(D, μ₀, σ₀, A, B, burnin, S, chain)
end

function NormalModel(D, μ₀, σ₀, A, B, burnin, S)
    NormalModel(D, μ₀, σ₀, A, B, burnin, S, zeros(S, 2))
end

# simulate hamiltonian dynamics for L steps
# and accept/reject based on metropolis using
# canonical distribution
function hmc_step(prior_mu, prior_sigma, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

    # potential energy
    function U(x)
        return -logpdf(prior_mu, x[1]) - logpdf(prior_sigma, x[2]) -
                loglikelihood(Normal(x[1], x[2]), D)
    end

    # half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(U, q)

    # take L full steps for position and L-1 for momentum
    for i in 1:L
        q += ϵ * p

        # constrain τ variable
        if q[2] < 0.001
            q[2] = 0.001 + (0.001 - q[2])
            p[2] *= -1.
        end

        if i != L
            p -= ϵ * ForwardDiff.gradient(U, q)
        end
    end

    # finish with a half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(U, q)

    # negate momentum so metropolis proposal is symmetric
    p = -p

    # calculate hamiltonian for new and old momentum and position
    current_U = U(current_q)
    current_K = sum(current_p .^ 2) / 2
    proposed_U = U(q)
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
    # construct prior
    prior_mu = Cauchy(model.μ₀, model.σ₀)
    prior_sigma = Gamma(model.A, 1. / model.B)

    # starting point
    model.chain[1, 1] = orig[1]
    model.chain[1, 2] = orig[2]

    for s in 2:S
        ϵ = rand(Uniform(ϵₐ, ϵᵦ), 1)[1]
        model.chain[s, :] = hmc_step(prior_mu, prior_sigma, ϵ, L,
                                      model.chain[s-1, :])
    end

    return model
end

model = NormalModel(D, 0, 100, 0.1, 0.1, 100, 1100)

# check acceptance ratio
rejects = 0
for i in 2:iter
    if chain[i, :] == chain[(i-1), :]
        rejects += 1
    end
end

(iter - rejects) / iter

mean(chain[(burnin+1):iter, :], 1)
std(chain[(burnin+1):iter, :], 1)

using Gadfly
plot(y=chain[:, 1])
plot(y=chain[:, 2])
