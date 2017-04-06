# normal with conjugate prior and known variance using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

function metropolis(prior, known_sd, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

    # half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> -logpdf(prior, x[1]) -
                                         loglikelihood(Normal(x[1], σ), D), q)

    # take L full steps for position and L-1 for momentum
    for i in 1:L
        q += ϵ * p

        if i != L
            p -= ϵ * ForwardDiff.gradient(x -> -logpdf(prior, x[1]) -
                                                 loglikelihood(Normal(x[1], σ),
                                                               D), q)
        end
    end

    # finish with a half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> -logpdf(prior, x[1]) -
                                         loglikelihood(Normal(x[1], σ), D), q)

    # negate momentum so metropolis proposal is symmetric
    p = -p

    # calculate hamiltonian for new and old momentum and position
    current_U = -logpdf(prior, current_q[1]) -
                 loglikelihood(Normal(current_q[1], σ), D)
    current_K = sum(current_p .^ 2) / 2
    proposed_U = -logpdf(prior, q[1]) - loglikelihood(Normal(q[1], σ), D)
    proposed_K = sum(p .^ 2) / 2
    accept_prob = exp(current_U - proposed_U + current_K - proposed_K)

    # accept or reject proposal with metropolis
    if rand(Uniform(), 1)[1] < accept_prob
        return q # accept
    else
        return current_q # reject
    end
end

function HMC(D, σ; μ₀=0, σ₀=100, S=1000, L=50, ϵₐ=0.0104, ϵᵦ=0.0156)
    # allocate chain
    chain = zeros(S, 1)

    # construct prior
    prior = Normal(μ₀, σ₀)

    # start at prior
    chain[1, :] = rand(prior, 1)

    for s in 2:S
        ϵ = rand(Uniform(ϵₐ, ϵᵦ), 1)[1]
        chain[s, :] = metropolis(Normal(μ₀, σ₀), σ, ϵ, L, chain[s-1, :])
    end

    return chain
end

# true values
μ = 10.0
σ = 1

# priors
μ₀ = 0
σ₀ = 100

# simulate data
srand(1)
n = 1000
D = rand(Normal(μ, σ), n)

# mcmc parameters
burnin = 100 # number of burnin samples
iter = 1100 # total samples
leapfrog = 10 # number of leapfrog steps

chain = HMC(D, σ, L=leapfrog, S=iter)
mean(chain[(burnin+1):iter, ])
std(chain[(burnin+1):iter, ])

using Gadfly
plot(y=chain[:, 1])
