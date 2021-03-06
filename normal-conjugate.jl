# normal with conjugate prior and known variance using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

# simulate hamiltonian dynamics for L steps
# and accept/reject based on metropolis using
# canonical distribution
function metropolis(prior_mu, prior_tau, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

    # potential energy
    function U(x)
        return -logpdf(prior_mu, x[1]) - logpdf(prior_tau, x[2]) -
                loglikelihood(Normal(x[1], sqrt(1. / x[2])), D)
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

function HMC(D, orig; μ₀=0, σ₀=100, A=0.01, B=0.01,
             S=1000, L=50, ϵₐ=0.0104, ϵᵦ=0.0156)
    # allocate chain
    chain = zeros(S, 2)

    # construct prior
    prior_mu = Normal(μ₀, σ₀)
    prior_tau = Gamma(A, 1. / B)

    # start at prior
    chain[1, 1] = orig[1]
    chain[1, 2] = orig[2]

    for s in 2:S
        ϵ = rand(Uniform(ϵₐ, ϵᵦ), 1)[1]
        chain[s, :] = metropolis(prior_mu, prior_tau, ϵ, L, chain[s-1, :])
    end

    return chain
end

# true values
μ = 10.0
τ = 0.01

# priors
μ₀ = 0.
σ₀ = 100.
A = 0.1
B = 0.1

# simulate data
srand(1)
n = 1000
D = rand(Normal(μ, sqrt(1./τ)), n)

# mcmc parameters
burnin = 500 # number of burnin samples
iter = 1500 # total samples
leapfrog = 100 # number of leapfrog steps
ϵₐ=0.0104 / 50. # lower bound of ϵ
ϵᵦ=0.0156 / 50. # upper bound of ϵ

chain = HMC(D, [25., 2.], L=leapfrog, S=iter, ϵₐ=ϵₐ, ϵᵦ=ϵᵦ, A=A, B=B)

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
