# normal with conjugate prior and known variance using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

# potential energy
function U(x, D, prior_mu, prior_sigma)
    return -logpdf(prior_mu, x[1]) - logpdf(prior_sigma, x[2]) -
            loglikelihood(Normal(x[1], x[2]), D)
end

# canonical probability density
function canonical(q, p, q_prime, p_prime, prior_mu, prior_sigma, D)
    current_U = U(q, D, prior_mu, prior_sigma)
    current_K = sum(p .^ 2) / 2
    proposed_U = U(q_prime, D, prior_mu, prior_sigma)
    proposed_K = sum(p_prime .^ 2) / 2
    accept_prob = exp(current_U - proposed_U + current_K - proposed_K)

    return accept_prob
end

function leapfrog(q, p, ϵ, prior_mu, prior_tau, D)
    # half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> U(x, D, prior_mu, prior_sigma), q)

    # take full step for position
    q += ϵ * p

    # constrain τ variable
    if q[2] < 0.001
        q[2] = 0.001 + (0.001 - q[2])
        p[2] *= -1.
    end

    # finish with a half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> U(x, D, prior_mu, prior_sigma), q)

    return q, p
end

# simulate hamiltonian dynamics for L steps
# and accept/reject based on metropolis using
# canonical distribution
function metropolis(prior_mu, prior_tau, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

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

function HMC(D, orig; μ₀=0, σ₀=5, A=0.1, B=0.1,
             S=1000, L=10)
    # allocate chain
    chain = zeros(S, 2)

    # construct prior
    prior_mu = Cauchy(μ₀, σ₀)
    prior_sigma = Gamma(A, 1. / B)

    # starting point
    chain[1, 1] = orig[1]
    chain[1, 2] = orig[2]

    # choose initial value of ϵ
    ϵ = 1.0
    q = chain[1, :]
    p = rand(Normal(0, 1), length(q))

    q_prime, p_prime = leapfrog(q, p, ϵ, prior_mu, prior_sigma, D)
    a = canonical(q, p, q_prime, p_prime, prior_mu, prior_sigma, D) > 0.5 ?
        1 : -1

    while (canonical(q, p, q_prime, p_prime, prior_mu, prior_sigma, D) ^ a) >
          2. ^ (-a)
          ϵ = 2. ^ a * ϵ
          q_prime, p_prime = leapfrog(q, p, ϵ, prior_mu, prior_sigma, D)
    end

    for s in 2:S
        chain[s, :] = metropolis(prior_mu, prior_tau, ϵ, L, chain[s-1, :])
    end

    return chain
end

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

chain = HMC(D, [100., 25.], L=leapfrog, S=iter, A=A, B=B)

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
