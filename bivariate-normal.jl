# sample from bivariate normal using HMC
# based on radford neals code in "MCMC using Hamiltonian Dynamics"
using Distributions, ForwardDiff

function metropolis(U, ϵ, L, current_q)
    q = copy(current_q)
    p = rand(Normal(0, 1), length(q))
    current_p = copy(p)

    # half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> -logpdf(U, x), q)

    # take L full steps for position and L-1 for momentum
    for i in 1:L
        q += ϵ * p

        if i != L
            p -= ϵ * ForwardDiff.gradient(x -> -logpdf(U, x), q)
        end
    end

    # finish with a half step for momentum
    p -= (ϵ / 2.) * ForwardDiff.gradient(x -> -logpdf(U, x), q)

    # negate momentum so metropolis proposal is symmetric
    p = -p

    # calculate hamiltonian for new and old momentum and position
    current_U = -logpdf(U, current_q)
    current_K = sum(current_p .^ 2) / 2
    proposed_U = -logpdf(U, q)
    proposed_K = sum(p .^ 2) / 2

    # accept or reject proposal with metropolis
    if rand(Uniform(), 1)[1] < exp(current_U - proposed_U + current_K - proposed_K)
        return q # accept
    else
        return current_q # reject
    end
end

function bivariate_HMC(μ, Σ, S=20)
    # allocate chain
    chain = zeros(S, length(μ))

    # initialize logpdf
    U = MvNormal(μ, Σ)

    for s in 2:S
        chain[s, :] = metropolis(U, 0.18, 20, chain[s-1, :])
    end

    return chain
end

using Gadfly

# figure 4
srand(1)
chain = bivariate_HMC([0., 0.], [1.0 0.95; 0.95 1.0])
plot(x=chain[:, 1], y=chain[:, 2])

# figure 5
srand(1)
@elapsed chain = bivariate_HMC([0., 0.], [1.0 0.95; 0.95 1.0], 1000)
plot(y=chain[:, 1])
