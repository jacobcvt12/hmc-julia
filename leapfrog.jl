# Discretizing Hamilton's equations
using Gadfly
iters = 21

p = Array{Float64}(iters)
q = Array{Float64}(iters)

m = 1.

p[1] = 1.
q[1] = 0.

# euler's method, stepsize 0.3
epsilon = 0.3

for i in 2:iters
    p[i] = p[i-1] - epsilon * q[i-1]
    q[i] = q[i-1] + epsilon * p[i-1] / m
end

plot(x=q, y=p)

# modified euler's method, stepsize 0.3
epsilon = 0.3

for i in 2:iters
    p[i] = p[i-1] - epsilon * q[i-1]
    q[i] = q[i-1] + epsilon * p[i] / m
end

plot(x=q, y=p)

# modified euler's method, stepsize 0.3
epsilon = 0.3

for i in 2:iters
    p[i] = p[i-1] - epsilon * q[i-1]
    q[i] = q[i-1] + epsilon * p[i] / m
end

plot(x=q, y=p)

# leapfrog method, stepsize 0.3
epsilon = 0.3

for i in 2:iters
    p_eps_2 = p[i-1] - epsilon * q[i-1] / 2.
    q[i] = q[i-1] + epsilon * p_eps_2 / m
    p[i] = p_eps_2 - epsilon * q[i] / 2.
end

plot(x=q, y=p)

# leapfrog method, stepsize 1.2
epsilon = 1.2

for i in 2:iters
    p_eps_2 = p[i-1] - epsilon * q[i-1] / 2.
    q[i] = q[i-1] + epsilon * p_eps_2 / m
    p[i] = p_eps_2 - epsilon * q[i] / 2.
end

plot(x=q, y=p)
