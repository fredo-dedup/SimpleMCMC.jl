include("../src/SimpleMCMC.jl")

srand(1)
duration = 1000  # 1000 time steps
mu0 = 10.  # target value
tau0 = 20  # convergence time
sigma0 = 0.1  # noise term

x = fill(NaN, duration)
x[1] = 1.
for i in 2:duration
	x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
end

# model definition (note : rescaling on tau and mu)
model = quote
    mu::real
    # tau::real
    itau::real
    sigma::real

    # tau ~ Gamma(2, 0.1)
    itau ~ Gamma(2, 0.1)
    sigma ~ Gamma(2, 1)
    mu ~ Gamma(2, 1)

    # fac = exp(- 0.001 / tau)
    fac = exp(- itau)
    resid = x[2:end] - x[1:end-1] * fac - 10 * mu * (1. - fac)
    resid ~ Normal(0, sigma)
end



# julia> zb1 = zb0 + grady / (theta0 * L0)
# 3-element Float64 Array:
#  0.999419 
#  0.0248938
#  4.1756e-9


