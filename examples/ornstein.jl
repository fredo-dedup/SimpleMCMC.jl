######### fitting an Ornstein–Uhlenbeck process  ###########

include("../src/SimpleMCMC.jl")

# generate serie
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
    tau::real
    sigma::real

    tau ~ Uniform(0, 0.1)
    sigma ~ Uniform(0, 2)
    mu ~ Uniform(0, 2)

    fac = exp(- 0.001 / tau)
    resid = x[2:end] - x[1:end-1] * fac - 10 * mu * (1. - fac)
    resid ~ Normal(0, sigma)
end


# run random walk metropolis (10000 steps, 1000 for burnin, setting initial values)
res = SimpleMCMC.simpleRWM(model, 10000, 1000, [1., 0.1, 1.])

[mean(res.params[:mu]) std(res.params[:mu]) ] * 10
[mean(res.params[:sigma]) std(res.params[:sigma]) ]
[mean(res.params[:tau]) std(res.params[:tau]) ] * 1000

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 5 inner steps, 0.002 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, [1., 0.1, 1.], 5, 0.002)

# run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 10000, 1000, [1., 0.1, 1.])

res.misc[:jmax] # check # of doublings in NUTS algo
res.misc[:epsilon] # check epsilon

