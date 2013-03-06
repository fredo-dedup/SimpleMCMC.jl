######### fitting an Ornstein–Uhlenbeck process  ###########

# generate serie
duration = 1000  # 1000 time steps
mu0 = 2.  # target value
tau0 = 50  # convergence time
sigma0 = 0.1  # noise term

x = fill(NaN, duration)
x[1] = 1.
for i in 2:duration
	x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
end

# model definition
model = quote
    mu::real
    tau::real
    sigma::real

    tau ~ Weibull(2, 100)
    sigma ~ Uniform(0, 1)
    mu ~ Weibull(2, 1)

    fac = exp(- 1 / tau)
    resid = x[2:end] - x[1:end-1] .* fac - mu * (1. - fac)
    resid ~ Normal(0, sigma)
end


# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000, 1000)

sum(res.params[:mu],1) / res.samples  # mu samples mean
sum(res.params[:sigma],1) / res.samples # sigma samples mean
sum(res.params[:tau],1) / res.samples # beta samples mean

# # run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 2 inner steps, 0.1 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 1000, 2, 0.005)

# # run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 1000, 0)  # very slow  (bug ?)


