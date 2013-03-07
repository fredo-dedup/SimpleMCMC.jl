######### fitting an Ornstein–Uhlenbeck process  ###########

include("../src/SimpleMCMC.jl")

# generate serie
duration = 1000  # 1000 time steps
mu0 = 10.  # target value
tau0 = 20  # convergence time
sigma0 = 1.  # noise term

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
    sigma ~ Uniform(0, 10)
    mu ~ Weibull(2, 10)

    fac = exp(- 1 / tau)
    resid = x[2:end] - x[1:end-1] * fac - mu * (1. - fac)
    resid ~ Normal(0, sigma)
end


# run random walk metropolis (10000 steps, 1000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000, 1000)

(mean(res.params[:mu]), std(res.params[:mu]))
(mean(res.params[:sigma]), std(res.params[:sigma]))
(mean(res.params[:tau]), std(res.params[:tau]))

# # run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 2 inner steps, 0.1 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 5000, 1, 0.01)

# # run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 100, 0)  # very slow  (bug ?)


