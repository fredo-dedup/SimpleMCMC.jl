######### fitting an Ornstein–Uhlenbeck process  ###########

using SimpleMCMC

# generate serie
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

# model definition 
model = quote
    tau ~ Gamma(2, 100)
    sigma ~ Gamma(2, 2)
    mu ~ Normal(0, 100)

    fac = exp(- 1 / tau)
    resid = x[2:end] - x[1:end-1] * fac - mu * (1. - fac)
    resid ~ Normal(0, sigma)
end

# finding the MLE with solving functions
simpleAGD(model, maxiter=1000, tau=17, mu=1., sigma=1.)  
# convergence on tau is slow, not concave enough on this dimension ?

simpleNM(model, maxiter=1000, tau=30, mu=1., sigma=1.) # more robust on this model

# run random walk metropolis (10000 steps, setting initial values)
res = simpleRWM(model, steps=10000, tau=1., mu=0., sigma=1.)

[ (mean(v), std(v)) for v in values(res.params)]

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 5 inner steps, 0.002 inner step size)
res = simpleHMC(model, steps=10000, burnin=1000, tau=1., mu=0., sigma=1.) # doesn't work as well as RMW
res = simpleHMC(model, steps=10000, burnin=1000, tau=20., mu=10., sigma=0.1) # better with centered init values

# now by clamping 2 parameters out of 3
mu=10.
sigma=0.1
res = simpleHMC(model, steps=10000, burnin=1000, tau=20.) 

[ (mean(v), std(v)) for v in values(res.params)]

