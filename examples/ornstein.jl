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

# model definition (note : rescaling on tau by x1000, and mu by x10)
model = quote
    tau ~ Uniform(0, 0.1)
    sigma ~ Uniform(0, 2)
    mu ~ Uniform(0, 2)

    fac = exp(- 1 / 1000tau)
    resid = x[2:end] - x[1:end-1] * fac - 10mu * (1. - fac)
    resid ~ Normal(0, sigma)
end

# let's find the MLE
simpleAGD(model, maxiter=1000, tau=0.05, mu=1., sigma=1.)
# tau= 0.020431633947170548  mu= 0.9944448337448664  sigma= 0.10001446640703672 , looks ok

simpleNM(model, tau=.05, mu=1., sigma=1.)  
# tau= 0.9947504351836987  mu= 0.9941563361988168  sigma= 0.9945514556258659 
# not what we're looking for, let's improve precision

simpleNM(model, precision=1e-5, maxiter=1000, tau=.05, mu=1., sigma=1.)  

# run random walk metropolis (10000 steps, 1000 for burnin, setting initial values)
res = simpleRWM(model, 10000, 1000, [1., 0.1, 1.])

[mean(res.params[:mu]) std(res.params[:mu]) ] * 10
[mean(res.params[:tau]) std(res.params[:tau]) ] * 1000
[mean(res.params[:sigma]) std(res.params[:sigma]) ]

# run Hamiltonian Monte-Carlo (10000 steps, 1000 for burnin, 5 inner steps, 0.002 inner step size)
res = simpleHMC(model, 10000, 1000, [1., 0.1, 1.], 5, 0.002)

# run NUTS - HMC (1000 steps, 500 for burnin)
res = simpleNUTS(model, 10000, 1000, [1., 0.1, 1.])

res.misc[:jmax] # check # of doublings in NUTS algo
res.misc[:epsilon] # check epsilon



##########


model2 = quote
    sigma ~ Gamma(2,2)
    mu ~ Gamma(2,20)
    tau ~ Uniform(0,100)

    fac = exp(- 1 / tau)
    resid = x[2:end] - x[1:end-1] * fac - mu * (1. - fac)
    resid ~ Normal(0, sigma)
end

# let's find the MLE
simpleAGD(model2, maxiter=1000, tau=10., mu=10., sigma=0.1)
# tau= 0.020431633947170548  mu= 0.9944448337448664  sigma= 0.10001446640703672 , looks ok

simpleNM(model2, precision=1e-4, maxiter=1000, tau=10., mu=10., sigma=1.)  
# tau= 0.9947504351836987  mu= 0.9941563361988168  sigma= 0.9945514556258659 
# not what we're looking for, let's improve precision

simpleNM(model, precision=1e-5, maxiter=1000, tau=.05, mu=1., sigma=1.)  



