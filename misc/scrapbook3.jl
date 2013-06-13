
using RDatasets

require("Gadfly")
using Gadfly

require("BinDeps")
using BinDeps


Pkg.runbuildscript("Cairo")


Pkg.add("BinDeps")

draw(SVG(2inch, 2inch), compose(canvas(), rectangle(), fill("plum"), stroke(nothing)))

Gadfly.draw(SVG(6inch, 3inch), plot([sin, cos], 1, 25))



function test2(model; steps=1000, burnin=100, kwa...)
	println("model : $model, steps : $steps, burnin : $burnin")
	println("kwa : $kwa")
end


test2(45)
test(45, steps=1000, theta=[0.,1])


Pkg.add("Cairo")


Pkg.update()




Pkg.add("Winston")

using Winston

#############################################


######### fitting an Ornstein–Uhlenbeck process  ###########

using SimpleMCMC

# generate serie
begin
	srand(1)
	duration = 100  # 100 time steps
	mu0 = 10.  # target value
	tau0 = 20  # convergence time
	sigma0 = 0.1  # noise term

	x = fill(NaN, duration)
	x[1] = 1.
	for i in 2:duration
		x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
	end
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
res = simpleRWM(model, 10000, 1000, [1., 0.1, 1.])
res = simpleNUTS(model, 10000, 1000, [1., 0.1, 1.])

[mean(res.params[:mu]) std(res.params[:mu]) ] * 10
[mean(res.params[:tau]) std(res.params[:tau]) ] * 1000
[mean(res.params[:sigma]) std(res.params[:sigma]) ]

# 
sigma = 1.
tau = 0.05
mu = 1.
ll, dummy = generateModelFunction(model, [1., 0.1, 1.], false, false)
ll([mu, 0.02, sigma])
map(t->(round(t,3), ll([1.0, t, 0.1])), 0.015:0.001:0.025)

# model 2

D = [ exp(-abs(i-j)*0.001/tau) for i in 1:duration, j in 1:duration]
det(D)
I = eye(duration)
K = D + sigma*sigma*eye(duration)
det(K)
diag(K)
KC = chol(K)
det(KC)
diag(KC)
KC * KC'
KC' * KC

KCI = eye(duration) / KC

KCI * KC


det([1. 0 ; 0 1])


duration=10

sigma = 0.1
mu = 1
function ll3(mu, tau::Real, sigma)
	local const D = Float64[ abs(i-j) for i in 1:duration, j in 1:duration]
	local K = map(d -> exp(-d/(1000*tau)), D) + sigma*sigma*eye(duration)

	-0.5*(log(2pi) + 2log(det(K)) + [(x-10mu)' * (K \ (x-10mu))][1] ) #'
end
function ll4(mu, tau::Real, sigma)
	local const D = Float64[ abs(i-j) for i in 1:duration, j in 1:duration]
	local K = map(d -> exp(-d/(1000*tau)), D) * sigma * sigma * tau / 2 

	-0.5*(log(2pi) + 2log(det(K)) + ((x-10mu)' * (K \ (x-10mu)))[1] ) #'
end


K \ (x-10mu)

ll2(0.05)
ll3(0.02)

tau = 0.02

typeof(D + sigma*sigma*eye(duration))
typeof(K)

ll, dummy = SimpleMCMC.generateModelFunction(model, [1., 0.1, 1.], false, false)
ll([mu, 0.02, sigma])
map(t->ll([1., t, 0.1]), 0.015:0.001:0.025)
map(t->ll3(mu, t, sigma), linspace(0.015:0.001:0.025)
map(t->ll4(mu, t, sigma), linspace(0.015, 0.025, 11))

simpleAGD(model, [1.,0.02,0.1])
# tau= 0.01942164844902119  mu= 0.9826333623455609  sigma= 0.10082789941895108

tv =  linspace(0.01, 0.03, 1000)
rv = [ tv map(t->ll([mu, t, sigma]), tv)]
cutoff = max(rv[:,2]) - 6
pos = findfirst(rv[:,2] .> cutoff)
[ tv[pos] tv[findfirst(rv[(pos+1):end,2] .< cutoff)] ]
 # 0.0161862  0.0207107

tv =  linspace(0.05, 0.2, 100)
rv = [ tv map(t->ll3(mu, t, sigma), tv)]
cutoff = max(rv[:,2]) - 6
pos = findfirst(rv[:,2] .> cutoff)
[ tv[pos] tv[findfirst(rv[(pos+1):end,2] .< cutoff)] ]
 # 0.0666667  0.134848


map(t->(round(t,4), ll3(mu, t, sigma)), linspace(0.02, 0.2, 10))



map(ll2, 0.015:0.001:0.025)








