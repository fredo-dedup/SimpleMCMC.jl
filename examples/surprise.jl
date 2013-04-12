
include("../src/SimpleMCMC.jl")

# define model
model = quote
	x::real
	y::real

	resid = 0.1*(x*x+y*y-4)
	resid *= 4 * ((abs(x)-0.5)^2 + 2*(y-0.7)^2)
	resid *= 0.5 * abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.5)  
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 50000, 0)
writedlm("c:/temp/circle.txt", [res.params[:x] res.params[:y]])

res.acceptRate  # acceptance rate
[ sum(res.params[:vars],2)./res.samples beta0 ] # show calculated and original coefs side by side

# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 2 inner steps, 0.05 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 10, 0.001)
writedlm("c:/temp/circle.2.txt", [res.params[:x] res.params[:y]])

# run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 10000, 0)
writedlm("c:/temp/circle.3.txt", [res.params[:x] res.params[:y]])


