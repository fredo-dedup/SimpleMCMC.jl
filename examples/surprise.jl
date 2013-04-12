######### Distribution with abs() and max()  ###########

include("../src/SimpleMCMC.jl")

# define model
model = quote
	x::real
	y::real

	resid = 0.2*(x*x+y*y-4)
	resid *= 2*( (abs(x)-0.7)^2 + (y-0.5)^2 )
	resid *= abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.4)
end


# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 20000, 0)
writedlm("c:/temp/circle.txt", [res.params[:x] res.params[:y]])

# run Hamiltonian Monte-Carlo (1000 steps, 500 for burnin, 2 inner steps, 0.05 inner step size)
res = SimpleMCMC.simpleHMC(model, 10000, 10, 0.1)
writedlm("c:/temp/circle.2.txt", [res.params[:x] res.params[:y]])

# run NUTS - HMC (1000 steps, 500 for burnin)
res = SimpleMCMC.simpleNUTS(model, 10, 0)   # extremely slow
writedlm("c:/temp/circle.3.txt", [res.params[:x] res.params[:y]])


