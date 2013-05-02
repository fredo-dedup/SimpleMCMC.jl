######### Distribution with abs() and max()  ###########

using SimpleMCMC

# define model
model = quote
	x::real
	y::real

	resid = 0.2*(x*x+y*y-4)
	resid *= 2*( (abs(x)-0.7)^2 + (y-0.5)^2 )
	resid *= abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.4)
end


# run random walk metropolis 
res = simpleRWM(model, 20000, 0)
writedlm("c:/temp/circle.txt", [res.params[:x] res.params[:y]])

# run Hamiltonian Monte-Carlo
res = simpleHMC(model, 10000, 10, 0.1)
writedlm("c:/temp/circle.2.txt", [res.params[:x] res.params[:y]])

# run NUTS - HMC
res = simpleNUTS(model, 10000, 0)   # extremely slow
writedlm("c:/temp/circle.3.txt", [res.params[:x] res.params[:y]])


