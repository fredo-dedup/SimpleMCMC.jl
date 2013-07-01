######### Model using abs() and max()  ###########

using SimpleMCMC
using Vega

# define model
model = quote
	resid = 0.2*(x*x+y*y-4)
	resid *= 2*( (abs(x)-0.7)^2 + (y-0.5)^2 )
	resid *= abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.4)
end

# run random walk metropolis 
res = simpleRWM(model, steps=100000, burnin=0, x=0., y=0.)
plot(x=res.params[:x][1:10:end], y=res.params[:y][1:10:end], kind=:scatter)

# run Hamiltonian Monte-Carlo
res = simpleHMC(model, steps=100000, burnin=0, isteps=2, stepsize=0.13, x=0., y=0.)
plot(x=res.params[:x][1:10:end], y=res.params[:y][1:10:end], kind=:scatter)

# run NUTS - HMC
res = simpleNUTS(model, steps=100000, burnin=0, x=0., y=0.) 
plot(x=res.params[:x][1:10:end], y=res.params[:y][1:10:end], kind=:scatter)
