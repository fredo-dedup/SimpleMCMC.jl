######### Model using abs() and max()  ###########

using SimpleMCMC

# define model
model = quote
	resid = 0.2*(x*x+y*y-4)
	resid *= 2*( (abs(x)-0.7)^2 + (y-0.5)^2 )
	resid *= abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.4)
end


# run random walk metropolis 
res = simpleRWM(model, steps=20000, burnin=0, x=0., y=0.)
writedlm("c:/temp/circle.txt", [res.params[:x] res.params[:y]])

# run Hamiltonian Monte-Carlo
res = simpleHMC(model, steps=20000, isteps=10, stepsize=0.1, x=1, y=1)

generateModelFunction(model, debug=true, x=0, y=0)
generateModelFunction(model, gradient=true, debug=true, x=1, y=1)

writedlm("c:/temp/circle.2.txt", [res.params[:x] res.params[:y]])

# run NUTS - HMC
res = simpleNUTS(model, steps=10000, x=0., y=0.)10000, 0)   # extremely slow
writedlm("c:/temp/circle.3.txt", [res.params[:x] res.params[:y]])


type test
	a::Float64
	b::Array{Float64}
end

test(4., [1., 3])
vector(test(4., [1., 3]))

serialize(test(4., [1., 3]))

typeof(res)


