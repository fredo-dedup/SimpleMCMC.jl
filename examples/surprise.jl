######### Model using abs() and max()  ###########

using SimpleMCMC
using DataFrames

# define model
model = quote
	resid = 0.2*(x*x+y*y-4)
	resid *= 2*( (abs(x)-0.7)^2 + (y-0.5)^2 )
	resid *= abs(x*x+y*y-1.8) + 3*max(0, y+1.0)
	resid ~ Normal(0, 0.4)
end

# run random walk metropolis 
res = simpleRWM(model, steps=100000, burnin=0, x=0., y=0.)
df = DataFrame(x=res.params[:x], y=res.params[:y])
df["method"] = "RMW"

# run Hamiltonian Monte-Carlo
res = simpleHMC(model, steps=100000, burnin=0, isteps=2, stepsize=0.5, x=0.5, y=0.5)
df2 = DataFrame(x=res.params[:x], y=res.params[:y])
df2["method"] = "HMC"

# run NUTS - HMC
res = simpleNUTS(model, steps=100000, burnin=0, x=0.5, y=0.5)   # extremely slow
df3 = DataFrame(x=res.params[:x], y=res.params[:y])
df3["method"] = "NUTS"

# save for plotting
df4 = [df, df2, df3][1:10:end, :]  # thinning by 10
write_table("c:/temp/surprise.txt", df4, '\t', '"')

# this may not work smoothly in your case..
using DThree
scatterChart(df4) | browse
