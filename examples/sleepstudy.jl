###### random effects example  #######

using RDatasets
using SimpleMCMC

sleep = data("lme4", "sleepstudy")  # load data

## step 1 : without random effects
days = vector(sleep[:Days])
reaction = vector(sleep[:Reaction])

model = quote
	r_bar = intercept + slope * days
	resid = r_bar - reaction
	resid ~ Normal(0,1)
end

# find MLE
simpleAGD(model, intercept=200., slope=1.0)

# Run MCMC sampling
res = simpleNUTS(model, steps=1e4, intercept=250., slope=10.0)
[ "$(round(mean(v),1)) +-$(round(std(v),2))" for v in values(res.params) ]



## step 2 : with random effects by subject for intercept and slope
subind = sleep[:Subject].refs
nsubjects = max(subind)
submatrix = [ subind[i]==j ? 1.0 : 0.0 for i in 1:length(subind), j in 1:nsubjects ]

model2 = quote
	r_bar = (intercept + submatrix * rand_int) + (slope + submatrix * rand_slope) .* days
	rand_int ~ Normal(0,100)
	rand_slope ~ Normal(0,100)
	resid = r_bar - reaction
	resid ~ Normal(0,10)
end

# let's put initial values in a Dict to simplify function calls
parinit = {	:intercept=>250., 
			:slope=>10.0, 
			:rand_int=>zeros(nsubjects), 
			:rand_slope=>zeros(nsubjects)}

res = simpleAGD(model2, maxiter=1000; parinit...)

[ res.params[:rand_int] res.params[:rand_slope]]
std(res.params[:rand_int])
std(res.params[:rand_slope])
