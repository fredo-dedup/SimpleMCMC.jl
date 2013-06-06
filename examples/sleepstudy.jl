###### random effects example  #######

using RDatasets
sleep = data("lme4", "sleepstudy")

using SimpleMCMC

## step 1 : without random effects

days = vector(sleep[:Days])
reaction = vector(sleep[:Reaction])

model = quote
	r_bar = intercept + slope * days
	resid = r_bar - reaction
	resid ~ Normal(0,1)
end

simpleAGD(model, intercept=200., slope=1.0)
res = simpleNUTS(model, steps=1e4, intercept=250., slope=10.0)
for d in values(res.params); println(round(mean(d),1), " +-", round(std(d),2)) ; end
# 251.4 +-2.75
# 10.5 +-0.51

## step 2 : with random effects by subject for intercept and slope
subject = vector(sleep[:Subject])
subind = sleep[:Subject].refs
nsubjects = max(subind)

submatrix = [ subind[i]==j ? 1.0 : 0.0 for i in 1:length(subject), j in 1:nsubjects ]

rand_slope = ones(nlevels)

model2 = quote
	r_bar = (intercept + submatrix * rand_int) + (slope + submatrix * rand_slope) .* days
	rand_int ~ Normal(0,100)
	rand_slope ~ Normal(0,100)
	resid = r_bar - reaction
	resid ~ Normal(0,10)
end

parinit = {	:intercept=>250., 
			:slope=>10.0, 
			:rand_int=>zeros(nlevels), 
			:rand_slope=>zeros(nlevels)}

res = simpleAGD(model2, maxiter=1000; parinit...)
simpleNM(model2, intercept=250., slope=10.0, rand_int=zeros(nlevels), rand_slope=zeros(nlevels))

model3 = quote
	r_bar = (submatrix * rand_int) + (submatrix * rand_slope) .* days
	resid = r_bar - reaction
	resid ~ Normal(0,10)
end

parinit = {	:rand_int=>zeros(nlevels), 
			:rand_slope=>zeros(nlevels)}

res = simpleAGD(model3, maxiter=100; parinit...)

for d in values(res.params); println(round(mean(d),1), " +-", round(std(d),2)) ; end
mean(res.params[:rand_int])
stdm(res.params[:rand_int], 251.403)

mean(res.params[:rand_slope])
stdm(res.params[:rand_slope], 10.467)

quantile(res.params[:rand_slope], 0.5)

res = simpleNM(model2, intercept=250., slope=10.0, rand_int=zeros(nlevels), rand_slope=zeros(nlevels))




simpleNUTS(model3, rand_int=zeros(nlevels), rand_slope=zeros(nlevels))
