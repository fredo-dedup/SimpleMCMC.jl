module SimpleMCMC

require("Distributions")

import 	Distributions.logpdf, Distributions.logpmf
import  Distributions.DiscreteDistribution, Distributions.ContinuousDistribution
import 	Distributions.Normal, 
		Distributions.Uniform, 
		Distributions.Weibull,
		Distributions.Bernoulli

include("parsing.jl") #  include model processing functions		
include("diff.jl") #  include derivatives definitions

export simpleRWM, simpleHMC, simpleNUTS
export buildFunction, buildFunctionWithGradient

# naming conventions
const ACC_SYM = :__acc
const PARAM_SYM = :__beta
const LLFUNC_SYM = :__loglik
const TEMP_NAME = "tmp"
const DERIV_PREFIX = "d"


##########################################################################################
#   Random Walk Metropolis function
##########################################################################################

function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
	const local target_accept = 0.234
	# steps=100; burnin=10; init=1

	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	(ll_func, nparams) = buildFunction(model) # build function, count the number of parameters
	Main.eval(ll_func) # create function (in Main !)

	beta = setInit(init, nparams) # build the initial values


	#  first calc
	__lp = Main.__loglik(beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

	S = eye(nparams) # initial value for jump scaling matrix
 	for i in 1:steps	
		jump = 0.1 * randn(nparams)
		oldbeta, beta = beta, beta + S * jump

 		old__lp, __lp = __lp, Main.__loglik(beta) 

 		alpha = min(1, exp(__lp - old__lp))
		if rand() > exp(__lp - old__lp)
			__lp, beta = old__lp, oldbeta
		end
		draws[i, :] = vcat(__lp, (old__lp != __lp), beta) 

		#  Adaptive scaling using R.A.M. method
		eta = min(1, nparams*i^(-2/3))
		# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
		SS = (jump * jump') / dot(jump, jump) * eta * (alpha - target_accept)
		SS = S * (eye(nparams) + SS) * S'
		S = chol(SS)
		S = S'
	end

	# print a few stats on the run
	runStats(draws[(burnin+1):steps,:], toq())

	draws[(burnin+1):steps, :]
end

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, min(steps-1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########################################################################################
#   Canonical HMC function
##########################################################################################

function simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, isteps::Integer, stepsize::Float64)
	local ll_func, nparams
	local beta, llik, grad
	local draws

	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	(ll_func, nparams) = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
	Main.eval(ll_func) # create function (in Main !)

	beta = setInit(init, nparams) # build the initial values

	#  first calc
	llik, grad = Main.__loglik(beta) # eval(llcall)
	assert(isfinite(llik), "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

 	for i in 1:steps  #i=1
 		local jump, beta0, llik0, jump0
 		local j
 		
 		jump = randn(nparams)
		beta0 = beta
		llik0 = llik
		jump0 = jump

		j=1
		while j <= isteps && isfinite(llik)
			jump += stepsize/2. * grad
			beta += stepsize * jump
			llik, grad = Main.__loglik(beta)
			jump += stepsize/2. * grad
			j +=1
		end

		# revert to initial values if new is not good enough
		if rand() > exp((llik - dot(jump,jump)/2.0) - (llik0 - dot(jump0,jump0)/2.0))
			llik, beta = llik0, beta0
		end
		draws[i, :] = vcat(llik, llik0 != llik, beta) 

	end

	# print a few stats on the run
	runStats(draws[(burnin+1):steps,:], toq())

	draws[(burnin+1):steps, :]
end

simpleHMC(model::Expr, steps::Integer, isteps::Integer, stepsize::Float64) = 
simpleHMC(model, steps, min(steps-1, div(steps,2)), isteps, stepsize)
simpleHMC(model::Expr, steps::Integer, burnin::Integer, isteps::Integer, stepsize::Float64) = 
	simpleHMC(model, steps, burnin, 1.0, isteps, stepsize)


##########################################################################################
#   NUTS sampler function
##########################################################################################

function simpleNUTS(model::Expr, steps::Integer, burnin::Integer, init::Any)
    local epsilon
    local beta0, r0, llik0, grad0  # starting state of each loop
    local u_slice 
	
	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	ll_func, nparams = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
	Main.eval(ll_func) # create function (in Main !)

	beta0 = setInit(init, nparams) # build the initial values

	# prepare storage matrix for samples
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

	# first calc
	llik0, grad0 = Main.__loglik(beta0)
	assert(isfinite(llik0), "Initial values out of model support, try other values")

	# Leapfrog step
	function leapFrog(beta, r, grad, ve, ll)
		local llik

		r += grad * ve / 2.
		beta += ve * r
		llik, grad = ll(beta) 
		r += grad * ve / 2.
	# println("        ++++  $llik  $beta   $grad   +++++")

		return beta, r, llik, grad
	end

	# find initial value for epsilon
	epsilon = 1.
	jump = randn(nparams)
	beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, Main.__loglik)

	ratio = exp(llik1-dot(jump1, jump1)/2. - (llik0-dot(jump,jump)/2.))
	a = 2*(ratio>0.5)-1.
	while ratio^a > 2^-a
		epsilon = 2^a * epsilon
		beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, Main.__loglik)
		ratio = exp(llik1-dot(jump1, jump1)/2. - (llik0-dot(jump,jump)/2.))
	end
	println("starting epsilon = $epsilon")

	### adaptation parameters
	const delta = 0.7  # target acceptance
	const nadapt = 1000  # nb of steps to adapt epsilon
	const gam = 0.05
	const kappa = 0.75
	const t0 = 10
	### adaptation inital values
	hbar = 0.
	mu = log(10*epsilon)
	lebar = 0.0

	# buidtree function
	function buildTree(beta, r, grad, dir, j, ll)
		local beta1, r1, llik1, grad1, n1, s1, alpha1, nalpha1
		local beta2, r2, llik2, grad2, n2, s2, alpha2, nalpha2
		local betam, rm, gradm, betap, rp, gradp
		local dummy
		const deltamax = 1000

		if j == 0
			beta1, r1, llik1, grad1 = leapFrog(beta, r, grad, dir*epsilon, ll)
			n1 = (u_slice <= ( llik1 - dot(r1,r1)/2.0 )) + 0 
			s1 = u_slice < ( deltamax + llik1 - dot(r1,r1)/2.0 )

			return beta1, r1, grad1,  beta1, r1, grad1,  beta1, llik1, grad1,  n1, s1, 
				min(1., exp(llik1 - dot(r1,r1)/2. - llik0 + dot(r0,r0)/2.)), 1
		else
			betam, rm, gradm,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha1, nalpha1 = 
				buildTree(beta, r, grad, dir, j-1, ll)
			if s1 
				if dir == -1
					betam, rm, gradm,  dummy, dummy, dummy,  beta2, llik2, grad2,  n2, s2, alpha2, nalpha2 = 
						buildTree(betam, rm, gradm, dir, j-1, ll)
	 			else
	 				dummy, dummy, dummy,  betap, rp, gradp,  beta2, llik2, grad2,  n2, s2, alpha2, nalpha2 = 
	 					buildTree(betap, rp, gradp, dir, j-1, ll)
	 			end
	 			if rand() <= n2/(n2+n1)
	 				beta1 = beta2
	 				llik1 = llik2
	 				grad1 = grad2
	 			end
	 			alpha1 += alpha2
	 			nalpha1 += nalpha2
	 			s1 = s2 && (dot((betap-betam), rm) >= 0.0) && (dot((betap-betam), rp) >= 0.0)
	 			n1 += n2
	 		end

	 		return betam, rm, gradm,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha1, nalpha1
		end
	end

	### main loop
 	for i in 1:steps  # i=1
 		local dummy, alpha, nalpha

 		r0 = randn(nparams)
 		u_slice  = log(rand()) + llik0 - dot(r0,r0)/2.0 # use log ( != paper) to avoid underflow
 		beta = betap = betam = beta0
 		grad = gradp = gradm = grad0
 		rp = rm = r0
 		llik = llik0

 		# inner loop
 		j, n = 0, 1
 		s = true
 		while s
 			dir = (randn() > 0.5) * 2. - 1.
 			if dir == -1
 				betam, rm, gradm,  dummy, dummy, dummy,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
 					buildTree(betam, rm, gradm, dir, j, Main.__loglik)
 			else
 				dummy, dummy, dummy,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
 					buildTree(betap, rp, gradp, dir, j, Main.__loglik)
 			end
 			# println("=== loop ($i/$j), dir $dir, s1 : $s1, n1/n : $n1 / $n")
 			if s1 && rand() < n1/n  # accept and set new beta
 				beta = beta1
 				llik = llik1
 				grad = grad1
 			end
 			n += n1
 			j += 1
 			s = s1 && (dot((betap-betam), rm) >= 0.0) && (dot((betap-betam), rp) >= 0.0)
 		end
 		
 		# epsilon adjustment
 		if i <= nadapt  # warming up period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^-kappa) * lebar
			epsilon = exp(le)
			# println("epsilon (adapt) = $epsilon")
		else # post warm up, keep same epsilon
			epsilon = exp(lebar)
		end

		draws[i, :] = vcat(llik, (beta != beta0), beta)
		beta0 = beta
		grad0 = grad
		llik0 = llik
	end


	runStats(draws[(burnin+1):steps,:], toq())

	draws[(burnin+1):steps, :]
end

simpleNUTS(model::Expr, steps::Integer) = simpleNUTS(model, steps, min(steps-1, div(steps,2)))
simpleNUTS(model::Expr, steps::Integer, burnin::Integer) = simpleNUTS(model, steps, burnin, 1.0)



##########################################################################################
#   Common functionality
##########################################################################################


### checks consistency of steps and burnin steps
function checkSteps(steps, burnin)
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > 0, "Steps ($steps) should be > 0")
end

### sets inital values from 'init' given as parameter
function setInit(init, nparams)
	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		return init
	elseif typeof(init) <: Real
		return ones(nparams) * init
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end
end

##### stats calculated after a full run
function runStats(res::Matrix{Float64}, delay::Float64)
	nsamp = size(res,1)
	nvar = size(res,2)

	print("$(round(delay,1)) sec., ")

	essfac(serie::Vector) = abs(cov(serie[2:end], serie[1:(end-1)])) / var(serie)
	# note the absolute value around the covar to penalize anti-correlation the same as
	# correlation. This will also ensure that ess is <= number of samples

	ess = [ essfac(res[:,i])::Float64 for i in 3:nvar ]
	ess = nsamp .* max(0., 1.-ess) ./ (1.+ess)
	if nvar==3
		print("effective samples $(round(ess[1])), ")
		println("effective samples by sec $(round(ess[1]/delay))")
	else
		print("effective samples $(round(min(ess))) to $(round(max(ess))), ")
		println("effective samples by sec $(round(min(ess)/delay)) to $(round(max(ess)/delay))")
	end
end	




# module end
end

