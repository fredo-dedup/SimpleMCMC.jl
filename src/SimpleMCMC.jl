module SimpleMCMC


include("parsing.jl") #  include model processing functions		
include("diff.jl") #  include derivatives definitions

import Base.show

export simpleRWM, simpleHMC, simpleNUTS
export buildFunction, buildFunctionWithGradient


# naming conventions
const ACC_SYM = :__acc
const PARAM_SYM = :__beta
const LLFUNC_NAME = "loglik"
const TEMP_NAME = "tmp"
const DERIV_PREFIX = "d"


# returned result structure
type MCMCRun
    acceptRate::Float64
    time::Float64
    steps::Integer
    burnin::Integer
    samples::Integer
    ess::NTuple{2,Float64}
    essBySec::NTuple{2,Float64}
    loglik::Vector
    accept::Vector
    params::Dict
end
MCMCRun(steps::Integer, burnin::Integer) = 
	MCMCRun(NaN, NaN, steps, burnin, steps-burnin, (NaN, NaN), (NaN, NaN), [], [], Dict())

function show(io::IO, x::MCMCRun)
	for v in keys(x.params)
		print("$v$(size(x.params[v])) ")
	end
	println()
	print("Time $(round(x.time,1)) sec., ")
	print("Accept rate $(round(100*x.acceptRate,1)) %, ")
	print("Eff. samples $(map(iround, x.ess)), ")
	println("Eff. samples per sec. $(map(iround, x.essBySec))")
end





##########################################################################################
#   Random Walk Metropolis function
##########################################################################################

function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
	const local target_accept = 0.234
	local ll_func, nparams, pmap

	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	ll_func, nparams, pmap = buildFunction(model) # build function, count the number of parameters

	beta = setInit(init, nparams) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	#  first calc
	__lp = ll_func(beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	S = eye(nparams) # initial value for jump scaling matrix
 	for i in 1:steps	
		jump = randn(nparams)
		oldbeta = copy(beta)
		beta += S * jump

 		old__lp, __lp = __lp, ll_func(beta) 

 		alpha = min(1, exp(__lp - old__lp))
		if rand() > alpha # reject, go back to previous state
			__lp, beta = old__lp, oldbeta
		end
		
		i > burnin ? addToRes!(res, pmap, i-burnin, __lp, old__lp != __lp, beta) : nothing

		#  Adaptive scaling using R.A.M. method
		eta = min(1, nparams*i^(-2/3))
		# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
		SS = (jump * jump') / dot(jump, jump) * eta * (alpha - target_accept)
		SS = S * (eye(nparams) + SS) * S'
		S = chol(SS)
		S = S'
	end

	# print a few stats
	calcStats!(res)

	res
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
	
	ll_func, nparams, pmap = buildFunctionWithGradient(model) # build function, count the number of parameters
	beta = setInit(init, nparams) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	#  first calc
	llik, grad = ll_func(beta) 
	assert(isfinite(llik), "Initial values out of model support, try other values")

	#  main loop
	local jump, beta0, llik0, jump0
 	for i in 1:steps  #i=1
 		local j

 		jump = randn(nparams)
 		jump0 = copy(jump)
		beta0 = copy(beta)
		llik0 = copy(llik)

		j=1
		while j <= isteps && isfinite(llik)
			jump += stepsize/2. * grad
			beta += stepsize * jump
			llik, grad = ll_func(beta)
			jump += stepsize/2. * grad
			j +=1
		end

		# revert to initial values if new is not good enough
		if rand() > exp((llik - dot(jump,jump)/2.0) - (llik0 - dot(jump0,jump0)/2.0))
			llik, beta = llik0, beta0
		end

		i > burnin ? addToRes!(res, pmap, i-burnin, llik, llik0 != llik, beta) : nothing
	end

	calcStats!(res) # calculate a few stats on the run
	res
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
    local beta0, r0, llik0, grad0, H0  # starting state of each loop
    local u_slice 
	
	# beta0 = ones(30)
	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	ll_func, nparams, pmap = buildFunctionWithGradient(model) # build function, count the number of parameters
	beta0 = setInit(init, nparams) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	# first calc
	llik0, grad0 = ll_func(beta0)
	assert(isfinite(llik0), "Initial values out of model support, try other values")

	# Leapfrog step
	function leapFrog(beta, r, grad, ve, ll)
		local llik

		# println("IN --- beta=$beta, r=$r, grad=$grad, ve=$ve")

		r += grad * ve / 2.
		beta += ve * r
		llik, grad = ll(beta) 
		r += grad * ve / 2.

		# println("OUT --- beta=$beta, r=$r, grad=$grad, llik=$llik")

		return beta, r, llik, grad
	end

	# find initial value for epsilon
	epsilon = 1.
	jump = randn(nparams)
	beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, ll_func)

	ratio = exp(llik1-dot(jump1, jump1)/2. - (llik0-dot(jump,jump)/2.))
	a = 2*(ratio>0.5)-1.
	while ratio^a > 2^-a
		epsilon = 2^a * epsilon
		beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, ll_func)
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
		local dummy, H1
		const deltamax = 100

		if j == 0
			beta1, r1, llik1, grad1 = leapFrog(beta, r, grad, dir*epsilon, ll)
			H1 = llik1 - dot(r1,r1)/2.0
			n1 = ( u_slice <= H1 ) + 0 
			s1 = u_slice < ( deltamax + H1 )

			return beta1, r1, grad1,  beta1, r1, grad1,  beta1, llik1, grad1,  n1, s1, 
				min(1., exp(H1 - H0)), 1
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
 		H0 = llik0 - dot(r0,r0)/2.

 		u_slice  = log(rand()) + H0 # use log ( != paper) to avoid underflow
 		
 		beta = copy(beta0)
 		betap = betam = beta0

 		grad = copy(grad0)
 		gradp = gradm = grad0

 		rp = rm = r0
 		llik = llik0

 		# inner loop
 		j, n = 0, 1
 		s = true
 		while s && j < 8
 			dir = (rand() > 0.5) * 2. - 1.
 			if dir == -1
 				betam, rm, gradm,  dummy, dummy, dummy,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
 					buildTree(betam, rm, gradm, dir, j, ll_func)
 			else
 				dummy, dummy, dummy,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
 					buildTree(betap, rp, gradp, dir, j, ll_func)
 			end
 			if s1 && rand() < n1/n  # accept and set new beta
 				# println("    accepted s1=$s1, n1/n=$(n1/n)")
 				beta = beta1
 				llik = llik1
 				grad = grad1
 			end
 			n += n1
 			j += 1
 			s = s1 && (dot((betap-betam), rm) >= 0.0) && (dot((betap-betam), rp) >= 0.0)
 			# println("---  dir=$dir, j=$j, n=$n, s=$s, s1=$s1, alpha/nalpha=$(alpha/nalpha)")
 		end
 		
 		# epsilon adjustment
 		if i <= nadapt  # warming up period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^(-kappa)) * lebar
			epsilon = exp(le)
			# println("alpha=$alpha, nalpha=$nalpha, hbar=$hbar, \n le=$le, lebar=$lebar, epsilon=$epsilon")
		else # post warm up, keep same epsilon
			epsilon = exp(lebar)
		end

		# println(llik, beta)
		i > burnin ? addToRes!(res, pmap, i-burnin, llik, beta != beta0, beta) : nothing

		beta0 = beta
		grad0 = grad
		llik0 = llik
	end

	calcStats!(res)
	res
end

simpleNUTS(model::Expr, steps::Integer) = simpleNUTS(model, steps, min(steps-1, div(steps,2)))
simpleNUTS(model::Expr, steps::Integer, burnin::Integer) = simpleNUTS(model, steps, burnin, 1.0)



##########################################################################################
#   Common functionality
##########################################################################################


### checks consistency of steps and burnin steps
function checkSteps(steps, burnin)
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
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

### sets the result structure
function setRes(steps, burnin, pmap)
	res = MCMCRun(steps, burnin)
	res.accept = fill(NaN, res.samples) 
	res.loglik = fill(NaN, res.samples)
	for p in pmap
		res.params[p.sym] = fill(NaN, tuple([p.size, res.samples]...))
	end

	res
end

### adds a sample to the result structure
function addToRes!(res::MCMCRun, pmap::Vector{MCMCParams}, index::Integer, ll::Float64, accept::Bool, beta::Vector{Float64})
	res.loglik[index] = ll
	res.accept[index] = accept
	for p in pmap
		str = prod(p.size)
		res.params[p.sym][((index-1)*str+1):(index*str)] = beta[ p.map]  #[1]:p.map[2]]
	end
end

##### stats calculated after a full run
function calcStats!(res::MCMCRun)
	res.time = toq()
	res.acceptRate = mean(res.accept)

	function ess(serie::Vector)
		fac = abs(cov(serie[2:end], serie[1:(end-1)])) / var(serie)
		length(serie) * max(0., 1. - fac) / (1. + fac)
	end
	# note the absolute value around the covar to penalize anti-correlation the same as
	# correlation. This will also ensure that ess is <= number of samples

	res.ess = (Inf, -Inf)
	for p in res.params 
		sa = p[2]
		pos = 1:stride(sa, ndims(sa)):length(sa)
		while max(pos) <= length(sa)
			en = ess(sa[pos])
			res.ess = (min(res.ess[1], en), max(res.ess[2], en))
			pos += 1
		end
	end
	res.essBySec = map(x->x/res.time, res.ess)
end	



end # module end

