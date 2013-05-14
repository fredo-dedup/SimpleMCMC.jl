##########################################################################
#
#    Sampling functions
#
##########################################################################

###### returned result structure ##########
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
    misc::Dict
end
MCMCRun(steps::Integer, burnin::Integer) = 
	MCMCRun(NaN, NaN, steps, burnin, steps-burnin, (NaN, NaN), (NaN, NaN), [], [], Dict(), Dict())

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


###### HMC related structures and functions  ##########
type Sample
	beta::Vector{Float64}  	# sample position
	grad::Vector{Float64}  	# gradient
	v::Vector{Float64}  	# speed
	llik::Float64			# log likelihood 
	H::Float64				# Hamiltonian
end
Sample(beta::Vector{Float64}) = Sample(beta, Float64[], Float64[], NaN, NaN)

calc!(s::Sample, ll::Function) = ((s.llik, s.grad) = ll(s.beta))
update!(s::Sample) = (s.H = s.llik - dot(s.v, s.v)/2)
uturn(s1::Sample, s2::Sample) = dot(s2.beta-s1.beta, s1.v) < 0. || dot(s2.beta-s1.beta, s2.v) < 0.

function leapFrog(s::Sample, ve, ll)
	n = deepcopy(s)  # make a copy
	n.v += n.grad * ve / 2.
	n.beta += ve * n.v
	calc!(n, ll)
	n.v += n.grad * ve / 2.
	update!(n)

	n
end


##########################################################################################
#
#   Random Walk Metropolis function with robust adaptative scaling
#
#  ref: ROBUST ADAPTIVE METROPOLIS ALGORITHM WITH COERCED ACCEPTANCE RATE - Matti Vihola
#
##########################################################################################

function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
	const local target_accept = 0.234
	local nparams, pmap

	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	# ll_func, nparams, pmap = buildFunction(model) # build function, count the number of parameters
	nparams, pmap = buildFunction(model) # build function, count the number of parameters

	beta = setInit(init, nparams) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	#  first calc
	__lp = llmod.ll(beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	S = eye(nparams) # initial value for jump scaling matrix
 	for i in 1:steps	
 		progress(i, steps, burnin)

		jump = randn(nparams)
		oldbeta = copy(beta)
		beta += S * jump

 		old__lp, __lp = __lp, llmod.ll(beta) 

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

	calcStats!(res)

	res
end

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, min(steps-1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########################################################################################
#   Canonical HMC function
##########################################################################################

function simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, isteps::Integer, stepsize::Float64)
	local ll_func, nparams, pmap
	local state0

	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	ll_func, nparams, pmap = buildFunctionWithGradient(model) # build function, count the number of parameters
	state0 = Sample(setInit(init, nparams)) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	#  first calc
	calc!(state0, ll_func)
	assert(isfinite(state0.llik), "Initial values out of model support, try other values")

	#  main loop
 	for i in 1:steps  #i=1
 		local j, state

 		progress(i, steps, burnin)

 		state0.v = randn(nparams)
 		update!(state0)
 		state = state0

		j=1
		while j <= isteps && isfinite(state.llik)
			state = leapFrog(state, stepsize, ll_func)
			j +=1
		end

		# accept if new is good enough
		if rand() < exp(state.H - state0.H)
			state0 = state
		end

		i > burnin ? addToRes!(res, pmap, i-burnin, state0.llik, state0 == state, state0.beta) : nothing
	end

	calcStats!(res) # calculate some stats on this run
	res
end

simpleHMC(model::Expr, steps::Integer, isteps::Integer, stepsize::Float64) = 
simpleHMC(model, steps, min(steps-1, div(steps,2)), isteps, stepsize)
simpleHMC(model::Expr, steps::Integer, burnin::Integer, isteps::Integer, stepsize::Float64) = 
	simpleHMC(model, steps, burnin, 1.0, isteps, stepsize)


##########################################################################################
#   NUTS sampler function
#
#   Ref : The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo - Hoffman/Gelman
#
##########################################################################################

function simpleNUTS(model::Expr, steps::Integer, burnin::Integer, init::Any)
    local epsilon, u_slice
    local state0  # starting state of each loop
	
	tic() # start timer
	checkSteps(steps, burnin) # check burnin steps consistency
	
	ll_func, nparams, pmap = buildFunctionWithGradient(model) # build function, count the number of parameters
	state0 = Sample(setInit(init, nparams)) # build the initial values
	res = setRes(steps, burnin, pmap) #  result structure setup

	# first calc
	calc!(state0, ll_func)
	assert(isfinite(state0.llik), "Initial values out of model support, try other values")

	# find initial value for epsilon
	epsilon = 1.
	state0.v = randn(nparams)
	state1 = leapFrog(state0, epsilon, ll_func)

	ratio = exp(state1.H - state0.H)
	a = 2*(ratio>0.5)-1.
	while ratio^a > 2^-a
		epsilon *= 2^a
		state1 = leapFrog(state0, epsilon, ll_func)
		ratio = exp(state1.H - state0.H)
	end

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
	function buildTree(state, dir, j, ll)
		local state1, n1, s1, alpha1, nalpha1
		local state2, n2, s2, alpha2, nalpha2
		local state_plus, state_minus
		local dummy
		const deltamax = 100

		if j == 0
			state1 = leapFrog(state, dir*epsilon, ll)
			n1 = ( u_slice <= state1.H ) + 0 
			s1 = u_slice < ( deltamax + state1.H )

			return state1, state1, state1, n1, s1, min(1., exp(state1.H - state0.H)), 1
		else
			state_minus, state_plus, state1, n1, s1, alpha1, nalpha1 = buildTree(state, dir, j-1, ll)
			if s1 
				if dir == -1
					state_minus, dummy, state2, n2, s2, alpha2, nalpha2 = buildTree(state_minus, dir, j-1, ll)
	 			else
	 				dummy, state_plus, state2, n2, s2, alpha2, nalpha2 = buildTree(state_plus, dir, j-1, ll)
	 			end
	 			if rand() <= n2/(n2+n1)
	 				state1 = state2
	 			end

	 			alpha1 += alpha2
	 			nalpha1 += nalpha2
	 			s1 = s2 && !uturn(state_minus, state_plus)
	 			n1 += n2
	 		end

	 		return state_minus, state_plus, state1, n1, s1, alpha1, nalpha1
		end
	end

	res.misc[:jmax] = fill(NaN, res.steps)
	res.misc[:epsilon] = fill(NaN, res.steps)

	### main loop
 	for i in 1:steps  # i=1
 		local alpha, nalpha, n, s, j, n1, s1
 		local dummy, state_minus, state_plus, state, state1

 		progress(i, steps, burnin)

 		state0.v = randn(nparams)
 		update!(state0)

 		u_slice  = log(rand()) + state0.H # use log ( != paper) to avoid underflow
 		
 		state = state_minus = state_plus = state0

 		# inner loop
 		j, n = 0, 1
 		s = true
 		while s && j < 12
 			dir = (rand() > 0.5) * 2. - 1.
 			if dir == -1
 				state_minus, dummy, state1, n1, s1, alpha, nalpha = buildTree(state_minus, dir, j, ll_func)
 			else
 				dummy, state_plus, state1, n1, s1, alpha, nalpha = buildTree(state_plus, dir, j, ll_func)
 			end
 			if s1 && rand() < n1/n  # accept 
 				state = state1
 			end
 			n += n1
 			j += 1
 			s = s1 && !uturn(state_minus, state_plus)
 		end
 		
 		# epsilon adjustment
 		if i <= nadapt  # warming up period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^(-kappa)) * lebar
			epsilon = exp(le)
		else # post warm up, keep dual epsilon
			epsilon = exp(lebar)
		end

		# println(llik, beta)
		i > burnin ? addToRes!(res, pmap, i-burnin, state.llik, state != state0, state.beta) : nothing

		res.misc[:epsilon][i] = epsilon
		res.misc[:jmax][i] = j

		state0 = state
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

##### update progress bar
function progress(i::Integer, steps::Integer, burnin::Integer)
	if rem(50*i, steps) == 0    # 50 characters for full run
		print(i > burnin ? "+" : "-")
		i == steps ? println() : nothing
	end
end	