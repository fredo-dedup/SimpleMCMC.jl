module SimpleMCMC

@unix_only begin
	require("Distributions")

	#  include model processing functions		
	include("parsing.jl")

	#  include derivatives definitions
	include("diff.jl")
end

@windows_only begin  # older version on my side requires a few tweaks
	include("../../.julia/Distributions.jl/src/Distributions.jl")
	
	push!(args...) = push(args...) # windows julia version not up to date
	delete!(args...) = del(args...) # windows julia version not up to date

	#  include model processing functions		
	include("../src/parsing.jl")	

	#  include derivatives definitions
	include("../src/diff.jl")
end


import 	Distributions.logpdf
import 	Distributions.Normal, 
		Distributions.Uniform, 
		Distributions.Weibull 


export simpleRWM, simpleHMC
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
	const local target_alpha = 0.234

	nparams = 10

	# check burnin steps consistency
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > 0, "Steps ($steps) should be > 0")

	# build function, count the number of parameters
	ll_func, nparams = buildFunction(model)
	Main.eval(ll_func) # create function in Main !
	llcall = expr(:call, expr(:., :Main, expr(:quote, LLFUNC_SYM)), :__beta)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		__beta = init
	elseif typeof(init) <: Real
		__beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end

	#  first calc
	__lp = Main.__loglik(__beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

	S = eye(nparams) # initial value for jump scaling matrix
 	for i in 1:steps	
 		# print(i, " old beta = ", round(__beta[1],3))
		jump = 0.1 * randn(nparams)
		old__beta, __beta = __beta, __beta + S * jump
		# print("new beta = ", round(__beta[1], 3), " diag = ", round(diag(S), 3))

 		old__lp, __lp = __lp, Main.__loglik(__beta) # eval(llcall)

 		alpha = min(1, exp(__lp - old__lp))
		if rand() > exp(__lp - old__lp)
			__lp, __beta = old__lp, old__beta
		end
		draws[i, :] = vcat(__lp, (old__lp != __lp), __beta) # println("$i : lp= $(round(__lp, 3))")

		#  Adaptive scaling using R.A.M. method
		eta = min(1, nparams*i^(-2/3))
		# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
		SS = (jump * jump') / dot(jump, jump) * eta * (alpha - target_alpha)
		SS = S * (eye(nparams) + SS) * S'
		S = chol(SS)
		S = S'
	end

	draws[(burnin+1):steps, :]
end

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, min(steps-1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########################################################################################
#   Canonical HMC function
##########################################################################################

function simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, isteps::Integer, stepsize::Float64)
	# TODO : manage cases when loglik = -Inf in inner loop

	# check burnin steps consistency
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > 0, "Steps ($steps) should be > 0")

	# build function, count the number of parameters
	(ll_func, nparams) = buildFunctionWithGradient(model)
	Main.eval(ll_func) # create function (in Main !)
	llcall = expr(:call, expr(:., :Main, expr(:quote, LLFUNC_SYM)), :__beta)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		__beta = init
	elseif typeof(init) <: Real
		__beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end

	#  first calc
	(__lp, grad) = Main.__loglik(__beta) # eval(llcall)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

 	for i in 1:steps
 
 		jump0 = randn(nparams)
		__beta0 = __beta
		__lp0 = __lp

		jump = jump0 - stepsize * grad / 2.0
		for j in 1:(isteps-1)
			__beta += stepsize * jump
			(__lp, grad) = Main.__loglik(__beta) # eval(llcall) # println("     $j : lp= $(round(__lp, 3))")
			jump += stepsize * grad
		end
		__beta += stepsize * jump
		(__lp, grad) = Main.__loglik(__beta) # eval(llcall) # println("     $isteps : lp= $(round(__lp, 3))")
		jump -= stepsize * grad / 2.0

		jump = -jump # print("new beta = ", round(__beta[1], 3), " diag = ", round(diag(S), 3))

		if rand() > exp((__lp + dot(jump,jump)/2.0) - (__lp0 + dot(jump0,jump0)/2.0))
			__lp, __beta = __lp0, __beta0
		end
		draws[i, :] = vcat(__lp, (__lp0 != __lp), __beta) # println("$i : lp= $(round(__lp, 3))")

	end

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

	# check burnin steps consistency
	assert(steps > burnin, "Steps ($steps) should be > to burnin ($burnin)")
	assert(burnin >= 0, "Burnin rounds ($burnin) should be >= 0")
	assert(steps > 0, "Steps ($steps) should be > 0")

	# build function, count the number of parameters
	(ll_func, nparams) = buildFunctionWithGradient(model)
	Main.eval(ll_func) # create function (in Main !)
	llcall = expr(:call, expr(:., :Main, expr(:quote, LLFUNC_SYM)), :__beta)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		__beta = init
	elseif typeof(init) <: Real
		__beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end

	#  first calc
	(__lp, grad) = Main.__loglik(__beta) # eval(llcall)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

	epsilon = findEpsilon()
	mu = log(10*epsilon)
	e0bar, hbar = 1., 0.
	lebar = 0.0
	gam = 0.05
	t0 = 10
	kappa = 0.75
 	for i in 1:steps
 
 		r0 = randn(nparams)
 		u  = rand() * exp(__lpbefore - dot(r0,r0)/2.0)
 		beta = betap = betam = betabefore
 		rp = rm = r0
 		j, n, s = 0, 1, 1

 		# inner loop
 		while s == 1
 			v = (randn() > 0.5) * 2. - 1.
 			if v == -1
 				betam, rm, nothing, nothing, beta1, n1, s1, alpha, nalpha = 
 					buildTree(betam, rm, u, v, j, epsilon, betabefore, r0)
 			else
 				nothing, nothing, betap, rp, beta1, n1, s1, alpha, nalpha = 
 					buildTree(betap, rp, u, v, j, epsilon, betabefore, r0)
 			end
 			if s1 == 1 && rand() < min(1.0, n1/n)
 				beta = beta1
 			end
 			n += n1
 			j += 1
 			s = s1 * (dot((betap-betam), rm) >= 0.0) * (dot((betap-betam), rp) >= 0.0)
 		end
 		# e adjustment
 		if i <= nadapt  # adjustment period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0))
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^-kappa) * lebar
			epsilon = exp(le)
		else # post adjustment period
			epsilon = exp(lebar)
		end

		betabefore = beta
		draws[i, :] = vcat(__lp, (__lp0 != __lp), beta)
	end

	# buidtree function
	function buildTree(beta, r, u, v, j, epsilon, betabefore, r0)
		if j == 0
			beta1, r1 = leapFrog(beta, r, v*epsilon)
			n1 = (u <= exp(__lp - dot(r1,r1)/2.0))
			s1 = (u < exp(deltamax + __lp - dot(r1,r1)/2.0))  # d'ou ça vient deltamax ??

			return beta1, r1, beta1, r1, beta1, n1, s1, 
				min(1., exp(__lp - dot(r1,r1)/2. - lp(betabefore) + dot(r0,r0)/2.)), 1
		else
			betam, rm, betap, rp, beta1, n1, s1, alpha1, nalpha1 = 
				buildTree(beta, r, u, v, j-1, epsilon, betabefore, r0)
			if s1 == 1 
				if v == -1
					betam, rm, nothing, nothing, beta2, n2, s2, alpha2, nalpha2 = 
	 					buildTree(betam, rm, u, v, j-1, epsilon, betabefore, r0)
	 			else
	 				nothing, nothing, betap, rp, beta2, n2, s2, alpha2, nalpha2 = 
	 					buildTree(betap, rp, u, v, j-1, epsilon, betabefore, r0)
	 			end
	 			if rand() <= n2/(n2+n1)
	 				beta1 = beta2
	 			end
	 			alpha1 += alpha2
	 			nalpha1 += nalpha2
	 			s1 = s2 * (dot((betap-betam), rm) >= 0.0) * (dot((betap-betam), rp) >= 0.0)
	 			n += n1
	 		end

	 		return betam, rm, betap, rp, beta1, n1, s1, alpha1, nalpha1
		end
	end

	# leapfrog function
	function leapFrog()

	end


	draws[(burnin+1):steps, :]
end

simpleNUTS(model::Expr, steps::Integer) = simpleNUTS(model, steps, min(steps-1, div(steps,2)))
simpleNUTS(model::Expr, steps::Integer, burnin::Integer) = simpleNUTS(model, steps, burnin, 1.0)



# module end
end


