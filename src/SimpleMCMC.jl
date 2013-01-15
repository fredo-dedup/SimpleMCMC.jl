module SimpleMCMC

using Base

# windows 
load("../../Distributions.jl/src/distributions.jl")
push!(args...) = push(args...) # windows julia not up to date
delete!(args...) = del(args...) # windows julia not up to date

# linux
# using Distributions

export simpleRWM, simpleHMC
export buildFunction, buildFunctionWithGradient

const ACC_SYM = :__acc
const PARAM_SYM = :__beta
const TEMP_NAME = "tmp"
const DERIV_PREFIX = "d"

##############################################################################
#   Model expression processing functions			  
##############################################################################

##########  creates new types to ease AST exploration  ############
const emap = [:(=) => :Exprequal,
			  :(::) => :Exprdcolon,
			  :(+=) => :Exprpequal]

for ex in [:equal, :dcolon, :pequal, :call, :block, :ref, :line]
	nt = symbol(strcat("Expr", ex))
	# println("defining ", nt)
	eval(quote
		type $nt
			head::Symbol
			args::Vector
			typ::Any
		end
		($nt)(ex::Expr) = ($nt)(ex.head, ex.args, ex.typ)
		toExpr(ex::$nt) = expr(ex.head, ex.args)
	end)
end

function etype(ex::Expr)
	nt = has(emap, ex.head) ? emap[ex.head] : symbol(strcat("Expr", ex.head))
	# nt = symbol(strcat("SimpleMCMC.", nt))
	if nt == :Exprequal
		Exprequal(ex)
	elseif nt == :Exprpequal
		Exprpequal(ex)
	elseif nt == :Exprdcolon
		Exprdcolon(ex)
	elseif nt == :Exprblock
		Exprblock(ex)
	elseif nt == :Exprref
		Exprref(ex)
	elseif nt == :Exprline
		Exprline(ex)
	elseif nt == :Exprcall
		Exprcall(ex)
	else
		error("[etype] unmapped expr type $(ex.head)")
	end

	#TODO : turn all this in a loop on expression heads
	# f = eval(e,t) -> t(e)
	# f(ex, nt)
	# (type(nt))(ex)
	# eval(:(((e,t)->(t)(e))($(expr(:quote, ex)), nt)))
	# eval(:( $nt($(expr(:quote, ex))) ))
end

######### extracts model parameters from model expression  #############
function findParams(ex::Expr)

	function explore(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(etype(ex2))
				ex3==nothing ? nothing : push!(al, ex3)
			else
				push!(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	explore(ex::Exprline) = nothing  # remove #line statements
	explore(ex::Exprref) = toExpr(ex) # no processing
	explore(ex::Exprcall) = toExpr(ex) # no processing
	explore(ex::Exprequal) = toExpr(ex) # no processing

	function explore(ex::Exprdcolon)
		assert(typeof(ex.args[1]) == Symbol, 
			"not a symbol on LHS of :: $(ex.args[1])")
		par = ex.args[1]  # param symbol defined here
		def = ex.args[2]

		if def == :real  #  simple decl : var::scalar
			pmap[par] = :($PARAM_SYM[$(index+1)])
			index += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :real
				nb = Main.eval(e2[2])
				assert(isa(nb, Integer) && nb > 0, 
					"invalid vector size $(e2[2]) = $(nb)")

				pmap[par] = :($PARAM_SYM[$(index+1):$(nb+index)])
				index += nb
			else
				error("unknown parameter type $(e2[1])")
			end
		else
			error("unknown parameter expression $(ex)")
		end
		nothing
	end

	pmap = Dict{Symbol, Expr}()
	index = 0

	ex = explore(etype(ex))
	(ex, index, pmap)
end

######### translates ~ expressions into equivalent log-likelihood accumulator #############
function translateTilde(ex::Expr)

	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = toExpr(ex)
	explore(ex::Exprequal) = toExpr(ex)

	function explore(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(etype(ex2))
				ex3==nothing ? nothing : push!(al, ex3)
			else
				push!(al, ex2)
			end
		end
		expr(:block, al)
	end

	function explore(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		return :($ACC_SYM = $ACC_SYM + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
	end

	explore(etype(ex))
end

# slightly different behaviour if model gradient is to be calculated
# TODO : unify
function translateTilde2(ex::Expr)

	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = toExpr(ex)
	explore(ex::Exprequal) = toExpr(ex)

	function explore(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(etype(ex2))
				ex3==nothing ? nothing : push!(al, ex3)
			else
				push!(al, ex2)
			end
		end
		expr(:block, al)
	end

	function explore(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		fn = symbol("logpdf$(ex.args[3].args[1])")
		args = {expr(:., :SimpleMCMC, expr(:quote, fn))}
		# cat(args, ex.args[3].args[2:end])
		# push!(args, ex.args[2])
		for a in ex.args[3].args[2:end]
			push!(args, a)
		end
		push!(args, ex.args[2])
		return :($ACC_SYM = $ACC_SYM + sum($(expr(:call, args))))
	end

	explore(etype(ex))
end

######## unfolds expression before derivation ###################
function unfold(ex::Expr)

	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = toExpr(ex)

	function explore(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(etype(ex2))
				if ex3==nothing
					# nothing to add
				elseif isa(etype(ex3), Exprblock) # if block insert block args instead of block expr
					al = vcat(al, ex3.args)
				else
					push!(al, ex3)
				end
			else
				push!(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	function explore(ex::Exprequal)
		lhs = ex.args[1]
		assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
			"[unfold] not a symbol on LHS of assigment $(ex)")

		rhs = ex.args[2]
		if isa(rhs, Symbol)
			return expr(:(=), lhs, rhs)
		elseif isa(rhs, Expr)
			ue = explore(etype(rhs))
			if isa(ue, Expr)
				return expr(:(=), lhs, rhs)
			elseif isa(ue, Tuple)
				lb = push!(ue[1], :($lhs = $(ue[2])))
				return expr(:block, lb)
			end
		else  # unmanaged kind of lhs
		 	error("[unfold] can't handle RHS of assignment $ex")
		end
	end

	function explore(ex::Exprcall)
		na = {ex.args[1]}   # function name
		args = ex.args[2:end]  # arguments

		# if more than 2 arguments, +, sum and * are converted  to nested expressions
		#  (easier for derivation)
		# TODO : apply to other n-ary (n>2) operators ?
		if contains([:+, :*, :sum], na) 
			while length(args) > 2
				a2 = pop(args)
				a1 = pop(args)
				push!(args, expr(:call, ex.args[1], a1, a2))
			end
		end

		lb = {}
		for e2 in args  # e2 = args[2]
			if isa(e2, Expr)
				ue = explore(etype(e2))
				if isa(ue, Tuple)
					append!(lb, ue[1])
					lp = ue[2]
				else
					lp = ue
				end
				nv = gensym(TEMP_NAME)
				push!(lb, :($nv = $(lp)))
				push!(na, nv)
			else
				push!(na, e2)
			end
		end

		return length(lb)==0 ? expr(ex.head, na) : (lb, expr(ex.head, na))
	end

	explore(etype(ex))
end

######### identifies derivation vars (descendants of model parameters)  #############
# TODO : further filtering to keep only those influencing the accumulator
# ERROR : add variable renaming when set several times (+ name tracking for accuulator)
function listVars(ex::Expr, avars) # entry function
	avars = Set{Symbol}(avars...)

	getSymbols(ex::Symbol) = Set{Symbol}(ex)
	getSymbols(ex::Expr) = getSymbols(etype(ex))
	getSymbols(ex::Exprref) = Set{Symbol}(ex.args[1])
	getSymbols(ex::Any) = Set{Symbol}()

	function getSymbols(ex::Exprcall)
		sl = Set{Symbol}()
		for ex2 in ex.args[2:end]
			sl = union(sl, getSymbols(ex2))
		end
		sl
	end

	function explore(ex::Exprequal)
		lhs = getSymbols(ex.args[1])
		rhs = getSymbols(ex.args[2])

		if length(intersect(rhs, avars)) > 0 
			 avars = union(avars, lhs)
		end
	end

	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = nothing
	explore(ex::Exprdcolon) = nothing
	explore(ex::Exprblock) = map(x->explore(etype(x)), ex.args)

	explore(etype(ex))
	avars
end

######### builds the gradient expression from unfolded expression ##############
function backwardSweep(ex::Expr, avars::Set{Symbol})
	assert(ex.head == :block, "[backwardSweep] not a block")

	explore(ex::Exprline) = nothing
	
	function explore(ex::Exprblock)
		el = {}

		for ex2 in ex.args
			ex3 = explore(etype(ex2))
			if ex3==nothing
				# nothing to add
			elseif isa(etype(ex3), Exprblock) # if block insert block args instead of block expr
				el = vcat(el, ex3.args)
			else
				push!(el, ex3)
			end
			ex3==nothing ? nothing : push!(el, ex3)
		end
		expr(:block, reverse(el))
	end

	function explore(ex::Exprequal)
		lhs = ex.args[1]
		if isa(lhs,Symbol) # simple var case
			dsym = lhs
		elseif isa(lhs,Expr) && lhs.head == :ref  # vars with []
			#dsym = expr(:ref, symbol("$(lhs.args[1])$(lhs.args[2])")
			dsym = lhs
		else
			error("[backwardSweep] not a symbol on LHS of assigment $(e)") 
		end
		
		rhs = ex.args[2]
		dsym2 = symbol("$(DERIV_PREFIX)$dsym")
		if isa(rhs,Symbol) 
			vsym2 = symbol("$(DERIV_PREFIX)$rhs")
			return :( $vsym2 = $dsym2)
			
		elseif isa(etype(rhs), Exprref)
			vsym2 = expr(:ref, symbol("$(DERIV_PREFIX)$(rhs.args[1])"), 
				rhs.args[2])
			#symbol("$(DERIV_PREFIX)$rhs")
			return :( $vsym2 = $dsym2)

		elseif isa(etype(rhs), Exprcall)  
			el = {}
			for i in 2:length(rhs.args) #i=3
				vsym = rhs.args[i]
				if isa(vsym, Symbol) && contains(avars, vsym)
				# derive(rhs, 1, :($(dsym)))
					push!(el, derive(rhs, i-1, dsym))
				end
			end
			if length(el) == 0
				return nothing
			elseif length(el) == 1
				return el[1]
			else
				return expr(:block, el)
			end
		else # TODO manage ref expressions
			error("[backwardSweep] can't derive $rhs")
		end
	end

	explore(etype(ex))
end

function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))
	op = opex.args[1]  # operator
	vs = opex.args[1+index]
	args = opex.args[2:end]
	dvs = symbol("$(DERIV_PREFIX)$vs")
	ds = symbol("$(DERIV_PREFIX)$dsym")

	# println(op, " ", vs, " ", args, " ", dvs, " ", ds)

	# TODO : all the dict expressions are evaluated, should be deferred
	# TODO : cleanup, factorize this mess
	if length(args) == 1 # unary operators
		drules_unary = {
			:- => :(-$ds),
			:log => :($ds ./ $vs),
			:sum => ds,
			:sin => :(cos($vs) .* $ds),
			:exp => :(exp($vs) .* $ds)
		}

		assert(has(drules_unary, op), "[derive] Doesn't know how to derive unary operator $op")
		return :($dvs += $(drules_unary[op]) )

	elseif length(args) == 2 # binary operators
		drules_binary = {
			:-  => {ds, :(-$ds)},
			:+  => {ds, ds},
			:sum  => {ds, ds},
			:*  => {:($ds * transpose($(args[2]))), 
					:(transpose($(args[1])) * $ds)},
			:dot => {:(sum($(args[2])) .* $ds), 
					 :(sum($(args[1])) .* $ds)},
			:^ => {:($(args[2]) * $vs ^ ($(args[2])-1) * $ds),
				   :(log($(args[1])) * $(args[1]) ^ $vs * $ds)},
			:/ => {:($vs ./ $(args[2]) .* $ds),
				   :(- $(args[1]) ./ ($vs .* $vs) .* $ds)}
		}

		assert(has(drules_binary, op), "[derive] Doesn't know how to derive binary operator $op")
		return :($dvs += $(drules_binary[op][index]) )

	elseif length(args) == 3 # ternary operators
		drules_ternary = {
			:sum  => {ds, ds, ds} #,
		}

		if isa(op, Symbol)
			assert(has(drules_ternary, op), "[derive] Doesn't know how to derive ternary operator $op")
			return :($dvs += $(drules_ternary[op][index]) )

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfNormal))
			rules = {	# mu
						:(sum($(args[3]) - $(args[1]) ) / $(args[2])),
					  	# sigma
					  	:(sum( ($(args[3]) - $(args[1])).^2 ./ $(args[2])^2 - 1.0) / $(args[2])),
					  	# x
					  	:(($(args[1]) - $(args[3]) ) ./ $(args[2]))
					}
			return :($dvs += $(rules[index]) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfUniform))
			rules = {	# a
						:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ))), 
						# b
					 	:(sum( log( -($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ) )),
					 	# x
					 	:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ ($(args[2]) - $(args[1])) ) ))
					}
			return :($dvs += $(rules[index]) .* $ds)

		elseif op == expr(:., :SimpleMCMC, expr(:quote, :logpdfWeibull))
			rules = {	# shape
						:(sum( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* log($(args[3])./$(args[2])) + 1./$(args[1]))), 
						# scale
					 	:(sum( (($(args[3])./$(args[2])).^$(args[1]) - 1.0) .* $(args[1]) ./ $(args[2]))),
					 	# x
					 	:(sum( ( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* $(args[1]) -1.0) ./ $(args[3])))
					}
			return :($dvs += $(rules[index]) .* $ds)

		else
			error("[derive] Doesn't know how to derive ternary operator $op")	
		end

	else
		error("[derive] Doesn't know how to derive n-ary operator $op")
	end
end

###############  hooks into Distributions library  ###################

#TODO : implement here functions that can be simplified (eg. logpdf(Normal)) as this is not always done in Distributions
#TODO : Distributions is not vectorized on distributions parameters (mu, sigma), another reason for rewriting here
logpdfNormal(mu, sigma, x) = Distributions.logpdf(Normal(mu, sigma), x)
logpdfWeibull(shape, scale, x) = logpdf(Weibull(shape, scale), x)
logpdfUniform(a, b, x) = logpdf(Distributions.Uniform(a, b), x)

######### builds the full functions ##############

function buildFunction(model::Expr)
	
	(model2, nparams, pmap) = findParams(model)
	model3 = translateTilde(model2)

	assigns = { expr(:(=), k, v) for (k,v) in pairs(pmap)}
	f = quote
		function __loglik($PARAM_SYM::Vector{Float64})
			local $ACC_SYM
			$(Expr(:block, assigns, Any))
			$ACC_SYM = 0.0
			$model3
			return($ACC_SYM)
		end
	end

	(f, nparams)
end

function buildFunctionWithGradient(model::Expr)
	
	(model2, nparams, pmap) = findParams(model)
	model3 = translateTilde(model2)
	model4 = translateTilde2(model2)
	model4 = unfold(model4)
	avars = listVars(model4, keys(pmap))
	dmodel = backwardSweep(model4, avars)

	assigns = { expr(:(=), k, v) for (k,v) in pairs(pmap)}

	if length(pmap) == 1
		dexp = symbol("$DERIV_PREFIX$(keys(pmap)[1])")
	else
		dexp = {:vcat}
		dexp = vcat(dexp, { symbol("$DERIV_PREFIX$v") for v in keys(pmap)})
		dexp = expr(:call, dexp)
	end

	delete!(avars, ACC_SYM) # remove accumulator, special treatment needed

	f = quote
		function __loglik($PARAM_SYM::Vector{Float64})
			local $ACC_SYM
			$(Expr(:block, assigns, Any))
			# first pass
			$ACC_SYM = 0.
			$model4

			# derivatives init
			$(symbol("$DERIV_PREFIX$ACC_SYM")) = 1.0
			$(expr(:block, {:($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))) for v in avars}))  

			$dmodel
			($ACC_SYM, $dexp)
		end
	end

	(f, nparams)
end


##########################################################################################
#   Random Walk Metropolis function
##########################################################################################

function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
	local beta, nparams
	local jump, S, eta, SS
	const local target_alpha = 0.234

	nparams = 10

	# check burnin steps consistency
	assert(steps >= burnin && burnin >= 0 && steps > 0, "Steps should be >= to burnin, and both positive")

	# build function, count the number of parameters
	ll_func, nparams = buildFunction(model)
	# create function (in Main !)
	Main.eval(ll_func)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		beta = init
	elseif typeof(init) <: Real
		beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end

	#  first calc
	__lp = Main.__loglik(beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

	S = eye(nparams) # initial value for jump scaling matrix
 	for i in 1:steps	 # i=1; burnin=10		
 		# print(i, " old beta = ", round(beta[1],3))
		jump = 0.1 * randn(nparams)
		oldbeta, beta = beta, beta + S * jump
		# print("new beta = ", round(beta[1], 3), " diag = ", round(diag(S), 3))

 		old__lp, __lp = __lp, Main.__loglik(beta)

 		alpha = min(1, exp(__lp - old__lp))
		if rand() > exp(__lp - old__lp)
			__lp, beta = old__lp, oldbeta
		end
 		# println("$i : lp= $(round(__lp, 3))")
		draws[i, :] = vcat(__lp, (old__lp != __lp), beta)

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

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, max(1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########################################################################################
#   Canonical HMC function
##########################################################################################

function simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, isteps::Integer, stepsize::Float64)
	# build function, count the number of parameters
	(ll_func, nparams) = buildFunctionWithGradient(model)
	# create function (in Main !)
	Main.eval(ll_func)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == nparams, "$nparams initial values expected, got $(length(init))")
		beta = init
	elseif typeof(init) <: Real
		beta = [ convert(Float64, init)::Float64 for i in 1:nparams]
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end

	#  first calc
	(__lp, grad) = Main.__loglik(beta)
	assert(__lp != -Inf, "Initial values out of model support, try other values")

	#  main loop
	draws = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

 	for i in 1:steps
 
 		jump0 = randn(nparams)
		beta0 = beta
		__lp0 = __lp

		jump = jump0 - stepsize * grad / 2.0
		for j in 1:(isteps-1)
			beta += stepsize * jump
			(__lp, grad) = Main.__loglik(beta)
			# println("     $j : lp= $(round(__lp, 3))")
			jump += stepsize * grad
		end
		beta += stepsize * jump
		(__lp, grad) = Main.__loglik(beta)
		# println("     $isteps : lp= $(round(__lp, 3))")
		jump -= stepsize * grad / 2.0

		jump = -jump
		# print("new beta = ", round(beta[1], 3), " diag = ", round(diag(S), 3))

		if rand() > exp((__lp + dot(jump,jump)/2.0) - (__lp0 + dot(jump0,jump0)/2.0))
			__lp, beta = __lp0, beta0
		end
 		# println("$i : lp= $(round(__lp, 3))")
		draws[i, :] = vcat(__lp, (__lp0 != __lp), beta)

	end

	draws[(burnin+1):steps, :]
end

simpleHMC(model::Expr, steps::Integer, isteps::Integer, stepsize::Float64) = 
simpleHMC(model, steps, max(1, div(steps,2)), isteps, stepsize)
simpleHMC(model::Expr, steps::Integer, burnin::Integer, isteps::Integer, stepsize::Float64) = 
	simpleHMC(model, steps, burnin, 1.0, isteps, stepsize)

##########  helper function to analyse expressions   #############

function expexp(ex::Expr, ident...)
	ident = (length(ident)==0 ? 0 : ident[1])::Integer
	println(rpad("", ident, " "), ex.head, " -> ")
	for e in ex.args
		typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
	end
end




end
