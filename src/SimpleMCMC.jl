
module SimpleMCMC

# using Base

# export simpleRWM, simpleHMC
# export simpleRWM
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
			  :(::) => :Exprdcolon]

for ex in [:equal, :dcolon, :call, :block, :ref, :line]
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

	# println(nt)
	# f = eval(e,t) -> t(e)
	# f(ex, nt)
	# (type(nt))(ex)
	# eval(:(((e,t)->(t)(e))($(expr(:quote, ex)), nt)))
	#println("$(ex.head) ... $ex ...  $nt")
	# Exprequal(ex)
	# eval(:( $nt($(expr(:quote, ex))) ))
end

######### extracts model parameters from model expression  #############
function findParams(ex::Expr)

	function findParams_generic(ex)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = findParams(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	findParams(ex::Exprline) = toExpr(ex)
	findParams(ex::Exprref) = toExpr(ex)
	findParams(ex::Exprcall) = toExpr(ex)
	findParams(ex::Exprblock) = findParams_generic(ex)
	findParams(ex::Exprequal) = findParams_generic(ex)

	function findParams(ex::Exprdcolon)
		assert(typeof(ex.args[1]) == Symbol, 
			"not a symbol on LHS of :: $(ex.args[1])")
		par = ex.args[1]  # param symbol defined here
		def = ex.args[2]

		if def == :scalar  #  simple decl : var::scalar
			pmap[par] = :($PARAM_SYM[$(index+1)])
			index += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :vector
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

	ex = findParams(etype(ex))
	(ex, index, pmap)
end

######### translates ~ expressions into equivalent log-likelihood accumulator #############
function translateTilde(ex::Expr)

	translateTilde(ex::Exprline) = nothing
	translateTilde(ex::Exprref) = toExpr(ex)
	translateTilde(ex::Exprequal) = toExpr(ex)

	function translateTilde(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = translateTilde(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(:block, al)
	end

	function translateTilde(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		return :($ACC_SYM = $ACC_SYM + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
	end

	translateTilde(etype(ex))
end

# slightly different behaviour if model gradient is to be calculated
# TODO : unify
function translateTilde2(ex::Expr)

	translateTilde(ex::Exprline) = nothing
	translateTilde(ex::Exprref) = toExpr(ex)
	translateTilde(ex::Exprequal) = toExpr(ex)

	function translateTilde(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = translateTilde(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(:block, al)
	end

	function translateTilde(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		args = {symbol("SimpleMCMC.logpdf$(ex.args[3].args[1])")}
		# cat(args, ex.args[3].args[2:end])
		# push(args, ex.args[2])
		for a in ex.args[3].args[2:end]
			push(args, a)
		end
		push(args, ex.args[2])
		return :($ACC_SYM = $ACC_SYM + sum($(expr(:call, args))))
	end

	translateTilde(etype(ex))
end

######## unfolds expression before derivation ###################
function unfold(ex::Expr)

	unfold(ex::Exprline) = nothing
	unfold(ex::Exprref) = toExpr(ex)
	function unfold(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = unfold(etype(ex2))
				ex3==nothing ? nothing : push(al, ex3)
			else
				push(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	function unfold(ex::Exprequal)
		lhs = ex.args[1]
		assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
			"[unfold] not a symbol on LHS of assigment $(ex)")

		rhs = ex.args[2]
		if isa(rhs, Symbol)
			return expr(:(=), lhs, rhs)
		elseif isa(rhs, Expr)
			ue = unfold(etype(rhs))
			if isa(ue, Expr)
				return expr(:(=), lhs, rhs)
			elseif isa(ue, Tuple)
				lb = push(ue[1], :($lhs = $(ue[2])))
				return expr(:block, lb)
			end
		else  # unmanaged kind of lhs
		 	error("[unfold] can't handle RHS of assignment $ex")
		end
	end

	function unfold(ex::Exprcall)
		na = {ex.args[1]}   # function name
		args = ex.args[2:end]  # arguments

		# if more than 2 arguments, +, sum and * are converted  to nested expressions
		#  (easier for derivation)
		# TODO : apply to other n-ary (n>2) operators ?
		if contains([:+, :*, :sum], na) 
			while length(args) > 2
				a2 = pop(args)
				a1 = pop(args)
				push(args, expr(:call, ex.args[1], a1, a2))
			end
		end

		lb = {}
		for e2 in args  # e2 = args[2]
			if isa(e2, Expr)
				ue = unfold(etype(e2))
				if isa(ue, Tuple)
					append!(lb, ue[1])
					lp = ue[2]
				else
					lp = ue
				end
				nv = gensym(TEMP_NAME)
				push(lb, :($nv = $(lp)))
				push(na, nv)
			else
				push(na, e2)
			end
		end

		return numel(lb)==0 ? expr(ex.head, na) : (lb, expr(ex.head, na))
	end

	unfold(etype(ex))
end

######### identifies derivation vars (descendants of model parameters)  #############
# TODO : further filtering to keep only those influencing the accumulator
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

	function listVars(ex::Exprequal)
		lhs = getSymbols(ex.args[1])
		rhs = getSymbols(ex.args[2])

		if length(intersect(rhs, avars)) > 0 
			 avars = union(avars, lhs)
		end
	end

	listVars(ex::Exprline) = nothing
	listVars(ex::Exprref) = nothing
	listVars(ex::Exprdcolon) = nothing
	listVars(ex::Exprblock) = map(x->listVars(etype(x)), ex.args)

	listVars(etype(ex))
	avars
end

######### builds the gradient expression from unfolded expression ##############
function backwardSweep(ex::Expr, avars::Set{Symbol})
	assert(ex.head == :block, "[backwardSweep] not a block")

	backwardSweep(ex::Exprline) = nothing
	function backwardSweep(ex::Exprblock)
		el = {}

		for ex2 in ex.args
			ex3 = backwardSweep(etype(ex2))
			ex3==nothing ? nothing : push(el, ex3)
		end
		expr(:block, reverse(el))
	end

	function backwardSweep(ex::Exprequal)
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
					push(el, derive(rhs, i-1, dsym))
				end
			end
			if numel(el) == 0
				return nothing
			elseif numel(el) == 1
				return el[1]
			else
				return expr(:block, el)
			end
		else # TODO manage ref expressions
			error("[backwardSweep] can't derive $rhs")
		end
	end

	backwardSweep(etype(ex))
end

function derive(opex::Expr, index::Integer, dsym::Union(Expr,Symbol))
	op = opex.args[1]  # operator
	vs = opex.args[1+index]
	args = opex.args[2:end]
	dvs = symbol("$(DERIV_PREFIX)$vs")
	ds = symbol("$(DERIV_PREFIX)$dsym")

	# println(op, " ", vs, " ", args, " ", dvs, " ", ds)

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
			:sum  => {ds, ds, ds},

			symbol("SimpleMCMC.logpdfNormal") => {	# mu
								:(sum($(args[3]) - $(args[1]) ) / $(args[2])),
							  	# sigma
					 		  	:(sum( ($(args[3]) - $(args[1])).^2 ./ $(args[2])^2 - 1.0) / $(args[2])),
							  	# x
					 		  	:(sum(- $(args[3]) + $(args[1]) ) / $(args[2]))
					 		  	},

			symbol("logpdfUniform") => {	# a
								:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ))), 
								# b
					 		  	:(sum( log( -($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ (($(args[2]) - $(args[1])).^2.0) ) )),
					 		  	# x
					 		  	:(sum( log( ($(args[1]) <= $(args[3]) <= $(args[2]) ? 1.0 : 0.0) ./ ($(args[2]) - $(args[1])) ) ))
					 		  	},

			symbol("logpdfWeibull") => {	# shape
								:(sum( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* log($(args[3])./$(args[2])) + 1./$(args[1]))), 
								# scale
					 		   	:(sum( (($(args[3])./$(args[2])).^$(args[1]) - 1.0) .* $(args[1]) ./ $(args[2]))),
					 		   	# x
					 		   	:(sum( ( (1.0 - ($(args[3])./$(args[2])).^$(args[1])) .* $(args[1]) -1.0) ./ $(args[3])))
					 		   	}
		}

		assert(has(drules_ternary, op), "[derive] Doesn't know how to derive ternary operator $op")
		return :($dvs += $(drules_ternary[op][index]) )

	else
		error("[derive] Doesn't know how to derive n-ary operator $op")
	end
end

logpdfNormal(mu, sigma, x) = logpdf(Normal(mu, sigma), x)
logpdfWeibull(shape, scale, x) = logpdf(Weibull(shape, scale), x)
logpdfUniform(a, b, x) = logpdf(Uniform(a, b), x)

######### builds the full functions ##############

function buildFunction(model::Expr)
	
	(model2, nparams, pmap) = SimpleMCMC.findParams(model)
	model3 = SimpleMCMC.translateTilde(model2)

	assigns = [ expr(:(=), k, v) for (k,v) in pairs(pmap)]
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

	assigns = [ expr(:(=), k, v) for (k,v) in pairs(pmap)]
	f = quote
		function __loglik($PARAM_SYM::Vector{Float64})
			local $ACC_SYM
			$(Expr(:block, assigns, Any))

			$ACC_SYM = 0.0
			$model4

			$(Expr(:block, {:($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))) for v in avars}, Any))  

			$dmodel
			($ACC_SYM, $(symbol("$DERIV_PREFIX$PARAM_SYM")))
		end
	end

	(f, nparams)
end


##########################################################################################
#   Random Walk Metropolis implementation
##########################################################################################

function simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
	local beta, nparams
	local jump, S, eta, SS
	const local target_alpha = 0.234

	# build function, count the number of parameters
	(ll_func, nparams) = buildFunction(model)
	# create function in Main (!)
	Main.eval(ll_func)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(numel(init) == nparams, "$nparams initial values expected, got $(numel(init))")
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
 		println("$i : lp= $(round(__lp, 3))")
		draws[i, :] = vcat(__lp, (old__lp != __lp), beta)

		#  Adaptive scaling using R.A.M. method
		eta = min(1, nparams*i^(-2/3))
		# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
		SS = (jump * jump') / dot(jump, jump) * eta * (alpha - target_alpha)
		SS = S * (eye(nparams) + SS) * S'
		S = chol(SS)
		S = S'
	end

	# temp = copy(samples)
	# println(size(temp))
	# (temp, 1.0, "abcd", size(temp))

	# x = copy(temp[ 1,1])
	# println("  x = $x")

	return draws[(burnin+1):steps, :]
end

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, max(1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########################################################################################
#   Canonical HMC implementation
##########################################################################################

function simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, length::Integer, stepsize::Float64)
	local beta, nparams
	local jump, S, eta, SS
	const local target_alpha = 0.234

	# build function, count the number of parameters
	(ll_func, nparams) = buildFunctionWithGradient(model)
	# create function in Main (!)
	Main.eval(ll_func)

	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(numel(init) == nparams, "$nparams initial values expected, got $(numel(init))")
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
	samples = zeros(Float64, (steps, 2+nparams)) # 2 additionnal columns for storing log lik and accept/reject flag

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
 		println("$i : lp= $(round(__lp, 3))")
		samples[i, :] = vcat(__lp, (old__lp != __lp), beta)

		#  Adaptive scaling using R.A.M. method
		eta = min(1, nparams*i^(-2/3))
		# eta = min(1, nparams * (i <= burnin ? 1 : i-burnin)^(-2/3))
		SS = (jump * jump') / dot(jump, jump) * eta * (alpha - target_alpha)
		SS = S * (eye(nparams) + SS) * S'
		S = chol(SS)
		S = S'
	end

	temp = copy(samples)
	# println(size(temp))
	# (temp, 1.0, "abcd", size(temp))

	x = copy(temp[ 1,1])
	println("  x = $x")

	return eval(:(x))
end

simpleRWM(model::Expr, steps::Integer) = simpleRWM(model, steps, max(1, div(steps,2)))
simpleRWM(model::Expr, steps::Integer, burnin::Integer) = simpleRWM(model, steps, burnin, 1.0)

##########  helper function  to analyse expressions   #############

function expexp(ex::Expr, ident...)
	ident = (length(ident)==0 ? 0 : ident[1])::Integer
	println(rpad("", ident, " "), ex.head, " -> ")
	for e in ex.args
		typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
	end
end




end
