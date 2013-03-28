###############################################################################
#    Parsing functions
###############################################################################

##########  creates a parameterized type to ease AST exploration  ############
type ExprH{H}
	head::Symbol
	args::Vector
	typ::Any
end
toExprH(ex::Expr) = ExprH{ex.head}(ex.head, ex.args, ex.typ)
toExpr(ex::ExprH) = expr(ex.head, ex.args)

typealias Exprequal    ExprH{:(=)}
typealias Exprdcolon   ExprH{:(::)}
typealias Exprpequal   ExprH{:(+=)}
typealias Exprcall     ExprH{:call}
typealias Exprblock	   ExprH{:block}
typealias Exprline     ExprH{:line}
typealias Exprref      ExprH{:ref}
typealias Exprif       ExprH{:if}

## variable symbol polling functions
getSymbols(ex::Expr) =       getSymbols(toExprH(ex))
getSymbols(ex::Symbol) =     Set{Symbol}(ex)
getSymbols(ex::Exprref) =    Set{Symbol}(ex.args[1])
getSymbols(ex::Exprequal) =  union(getSymbols(ex.args[1]), getSymbols(ex.args[2]))
getSymbols(ex::Any) =        Set{Symbol}()
getSymbols(ex::Exprcall) =   mapreduce(getSymbols, union, ex.args[2:end])
getSymbols(ex::Exprif) =     mapreduce(getSymbols, union, ex.args)
getSymbols(ex::Exprblock) =  mapreduce(getSymbols, union, ex.args)

## symbol subsitution functions
subst(ex::Expr, smap::Dict) = expr(ex.head, map(ex -> subst(ex, smap), ex.args))
subst(ex::Symbol, smap::Dict) = has(smap, ex) ? smap[ex] : ex
subst(ex::Any, smap::Dict) = ex


######### parameters structure  ############
type MCMCParams
	sym::Symbol
	size::Vector{Integer}
	map::Union(Integer, Range1)  
end


######### parses model to extracts parameters and rewrite ~ operators #############
function parseModel(ex::Expr)

	explore(ex::Expr) = explore(toExprH(ex))
	explore(ex::ExprH) = error("[parseModel] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) = nothing  # remove #line statements
	explore(ex::Exprref) = toExpr(ex) # no processing
	explore(ex::Exprequal) = toExpr(ex) # no processing

	function explore(ex::Exprblock)
		al = {}
		for ex2 in ex.args
			if isa(ex2, Expr)
				ex3 = explore(ex2)
				ex3==nothing ? nothing : push!(al, ex3)
			else
				push!(al, ex2)
			end
		end
		expr(ex.head, al)
	end

	function explore(ex::Exprdcolon)
		assert(typeof(ex.args[1]) == Symbol, "not a symbol on LHS of :: $(ex.args[1])")
		par = ex.args[1]  # param symbol defined here
		def = ex.args[2]

		if def == :real  #  single param declaration
			push!(pmap, MCMCParams(par, Integer[], index+1)) #Integer[index+1, index+1]))
			index += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :real 
				if length(e2) == 2 #  vector param declaration
					nb = Main.eval(e2[2])
					assert(isa(nb, Integer) && nb > 0, "invalid vector size $(e2[2]) = $(nb)")

					push!(pmap, MCMCParams(par, Integer[nb], (index+1):(index+nb)))
					index += nb
				elseif length(e2) == 3 #  matrix param declaration
					nb1 = Main.eval(e2[2])
					assert(isa(nb1, Integer) && nb1 > 0, "invalid vector size $(e2[2]) = $nb1")
					nb2 = Main.eval(e2[3])
					assert(isa(nb2, Integer) && nb2 > 0, "invalid vector size $(e2[3]) = $nb2")

					push!(pmap, MCMCParams(par, Integer[nb1, nb2], (index+1):(index+nb1*nb2))) #Integer[index+1, index+nb1*nb2]))
					index += nb1*nb2
				else
					error("up to 2 dim for parameters in $ex")
				end
			else
				error("unknown parameter type $(e2[1])")
			end
		else
			error("unknown parameter expression $(ex)")
		end
		nothing
	end

	function explore(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		fn = symbol("logpdf$(ex.args[3].args[1])")
		args = {expr(:., :SimpleMCMC, expr(:quote, fn))}
		args = vcat(args, ex.args[3].args[2:end])
		push!(args, ex.args[2])
		return :($ACC_SYM = $ACC_SYM + $(expr(:call, args)))

	end

	pmap = MCMCParams[]
	index = 0

	ex = explore(ex)
	(ex, index, pmap)
end



######## unfolds expressions to prepare derivation ###################
function unfold(ex::Expr)
	# Assumes there is only refs or calls on rhs of equal expressions, TODO : generalize ? (add blocks ?)

	explore(ex::Expr) = explore(toExprH(ex))
	explore(ex::ExprH) = error("[unfold] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = toExpr(ex)

	function explore(ex::Exprblock)
		for ex2 in ex.args # ex2 = ex.args[1]
			if isa(ex2, Expr)
				explore(ex2)
			else  # is that possible ??
				push!(el, ex2)
			end
		end
	end

	function explore(ex::Exprequal) 
		lhs = ex.args[1]
		assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
			"[unfold] not a symbol on LHS of assigment $(ex)")

		rhs = ex.args[2]
		if isa(rhs, Symbol)
			push!(el, expr(:(=), lhs, rhs))
		elseif isa(rhs, Expr) # only refs and calls will work
				ue = explore(toExprH(rhs)) # explore will return something in this case
				push!(el, expr(:(=), lhs, ue))
		elseif isa(rhs, Real)
			push!(el, expr(:(=), lhs, rhs))
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
		if contains([:+, :*, :sum], na[1]) 
			while length(args) > 2
				a2 = pop!(args)
				a1 = pop!(args)
				push!(args, expr(:call, ex.args[1], a1, a2))
			end
		end

		for e2 in args  
			if isa(e2, Expr) # only refs and calls will work
				ue = explore(e2)
				nv = gensym(TEMP_NAME)
				push!(el, :($nv = $(ue)))
				push!(na, nv)
			else
				push!(na, e2)
			end
		end

		expr(ex.head, na)
	end

	el = {}
	explore(ex)

	# before returning, rename variables set several times as this would make
	#  the automated derivation fail
	# ERROR : algo doesn't work when a variable sets individual elements, x = .. then x[3] = ...; 

    subst = Dict{Symbol, Symbol}()
    used = [ACC_SYM]
    for idx in 1:length(el) 
    	ex2 = el[idx]
        lhs = elements(SimpleMCMC.getSymbols(ex2.args[1]))[1]  # there should be only one
        rhs = SimpleMCMC.getSymbols(ex2.args[2])

        if isa(toExprH(el[idx]), Exprequal)
        	if isa(el[idx].args[2], Symbol) # simple assign
        		if has(subst, el[idx].args[2])
        			el[idx].args[2] = subst[el[idx].args[2]]
        		end
        	elseif isa(el[idx].args[2], Real) # if number do nothing
        	elseif isa(toExprH(el[idx].args[2]), Exprref) # simple assign of a ref
        		if has(subst, el[idx].args[2].args[1])
        			el[idx].args[2].args[1] = subst[el[idx].args[2].args[1]]
        		end
        	elseif isa(toExprH(el[idx].args[2]), Exprcall) # function call
        		for i in 2:length(el[idx].args[2].args) # i=4
	        		if !isa(el[idx].args[2].args[i], Real) && has(subst, el[idx].args[2].args[i])
	        			el[idx].args[2].args[i] = subst[el[idx].args[2].args[i]]
	        		end
	        	end
	        else
	        	error("[unfold] can't subsitute var name in $ex2")
	        end
	    else
	    	error("[unfold] not an assignment ! : $ex2")
	    end

        if contains(used, lhs) # var already set once
            subst[lhs] = gensym("$lhs")
        	if isa(el[idx].args[1], Symbol) # simple assign
        		if has(subst, el[idx].args[1])
        			el[idx].args[1] = subst[el[idx].args[1]]
        		end
        	elseif isa(toExprH(el[idx].args[1]), Exprref) # simple assign of a ref
        		if has(subst, el[idx].args[1].args[1])
        			el[idx].args[1].args[1] = subst[el[idx].args[1].args[1]]
        		end
	        else
	        	error("[unfold] can't subsitute var name in $lhs")
	        end
        else # var set for the first time
            push!(used, lhs)
        end

    end

	(el, has(subst, ACC_SYM) ? subst[ACC_SYM] : ACC_SYM )
end

######### identifies derivation vars (descendants of model parameters)  #############
# TODO : further filtering to keep only those influencing the accumulator
function listVars(ex::Vector, avars) 
	# 'avars' : parameter names whose descendants are to be listed by this function
	avars = Set{Symbol}(avars...)
	for ex2 in ex # ex2 = ex[1]
		assert(isa(ex2, Expr), "[processVars] not an expression : $ex2")

		lhs = getSymbols(ex2.args[1])
		rhs = getSymbols(ex2.args[2])

		if length(intersect(rhs, avars)) > 0 
			 avars = union(avars, lhs)
		end
	end

	avars
end

######### builds the gradient expression from unfolded expression ##############
function backwardSweep(ex::Vector, avars::Set{Symbol})  

	explore(ex::Expr) = explore(toExprH(ex))
	explore(ex::ExprH) = error("[backwardSweep] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) = nothing

	function explore(ex::Exprequal)
		lhs = ex.args[1]
		if isa(lhs,Symbol) # simple var case
			dsym = lhs
			dsym2 = symbol("$(DERIV_PREFIX)$lhs")
		elseif isa(lhs,Expr) && lhs.head == :ref  # vars with []
			dsym = lhs
			dsym2 = expr(:ref, symbol("$(DERIV_PREFIX)$(lhs.args[1])"), lhs.args[2:end]...)
		else
			error("[backwardSweep] not a symbol on LHS of assigment $(e)") 
		end
		
		rhs = ex.args[2]
		if isa(rhs,Symbol) 
			if contains(avars, rhs)
				vsym2 = symbol("$(DERIV_PREFIX)$rhs")
				push!(el, :( $vsym2 = $dsym2))
			end

		elseif isa(toExprH(rhs), Exprref)
			if contains(avars, rhs.args[1])
				vsym2 = expr(:ref, symbol("$(DERIV_PREFIX)$(rhs.args[1])"), rhs.args[2:end]...)
				push!(el, :( $vsym2 = $dsym2))
			end

		elseif isa(toExprH(rhs), Exprcall)  
			for i in 2:length(rhs.args) 
				vsym = rhs.args[i]
				if isa(vsym, Symbol) && contains(avars, vsym)
					push!(el, derive(rhs, i-1, dsym))
				end
			end
		else 
			error("[backwardSweep] can't derive $rhs")
		end
	end

	el = {}
	for ex2 in reverse(ex)
		assert(isa(ex2, Expr), "[backwardSweep] not an expression : $ex2")
		explore(ex2)
	end

	el
end

######### builds the full functions ##############

function buildFunction(model::Expr)
	(model2, nparams, pmap) = parseModel(model)

	body = [betaAssign(pmap), 
				[:($ACC_SYM = 0.)], 
				model2.args, 
				[:(return($ACC_SYM))] ]

	func = Main.eval(tryAndFunc(body, false))

	(func, nparams, pmap)
end

function buildFunctionWithGradient(model::Expr)
	
	(model2, nparams, pmap) = parseModel(model)
	exparray, finalacc = unfold(model2)
	avars = listVars(exparray, [p.sym for p in pmap])
	dmodel = backwardSweep(exparray, avars)

	body = betaAssign(pmap)
	push!(body, :($ACC_SYM = 0.)) # acc init
	body = vcat(body, exparray)

	push!(body, :($(symbol("$DERIV_PREFIX$finalacc")) = 1.0))
	if contains(avars, finalacc) # remove accumulator, treated above
		delete!(avars, finalacc)
	end
	for v in avars 
		push!(body, :($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))))
	end

	body = vcat(body, dmodel)

	if length(pmap) == 1
		dn = symbol("$DERIV_PREFIX$(pmap[1].sym)")
		dexp = :(vec([$dn]))  # reshape to transform potential matrices into vectors
	else
		dexp = {:vcat}
		# dexp = vcat(dexp, { (dn = symbol("$DERIV_PREFIX$(p.sym)"); :(vec([$DERIV_PREFIX$(p.sym)])) for p in pmap})
		dexp = vcat(dexp, { :( vec([$(symbol("$DERIV_PREFIX$(p.sym)"))]) ) for p in pmap})
		dexp = expr(:call, dexp)
	end

	push!(body, :(($finalacc, $dexp)))
	func = Main.eval(tryAndFunc(body, true))

	# println(expr(:block, body))
	(func, nparams, pmap)
end



######   Common functionality  ###########

# returns an array of expr assigning parameters from the beta vector
function betaAssign(pmap::Vector{MCMCParams})
	assigns = Expr[]
	for p in pmap
		if length(p.size) <= 1  # scalar or vector
			push!(assigns, :($(p.sym) = $PARAM_SYM[ $(p.map) ]) )
		else # matrix case  (needs a reshape)
			push!(assigns, :($(p.sym) = reshape($PARAM_SYM[ $(p.map) ], $(p.size[1]), $(p.size[2]))) )
		end
	end			
	assigns
end

# encloses an array of expr in a try block to catch zero likelihoods (-Inf log likelihood)
#  and generates function expression
function tryAndFunc(body::Vector, grad::Bool)
	body = expr(:try, expr(:block, body),
					:e, 
					expr(:block, 
						grad ? :(if e == "give up eval"; return(-Inf, zero($PARAM_SYM)); else; throw(e); end) :
									:(if e == "give up eval"; return(-Inf); else; throw(e); end)))

	expr(:function, expr(:call, gensym(LLFUNC_NAME), :($PARAM_SYM::Vector{Float64})),	
					expr(:block, body) )
end
