###############################################################################
#    Parsing functions
###############################################################################

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
	#TODO : turn all this into a Dict lookup
	nt = has(emap, ex.head) ? emap[ex.head] : symbol(strcat("Expr", ex.head))
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
end

##########  helper function to get symbols appearing in AST ############
getSymbols(ex::Expr) = getSymbols(etype(ex))
getSymbols(ex::Symbol) = Set{Symbol}(ex)
getSymbols(ex::Exprref) = Set{Symbol}(ex.args[1])
getSymbols(ex::Any) = Set{Symbol}()

function getSymbols(ex::Exprcall)
	sl = Set{Symbol}()
	for ex2 in ex.args[2:end]
		sl = union(sl, getSymbols(ex2))
	end
	sl
end

######### parses model to extracts parameters and rewrite ~ operators #############
function parseModel(ex::Expr, gradient::Bool)
	# 'gradient' specifies if gradient is to be calculated later because it affects ~ translation

	explore(ex::Exprline) = nothing  # remove #line statements
	explore(ex::Exprref) = toExpr(ex) # no processing
	explore(ex::Exprequal) = toExpr(ex) # no processing

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

	function explore(ex::Exprdcolon)
		assert(typeof(ex.args[1]) == Symbol, 
			"not a symbol on LHS of :: $(ex.args[1])")
		par = ex.args[1]  # param symbol defined here
		def = ex.args[2]

		if def == :real  #  single param declaration
			pmap[par] = :($PARAM_SYM[$(index+1)])
			index += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :real #  vector param declaration
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

	function explore(ex::Exprcall)
		ex.args[1] == :~ ? nothing : return toExpr(ex)

		if gradient
			fn = symbol("logpdf$(ex.args[3].args[1])")
			args = {expr(:., :SimpleMCMC, expr(:quote, fn))}
			args = vcat(args, ex.args[3].args[2:end])
			push!(args, ex.args[2])
			return :($ACC_SYM = $ACC_SYM + $(expr(:call, args)))

		else	
			args = {expr(:., :Distributions, expr(:quote, symbol("logpdf"))),
					expr(:call, expr(:., :Distributions, expr(:quote, ex.args[3].args[1])), ex.args[3].args[2:end]...),
					ex.args[2]}
			return :($ACC_SYM = $ACC_SYM + $(expr(:call, args)))
		end

	end

	pmap = Dict{Symbol, Expr}()
	index = 0

	ex = explore(etype(ex))
	(ex, index, pmap)
end



######## unfolds expressions to prepare derivation ###################
function unfold(ex::Expr)
	# TODO : assumes there is only refs or calls within equal expressions, improve (add blocks ?)

	explore(ex::Exprline) = nothing
	explore(ex::Exprref) = toExpr(ex)

	function explore(ex::Exprblock)
		for ex2 in ex.args # ex2 = ex.args[1]
			if isa(ex2, Expr)
				explore(etype(ex2))
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
				ue = explore(etype(rhs)) # explore will return something in this case
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
				a2 = pop(args)
				a1 = pop(args)
				push!(args, expr(:call, ex.args[1], a1, a2))
			end
		end

		for e2 in args  
			if isa(e2, Expr) # only refs and calls will work
				ue = explore(etype(e2))
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
	explore(etype(ex))

	# before returning, rename variables set several times as this would make
	#  the automated derivation fail
    subst = Dict{Symbol, Symbol}()
    used = [ACC_SYM]
    for idx in 1:length(el) 
    	ex2 = el[idx]
        lhs = elements(SimpleMCMC.getSymbols(ex2.args[1]))[1]  # there should be only one
        rhs = SimpleMCMC.getSymbols(ex2.args[2])

        if isa(etype(el[idx]), Exprequal)
        	if isa(el[idx].args[2], Symbol) # simple assign
        		if has(subst, el[idx].args[2])
        			el[idx].args[2] = subst[el[idx].args[2]]
        		end
        	elseif isa(el[idx].args[2], Real) # if number do nothing
        	elseif isa(etype(el[idx].args[2]), Exprref) # simple assign of a ref
        		if has(subst, el[idx].args[2].args[1])
        			el[idx].args[2].args[1] = subst[el[idx].args[2].args[1]]
        		end
        	elseif isa(etype(el[idx].args[2]), Exprcall) # function call
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
        	elseif isa(etype(el[idx].args[1]), Exprref) # simple assign of a ref
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

	explore(ex::Exprline) = nothing

	function explore(ex::Exprequal)
		lhs = ex.args[1]
		if isa(lhs,Symbol) # simple var case
			dsym = lhs
		elseif isa(lhs,Expr) && lhs.head == :ref  # vars with []
			dsym = lhs
		else
			error("[backwardSweep] not a symbol on LHS of assigment $(e)") 
		end
		
		rhs = ex.args[2]
		dsym2 = symbol("$(DERIV_PREFIX)$dsym")
		if isa(rhs,Symbol) 
			if contains(avars, rhs)
				vsym2 = symbol("$(DERIV_PREFIX)$rhs")
				push!(el, :( $vsym2 = $dsym2))
			end

		elseif isa(etype(rhs), Exprref)
			if contains(avars, rhs.args[1])
				vsym2 = expr(:ref, symbol("$(DERIV_PREFIX)$(rhs.args[1])"), rhs.args[2])
				push!(el, :( $vsym2 = $dsym2))
			end

		elseif isa(etype(rhs), Exprcall)  
			for i in 2:length(rhs.args) #i=3
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
		explore(etype(ex2))
	end

	el
end

######### builds the full functions ##############

function buildFunction(model::Expr)
	(model2, nparams, pmap) = parseModel(model, false)

	assigns = {expr(:(=), k, v) for (k,v) in pairs(pmap)}

	body = expr(:block, vcat(assigns, {:($ACC_SYM = 0.)}, model2.args, {:(return($ACC_SYM))}))
	func = expr(:function, expr(:call, LLFUNC_SYM, :($PARAM_SYM::Vector{Float64})),	body)

	(func, nparams)
end

function buildFunctionWithGradient(model::Expr)
	
	(model2, nparams, pmap) = parseModel(model, true)
	exparray, finalacc = unfold(model2)
	avars = listVars(exparray, keys(pmap))
	dmodel = backwardSweep(exparray, avars)

	# build body of function
	body = { expr(:(=), k, v) for (k,v) in pairs(pmap)}

	push!(body, :($ACC_SYM = 0.)) 

	body = vcat(body, exparray)

	push!(body, :($(symbol("$DERIV_PREFIX$finalacc")) = 1.0))
	if contains(avars, finalacc)
		delete!(avars, finalacc)
	end
	for v in avars # remove accumulator, treated above
		push!(body, :($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))))
	end

	body = vcat(body, dmodel)

	if length(pmap) == 1
		dexp = symbol("$DERIV_PREFIX$(keys(pmap)[1])")
	else
		dexp = {:vcat}
		dexp = vcat(dexp, { symbol("$DERIV_PREFIX$v") for v in keys(pmap)})
		dexp = expr(:call, dexp)
	end

	push!(body, :(($finalacc, $dexp)))

	# build function
	func = expr(:function, expr(:call, LLFUNC_SYM, :($PARAM_SYM::Vector{Float64})),	
				expr(:block, body))

	(func, nparams)
end


##########  helper function to analyse expressions   #############

function expexp(ex::Expr, ident...)
	ident = (length(ident)==0 ? 0 : ident[1])::Integer
	println(rpad("", ident, " "), ex.head, " -> ")
	for e in ex.args
		typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
	end
end


