###############################################################################
#    Model expression parsing
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
typealias Exprmequal   ExprH{:(-=)}
typealias Exprtequal   ExprH{:(*=)}
typealias Exprtrans    ExprH{symbol("'")} 
typealias Exprcall     ExprH{:call}
typealias Exprblock	   ExprH{:block}
typealias Exprline     ExprH{:line}
typealias Exprvcat     ExprH{:vcat}
typealias Exprref      ExprH{:ref}
typealias Exprif       ExprH{:if}
typealias Exprcomp     ExprH{:comparison}

## variable symbol polling functions
getSymbols(ex::Any) =        Set{Symbol}()
getSymbols(ex::Symbol) =     Set{Symbol}(ex)
getSymbols(ex::Array) =      mapreduce(getSymbols, union, ex)
getSymbols(ex::Expr) =       getSymbols(toExprH(ex))
getSymbols(ex::ExprH) =      mapreduce(getSymbols, union, ex.args)
getSymbols(ex::Exprcall) =   mapreduce(getSymbols, union, ex.args[2:end])  # skip function name
getSymbols(ex::Exprref) =    mapreduce(getSymbols, union, ex.args) - Set(:(:), symbol("end")) # ':'' and 'end' do not count
getSymbols(ex::Exprcomp) =   mapreduce(getSymbols, union, ex.args) - Set(:(>), :(<), :(>=), :(<=), :(.>), :(.<), :(.<=), :(.>=), :(==))


## variable symbol subsitution functions
substSymbols(ex::Expr, smap::Dict) =          substSymbols(toExprH(ex), smap::Dict)
substSymbols(ex::Exprcall, smap::Dict) =      expr(:call, {ex.args[1], map(e -> substSymbols(e, smap), ex.args[2:end])...})
substSymbols(ex::ExprH, smap::Dict) =         expr(ex.head, map(e -> substSymbols(e, smap), ex.args))
substSymbols(ex::Symbol, smap::Dict) =        has(smap, ex) ? smap[ex] : ex
substSymbols(ex::Vector{Expr}, smap::Dict) =  map(e -> substSymbols(e, smap), ex)
substSymbols(ex::Any, smap::Dict) =           ex

######### parameters structure  ############
type MCMCParams
	sym::Symbol
	size::Vector{Integer}
	map::Union(Integer, Range1)  
end

######### model structure   ##############
type MCMCModel
	bsize::Int               # length of beta, the parameter vector
	pars::Vector{MCMCParams} # parameters with their mapping to the beta real vector
	init::Vector{Float64}    # initial values of beta
	source::Expr             # model source, after first pass
	exprs::Vector{Expr}      # vector of assigments that make the model
	dexprs::Vector{Expr}     # vector of assigments that make the gradient
	finalacc::Symbol         # last symbol of loglik accumulator after renaming
	varsset::Set{Symbol}     # all the vars set in the model
	pardesc::Set{Symbol}     # all the vars set in the model that depend on model parameters
	accanc::Set{Symbol}      # all the vars (possibly external) that influence the accumulator
end
MCMCModel() = MCMCModel(0, MCMCParams[], Float64[], :(), Expr[], Expr[], ACC_SYM, 
	Set{Symbol}(), Set{Symbol}(), Set{Symbol}())


######### first pass on the model
#  - extracts parameters definition
#  - rewrite ~ operators  as acc += logpdf..(=)
#  - translates x += y into x = x + y, same for -= and *=
function parseModel(ex::Expr)

	explore(ex::Expr) =       explore(toExprH(ex))
	explore(ex::ExprH) =      error("[parseModel] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) =   nothing  # remove #line statements
	explore(ex::Exprref) =    toExpr(ex) # no processing
	explore(ex::Exprequal) =  toExpr(ex) # no processing
	explore(ex::Exprvcat) =   toExpr(ex) # no processing
	
	explore(ex::Exprpequal) = (args = ex.args ; expr(:(=), args[1], expr(:call, :+, args...)) )
	explore(ex::Exprmequal) = (args = ex.args ; expr(:(=), args[1], expr(:call, :-, args...)) )
	explore(ex::Exprtequal) = (args = ex.args ; expr(:(=), args[1], expr(:call, :*, args...)) )

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
			push!(m.pars, MCMCParams(par, Integer[], m.bsize+1)) 
			m.bsize += 1

		elseif isa(def, Expr) && def.head == :call
			e2 = def.args
			if e2[1] == :real 
				if length(e2) == 2 #  vector param declaration
					nb = Main.eval(e2[2])
					assert(isa(nb, Integer) && nb > 0, "invalid vector size $(e2[2]) = $(nb)")

					push!(m.pars, MCMCParams(par, Integer[nb], (m.bsize+1):(m.bsize+nb)))
					m.bsize += nb
				elseif length(e2) == 3 #  matrix param declaration
					nb1 = Main.eval(e2[2])
					assert(isa(nb1, Integer) && nb1 > 0, "invalid vector size $(e2[2]) = $nb1")
					nb2 = Main.eval(e2[3])
					assert(isa(nb2, Integer) && nb2 > 0, "invalid vector size $(e2[3]) = $nb2")

					push!(m.pars, MCMCParams(par, Integer[nb1, nb2], (m.bsize+1):(m.bsize+nb1*nb2))) 
					m.bsize += nb1*nb2
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
		return :($ACC_SYM = $ACC_SYM + $(expr(:call, {fn, ex.args[3].args[2:end]..., ex.args[2]})))
	end

	m = MCMCModel()
	m.source = explore(ex)
	m
end

######## unfolds expressions to prepare derivation ###################
function unfold!(m::MCMCModel)

	explore(ex::Expr) =       explore(toExprH(ex))
	explore(ex::ExprH) =      error("[unfold] unmanaged expr type $(ex.head)")
	explore(ex::Exprline) =   nothing     # remove line info
	explore(ex::Exprref) =    toExpr(ex)   # unchanged
	explore(ex::Exprcomp) =   toExpr(ex)  # unchanged
	explore(ex::Exprvcat) =   explore(expr(:call, :vcat, ex.args...) )  # translate to vcat(), and explore
	explore(ex::Exprtrans) =  explore(expr(:call, :transpose, ex.args[1]) )  # translate to transpose() and explore
	explore(ex::Any) =        ex

	explore(ex::Exprblock) =  mapreduce(explore, (a,b)->b, ex.args)  # process, and return last evaluated
	
	function explore(ex::Exprequal) 
		lhs = ex.args[1]
		assert(typeof(lhs) == Symbol ||  (typeof(lhs) == Expr && lhs.head == :ref),
			"[unfold] not a symbol on LHS of assigment $ex")

		rhs = ex.args[2]
		if isa(rhs, Symbol) || isa(rhs, Real)
			push!(m.exprs, expr(:(=), lhs, rhs))
		elseif isa(rhs, Expr) # only refs and calls will work
				ue = explore(toExprH(rhs)) # explore will return something in this case
				push!(m.exprs, expr(:(=), lhs, ue))
		else  # unmanaged kind of lhs
		 	error("[unfold] can't handle RHS of assignment $ex")
		end
		lhs
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
				push!(m.exprs, :($nv = $ue))
				push!(na, nv)
			else
				push!(na, e2)
			end
		end

		expr(ex.head, na)
	end

	explore(m.source)
end

######### renames variables set several times to make them unique  #############
# FIXME : algo doesn't work when a variable sets individual elements, x = .. then x[3] = ...; 
# FIXME 2 : external variables redefined within model are not renamed
function uniqueVars!(m::MCMCModel)
	el = m.exprs
    subst = Dict{Symbol, Symbol}()
    used = Set(ACC_SYM)

    for idx in 1:length(el) # idx=4
        # first, substitute in the rhs the variables names that have been renamed
        el[idx].args[2] = substSymbols(el[idx].args[2], subst)

        # second, rename lhs symbol if set before
        lhs = collect(getSymbols(el[idx].args[1]))[1]  # there should be only one
        if contains(used, lhs) # if var already set once => create a new one
            subst[lhs] = gensym("$lhs") # generate new name, add it to substitution list for following statements
            el[idx].args[1] = substSymbols(el[idx].args[1], subst)
        else # var set for the first time
            used |= Set(lhs) 
        end
    end

	m.finalacc = has(subst, ACC_SYM) ? subst[ACC_SYM] : ACC_SYM  # keep reference of potentially renamed accumulator
end

######### identifies vars #############
# - lists variables that depend on model parameters 
# - lists variables that influence the accumulator
# - lists variables defined
# In order to 
#   1) restrict gradient code to the strictly necessary variables 
#   2) move parameter independant variables definition out the function (but within closure) 
#   3) remove unnecessary variables (with warning)
function categorizeVars!(m::MCMCModel) 
    m.varsset = mapreduce(p->getSymbols(p.args[1]), union, m.exprs)

    m.pardesc = Set{Symbol}([p.sym for p in m.pars]...)  # start with parameter symbols
    for ex2 in m.exprs 
        lhs = getSymbols(ex2.args[1])
        rhs = getSymbols(ex2.args[2])

        length(rhs & m.pardesc) > 0 ? m.pardesc = m.pardesc | lhs : nothing
    end

    m.accanc = Set{Symbol}(m.finalacc)
    for ex2 in reverse(m.exprs) # proceed backwards
        lhs = getSymbols(ex2.args[1])
        rhs = getSymbols(ex2.args[2])
        isa(ex2.args[1], Expr) && ex2.args[1].head == :ref ? rhs |= getSymbols(ex2.args[1].args[2]) : nothing

        length(lhs & m.accanc) > 0 ? m.accanc = m.accanc | rhs : nothing
    end
end

######### builds the gradient expression from unfolded expression ##############
function backwardSweep!(m::MCMCModel)  

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
			error("[backwardSweep] not a symbol on LHS of assigment $(ex)") 
		end
		
		rhs = ex.args[2]
		if !isa(rhs,Symbol) && !isa(rhs,Expr) # some kind of number, nothing to do

		elseif isa(rhs,Symbol) 
			if contains(avars, rhs)
				vsym2 = symbol("$(DERIV_PREFIX)$rhs")
				push!(m.dexprs, :( $vsym2 = $dsym2))
			end

		elseif isa(toExprH(rhs), Exprref)
			if contains(avars, rhs.args[1])
				vsym2 = expr(:ref, symbol("$(DERIV_PREFIX)$(rhs.args[1])"), rhs.args[2:end]...)
				push!(m.dexprs, :( $vsym2 = $dsym2))
			end

		elseif isa(toExprH(rhs), Exprcall)  
			for i in 2:length(rhs.args) 
				vsym = rhs.args[i]
				if isa(vsym, Symbol) && contains(avars, vsym)
					m.dexprs = vcat(m.dexprs, derive(rhs, i-1, dsym))
				end
			end
		else 
			error("[backwardSweep] can't derive $rhs")
		end
	end

	avars = m.accanc & m.pardesc
	for ex2 in reverse(m.exprs)  # proceed backwards
		assert(isa(ex2, Expr), "[backwardSweep] not an expression : $ex2")
		explore(ex2)
	end
end

######## sets inital values from 'init' given as parameter  ##########
function setInit!(m::MCMCModel, init)
	# build the initial values
	if typeof(init) == Array{Float64,1}
		assert(length(init) == m.bsize, "$(m.bsize) initial values expected, got $(length(init))")
		m.init = init
	elseif typeof(init) <: Real
		m.init = ones(m.bsize) * init
	else
		error("cannot assign initial values (should be a Real or vector of Reals)")
	end
end

######### returns an array of expr assigning parameters from the beta vector  ############
function betaAssign(m::MCMCModel)
	pmap = m.pars
	assigns = Expr[]
	for p in pmap
		if length(p.size) <= 1  # scalar or vector
			push!(assigns, :($(p.sym) = $PARAM_SYM[ $(expr(:quote,p.map)) ]) )
		else # matrix case  (needs a reshape)
			push!(assigns, :($(p.sym) = reshape($PARAM_SYM[ $(expr(:quote,p.map)) ], $(p.size[1]), $(p.size[2]))) )
		end
	end			
	assigns
end

######### evaluates once all variables to give type hints for derivation ############
#  most gradient calculation statements depend on dimensions of variables (Scalar or Array for example)
#  this is where they are evaluated (with values stored in global Dict 'vhint' )
function preCalculate(m::MCMCModel)
    global vhint = Dict()

    body = Expr[ betaAssign(m)..., 
                 :(local $ACC_SYM = 0.), 
                 m.exprs...]
    
    vl = getSymbols(body)  # list of all vars (external, parameters, set by model, and accumulator)
    body = vcat(body, [ :(vhint[$(expr(:quote, v))] = $v) for v in vl ])

	# enclose in a try block to catch zero likelihoods (-Inf log likelihood)
	body = expr(:try, expr(:block, body...),
			          :e, 
			          expr(:block, :(if e == "give up eval"; return(-Inf); else; throw(e); end)))

	# identify external vars and add definitions x = Main.x
	ev = m.accanc - m.varsset - Set(ACC_SYM, [p.sym for p in m.pars]...) # vars that are external to the model
	vhooks = expr(:block, [ :( local $v = $(expr(:., :Main, expr(:quote, v))) ) for v in ev]...) # assigment block

	# build and evaluate the let block containing the function and external vars hooks
	fn = gensym()
	body = expr(:function, expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	expr(:block, body) )
	body = :(let; global $fn; $vhooks; $body; end)
	eval(body)
	fn = eval(fn)

	# now evaluate vhint (or throw error if model does not evaluate for given initial values)
	assert(fn(m.init) != -Inf, "Initial values out of model support, try other values")
end


######### builds the model function ##############
function generateModelFunction(model::Expr, init, gradient::Bool, debug::Bool)
	## extracts model parameters, rewrites ~ , ..
	m = parseModel(model)

	## checks initial values
	setInit!(m, init)

	## process model
	unfold!(m)
	uniqueVars!(m)
	categorizeVars!(m)

	## build function expression
	if gradient  # case with gradient
		preCalculate(m)
		backwardSweep!(m)

		body = Expr[] # list of = expr making the model
		dsym(v::Symbol) = symbol("$DERIV_PREFIX$(v)")

		# initialization statements 
		body = [ betaAssign(m)...,        # assigments beta vector -> model parameter vars
		         :($ACC_SYM = 0.),        # initialize accumulator
		         :($(dsym(m.finalacc)) = 1.0)] # initialize accumulator gradient accumulator  

		avars = m.accanc & m.pardesc - Set(m.finalacc) # active vars without accumulator, treated above  
		for v in avars 
			vh = vhint[v]
			if isa(vh, Real)
				push!(body, :($(dsym(v)) = 0.))
			else	
				# push!(body, :($(dsym(v)) = zeros(Float64, $(expr(:quote,size(vh))))) )
				push!(body, :($(dsym(v)) = zeros(Float64, $(expr(:tuple,size(vh)...)))) )
				# dump( :($(dsym(v)) = zeros(Float64, $(expr(:quote,size(vh))))) )
			end
		end

		# build function statements, and move to let block constant statements for optimization
		header = Expr[]  # let block statements
		fvars = Set([e.args[1] for e in body]...) | Set(PARAM_SYM) # vars that are re-evaluated at each function call
		for ex in [m.exprs..., m.dexprs...]
			if length(getSymbols(ex.args[2]) & fvars) > 0
				push!(body, ex)
				fvars |= getSymbols(ex.args[1])
			else
				push!(header, ex)
			end
		end

		# prefix statements with 'local' at first occurence
		vars  = Set(PARAM_SYM)
		for i in 1:length(header)
			if length(getSymbols(header[i].args[1]) & vars) == 0
				header[i] = :(local $(header[i]))
				vars |= getSymbols(header[i].args[1])
			end
		end
		for i in 1:length(body)
			if length(getSymbols(body[i].args[1]) & vars) == 0
				body[i] = :(local $(body[i]))
				vars |= getSymbols(body[i].args[1])
			end
		end

		# return statement
		dexp = { :( vec([$(dsym(p.sym))]) ) for p in m.pars}
		dexp = length(m.pars) > 1 ? expr(:call, :vcat, dexp...) : dexp[1]
		push!(body, :(($(m.finalacc), $dexp)))

		# enclose in a try block
		body = expr(:try, expr(:block, body...),
				          :e, 
				          expr(:block, :(if e == "give up eval"; return(-Inf, zero($PARAM_SYM)); else; throw(e); end)))

	else  # case without gradient
		body = [ betaAssign(m)...,        # assigments beta vector -> model parameter vars
		         :(local $ACC_SYM = 0.),  # initialize accumulator
                 m.source.args...,        # model statements
                 :(return($ACC_SYM)) ]

		# enclose in a try block
		body = expr(:try, expr(:block, body...),
				          :e, 
				          expr(:block, :(if e == "give up eval"; return(-Inf); else; throw(e); end)))

		header = Expr[]
	end

	# identify external vars and add definitions x = Main.x
	ev = m.accanc - m.varsset - Set(ACC_SYM, [p.sym for p in m.pars]...) # vars that are external to the model
	# vhooks = expr(:block, [ :( local $v = $(expr(:., :Main, expr(:quote, v))) ) for v in ev]...) # assigment block
	header = [[ :( local $v = $(expr(:., :Main, expr(:quote, v))) ) for v in ev]..., header...] # assigment block

	# build and evaluate the let block containing the function and external vars hooks
	fn = gensym("ll")
	body = expr(:function, expr(:call, fn, :($PARAM_SYM::Vector{Float64})),	expr(:block, body) )
	body = expr(:let, expr(:block, :(global $fn), header..., body))

	debug ? body : (eval(body) ; (eval(fn), m.bsize, m.pars, m.init) )
end




		# # initialization statements 
		# body = [ betaAssign(m)...,        # assigments beta vector -> model parameter vars
		#          :(l $ACC_SYM = 0.),  # initialize accumulator
		#        :(local $(symbol("$DERIV_PREFIX$(m.finalacc)")) = 1.0)] # initialize accumulator gradient accumulator  

		# avars = m.accanc & m.pardesc - Set(m.finalacc) # active vars without accumulator, treated above  
		# for v in avars 
		# 	vh = vhint[v]
		# 	if isa(vh, Real)
		# 		push!(ie, :(local $(symbol("$DERIV_PREFIX$v")) = 0.))
		# 	else	
		# 		push!(ie, :(local $(symbol("$DERIV_PREFIX$v")) = zeros(Float64, $(size(vh)))) )
		# 	end
		# end

		# # model statements 'me'		
		# me = [[ :(local $e) for e in m.exprs]..., m.dexprs...]

		# # return statement
		# dexp = { :( vec([$(symbol("$DERIV_PREFIX$(p.sym)"))]) ) for p in m.pars}
		# dexp = length(m.pars) > 1 ? expr(:call, :vcat, dexp...) : dexp[1]
		# # 	dn = symbol("$DERIV_PREFIX$(m.pars[1].sym)")
		# # 	dexp = :(vec([$dn]))  # reshape to transform potential matrices into vectors
		# # else
		# # 	dexp = {:vcat}
		# # 	dexp = vcat(dexp, { :( vec([$(symbol("$DERIV_PREFIX$(p.sym)"))]) ) for p in m.pars})
		# # 	dexp = expr(:call, dexp)
		# # end

		# body = vcat(ie, me)
		# push!(body, :(($(m.finalacc), $dexp)))

		# # enclose in a try block
		# body = expr(:try, expr(:block, body...),
		# 		          :e, 
			