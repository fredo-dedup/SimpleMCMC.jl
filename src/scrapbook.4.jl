
expexp(model2)
expexp(:(a+b+c))
expexp(:(sum(a,b,c)))
expexp(:(if a==b ; c=d; elseif a<b ; c=2d; else c=3d; end))
expexp(:(for i in 1:10; b=3.0;end))
expexp(:(while d > 4; d+= 1;end))

##########  unfoldBlock ##############

function unfoldBlock(ex::Expr)
	assert(ex.head == :block, "[unfoldBlock] not a block")

	lb = {}
	for e in ex.args  #  e  = ex.args[2]
		if e.head == :line   # line number marker, no treatment
			push(lb, e)
		elseif e.head == :(=)  # assigment
			assert(typeof(e.args[1]) == Symbol, "not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			rhs = e.args[2]
			if typeof(rhs) == Expr
				ue = unfoldExpr(rhs)
				if length(ue)==1 # no variable creation necessary
					push(lb, e)
				else
					for a in ue[1:end-1]
						push(lb, a)
					end
					push(lb, :($(e.args[1]) = $(ue[end])))
				end
			elseif typeof(rhs) == Symbol
				push(lb, e)
			else
				push(lb, e)
			# 	error("[unfoldBlock] can't parse $e")
			end

		elseif e.head == :for  # TODO parse loops

		elseif e.head == :while  # TODO parse loops

		elseif e.head == :if  # TODO parse if structures

		elseif e.head == :block
			e2 = unfoldBlock(e)
			push(lb, e2)
		else
			error("[unfoldBlock] can't handle $(e.head) expressions")
		end
	end

	Expr(:block, lb, Any)
end

function localVars(ex::Expr)
	assert(ex.head == :block, "[localVars] not a block")

	lb = {}
	for e in ex.args  #  e  = ex.args[2]
		if e.head == :(=)  # assigment
			assert(typeof(e.args[1]) == Symbol, "[localVars] not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			contains(lb, e.args[1]) ? nothing : push(lb, e.args[1])

		elseif e.head == :line  # TODO parse loops

		elseif e.head == :for  # TODO parse loops

		elseif e.head == :while  # TODO parse loops

		elseif e.head == :if  # TODO parse if structures

		elseif e.head == :block
			e2 = localVars(e)
			for v in e2
				push(lb, v)
			end
		else
			error("[localVars] can't handle $(e.head) expressions")
		end
	end

	lb
end



##########  backwardSweep ##############

function backwardSweep(ex::Expr, locals::Vector)
	assert(ex.head == :block, "[backwardSweep] not a block")

	lb = {}
	for e in ex.args  #  e  = ex.args[2]
		if e.head == :line   # line number marker, no treatment
			push(lb, e)
		elseif e.head == :(=)  # assigment
			dsym = e.args[1]
			assert(typeof(dsym) == Symbol, "[backwardSweep] not a symbol on LHS of assigment $(e)") # TODO : manage symbol[]
			
			rhs = e.args[2]
			if typeof(rhs) == Expr
				println
				for i in 2:length(rhs.args)
					vsym = rhs.args[i]
					if contains(locals, vsym)
						push(lb, derive(rhs, i-1, dsym))
					end
				end
			elseif typeof(rhs) == Symbol
				dsym2 = symbol("__d$dsym")
				vsym2 = symbol("__d$rhs")
				push(lb, :( $vsym2 = $dsym2) )
			else 
			end

		elseif e.head == :for  # TODO parse loops
			# e2 = backwardSweep(e)
			# push(lb, e2)

		elseif e.head == :while  # TODO parse loops
			# e2 = backwardSweep(e)
			# push(lb, e2)

		elseif e.head == :if  # TODO parse if structures
			# e2 = backwardSweep(e.args[2])
			# push(lb, e2)

		elseif e.head == :block
			e2 = backwardSweep(e, locals)
			push(lb, e2)
		else
			error("[backwardSweep] can't handle $(e.head) expressions")
		end
	end

	Expr(:block, reverse(lb), Any)
end

function derive(opex::Expr, index::Integer, dsym::Symbol)
	op = opex.args[1]  # operator
	vsym = opex.args[1+index]
	vsym2 = symbol("__d$vsym")
	dsym2 = symbol("__d$dsym")

	if op == :+
		return :($vsym2 += $dsym2)
	elseif op == :-
		if length(opex.args) == 2  # unary minus
			return :($vsym2 += -$dsym2)
		elseif index == 1
			return :($vsym2 += $dsym2)
		else
			return :($vsym2 += -$dsym2)
		end
	elseif op == :^
		if index == 1
			e = opex.args[3]
			if e == 2.0
				return :($vsym2 += 2 * $vsym * $dsym2)
			else
				return :($vsym2 += $e * $vsym ^ $(e-1) * $dsym2)
			end
		else
			v = opex.args[2]
			return :($vsym2 += log($v) * $v ^ $vsym * $dsym2)
		end
	elseif op == :*
		e = :(1.0)
		for i in 2:length(opex.args)
			if i != index+1
				if i < index+1
					e = :($e * $(opex.args[i])')
				else	
					e = :($(opex.args[i]) * $e)
				end
			end
		end
		return :($vsym2 += $e * $dsym2)
	elseif op == :dot
		e = index == 1 ? opex.args[3] : opex.args[2]
		return :($vsym2 += sum($e) .* $dsym2)
	elseif op == :log
		return :($vsym2 += $dsym2 ./ $vsym)
	elseif op == :exp
		return :($vsym2 += exp($vsym) .* $dsym2)
	elseif op == :/
		if index == 1
			e = opex.args[3]
			return :($vsym2 += $vsym ./ $e .* $dsym2)
		else
			v = opex.args[2]
			return :($vsym2 += - $v ./ ($vsym .* $vsym) .* $dsym2)
		end
	else
		error("Can't derive operator $op")
	end
end


##########  unfold Expr ##############
function unfoldExpr(ex::Expr)
	assert(typeof(ex) == Expr, "[unfoldExpr] not an Expr $(ex)")
	assert(ex.head == :call, "[unfoldExpr] call expected  $(ex)")

	lb = {}
	na = {ex.args[1]}   # function name
	for e in ex.args[2:end]  # e = :b
		if typeof(e) == Expr
			ue = unfoldExpr(e)
			for a in ue[1:end-1]
				push(lb, a)
			end
			nv = consume(nameTask)
			push(lb, :($nv = $(ue[end])))
			push(na, nv)
		else
			push(na, e)
		end
	end
	push(lb, Expr(ex.head, na, Any))
	lb
end

function nameFactory()
	for i in 1:10000
		produce(symbol("__t$i"))
	end
end

nameTask = Task(nameFactory)
@assert isequal(unfoldExpr(:(3 + x)), [:(3+x)])
@assert isequal(unfoldExpr(:(3*b + x)), [:(__t1 = 3*b), :(__t1 + x)])


##########  tests ##############

nameTask = Task(nameFactory)
# consume(nameTask)

ex=:(a+b)
unfoldExpr(:(a+b))
unfoldExpr(:(3*a+b))

ex = quote
	a = b +c
	d = 2f - sin(y)
	ll = log(a+d)
end

unfoldBlock(ex)

ex = quote
	a = b +c
	begin
		z = 3
		k = z*z + x
	end
	d = 2f - sin(y)
	ll = log(a+d)
end

unfoldBlock(ex)

ex = quote
	(y = x + 13 ; z = y^2 + x ; ll = z)
end

ex2 = unfoldBlock(ex)
vars = localVars(ex2)
backwardSweep(ex2, vars)
beta1, beta2 = 1.1, 3.0
(eval(ex) - 136.75) * 10 # 23.80 ok

ex = quote
	a = 5 + z
	y = 78
	z = a * z * y
end
vars = localVars(unfoldBlock(ex))
backwardSweep(unfoldBlock(ex), vars)

ex = quote
	a = 5 + beta1
	y = 7beta2 + log(a)
	z = a * y
end
ex2 = unfoldBlock(ex)
vars = localVars(ex2)
push(vars, :beta1)
push(vars, :beta2)
ex3 = backwardSweep(ex2, vars)

finalexp = quote
	$ex2
	$(Expr(:block, {:($(symbol("__d$v")) = 0.0) for v in vars}, Any))  
	__dz = 1.0
	# y, __dz, __da, __dt9, __dt10 = 0.0, 1.0, 0.0, 0.0, 0.0
	$ex3
	(z, __dbeta1, __dbeta2)
end

beta1, beta2 = 1.0, 3.0
eval(finalexp)
# (136.75055681536833,23.791759469228055,42.0)

eval(ex) #136
beta1, beta2 = 1.1, 3.0
(eval(ex) - 136.75) * 10 # 23.80 ok
beta1, beta2 = 1.0, 3.1
(eval(ex) - 136.75) * 10 # 42.005 ok

ref_time = timefactor(ex, 1000000)  # 0.26 sec
timefactor(finalexp, 1000000, ref_time) # x3.5

##############################################################
begin
	srand(1)
	n = 100
	nbeta = 4
	X = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = (1/(1+exp(-(X * beta0)))) .> rand(n)
end

model = quote
	ll = - dot(beta, beta)
	resid = Y - (1/(1+exp(- X * beta)))
	ll = ll - dot(resid, resid)
end

ex2 = unfoldBlock(model)
vars = localVars(ex2)
push(vars, :beta)
ex3 = backwardSweep(ex2, vars)

finalexp = quote
	$ex2
	$(Expr(:block, {:($(symbol("__d$v")) = zero($(symbol("$v")))) for v in vars}, Any))  
#	$(Expr(:block, {:($(symbol("__d$v")) = 0.0) for v in vars}, Any))  
	__dll = 1.0
    __d__t27 += __dll
    __dll += __dll
    __dresid += .*(sum(resid), __d__t27)
    __dresid += .*(sum(resid), __d__t27)	#  line 4:
    __d__t26 += -(__dresid)
    __d__t25 += .*(./(-(1), .*(__t25, __t25)), __d__t26)
    __d__t24 += __d__t25
    __d__t23 += .*(exp(__t23), __d__t24)
    __dbeta += *(*(1.0, __t22'), __d__t23)
    __d__t22 += *(__d__t23, *(beta, 1.0)')	#  line 3:
    __d__t21 += -(__dll)
    __dbeta += .*(sum(beta), __d__t21)
    __dbeta += .*(sum(beta), __d__t21)	#  line 2:
	(ll, __dbeta)
end

beta = ones(4)
eval(finalexp)
eval(ex2)

function HMC(finalexp::Expr, epsilon, L, current_beta)
	beta = current_beta
	p = randn(nbeta) # independent standard normal variates
	current_beta = p

	(ll, dll) = eval(finalexp)	
	# Make a half step for momentum at the beginning
	p -= epsilon * dll / 2
	# Alternate full steps for position and momentum
	for (i in 1:L)
		# Make a full step for the position
		beta += epsilon * p
		# Make a full step for the momentum, except at end of trajectory
		if i!=L
			p -= epsilon * dll
		end
	end
	# Make a half step for momentum at the end.
	p -= epsilon * dll / 2
	# Negate momentum at end of trajectory to make the proposal symmetric
	p = -p
	# Evaluate potential and kinetic energies at start and end of trajectory

	(ll, dll) = eval(finalexp)	
	current_U = U(current_q)
	current_K = sum(current_p^2) / 2
	proposed_U = U(q)
	proposed_K = sum(p^2) / 2
	# Accept or reject the state at end of trajectory, returning either
	# the position at the end of the trajectory or the initial position
	if (rand(1) < exp(current_U-proposed_U+current_K-proposed_K))
	{
	return (q) # accept
	}
	else
	{
	return (current_q) # reject
	}
}


