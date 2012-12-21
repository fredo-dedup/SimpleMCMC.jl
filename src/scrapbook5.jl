include("SimpleMCMC.jl")
using SimpleMCMC

##########  tests ##############
nameTask = Task(nameFactory)
@assert isequal(unfoldExpr(:(3 + x)), [:(3+x)])
@assert isequal(unfoldExpr(:(3*b + x)), [:(__t1 = 3*b), :(__t1 + x)])

# consume(nameTask)

# start with beta active variable
# proceed through new definitions, marking dependant variables

unfoldExpr(:(a+b))
unfoldExpr(:(a+12*b))

#############################################
ex = quote
	b::scalar
	ll::vector(10)

	a[12] = b
	c = beta[3]
	d[j] = ll[k:(k+5)]
	z = sin(beta) / exp(a[45]) * c ^ (d + 5 +k)
	z ~ Normal(0,1)
end
(ex2, param_map, nparams) = parseExpr(ex)
ex3 = unfoldBlock(ex2)
avars = findActiveVars(ex3, map(x->x.args[1], param_map))
push(vars, :__beta)
ex4 = backwardSweep(ex3, avars)

ex = quote
	a = __beta[12]
end
(ex2, param_map, nparams) = parseExpr(ex)
(ex3, vars) = unfoldBlock(ex2)
push(vars, :__beta)
ex4 = backwardSweep(ex3, vars)



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
    __d__t22 += *(__d__t23, *(beta, 1.0)')	#  line 3: #'
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

