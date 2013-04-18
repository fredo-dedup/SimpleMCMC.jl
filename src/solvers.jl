##########################################################################
#
#    Solving functions
#
##########################################################################

###### returned result structure ##########
type SolverRun
    time::Float64
    steps::Integer
    maximum::Float64
    converged::Bool
    params::Dict
    misc::Dict
end

function show(io::IO, x::SolverRun)
	for v in keys(x.params)
		print("$v= $(x.params[v])  ")
	end
	println()
	println("max. iter reached $(x.converged), $(x.steps) iterations, $(round(x.time,1)) sec.")
end

##########################################################################################
#   Accelerated Gradient Descent solver
##########################################################################################
init = [1., 0.01, 1.0]
maxiter = 50
precision = 1e-10

function simpleAGD(model::Expr, init::Any, maxiter::Integer, precision::Float64)
	tic() # start timer
	
	ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
	z0 = SimpleMCMC.setInit(init, nparams) # build the initial values

	# first calc
	f0, grad0 = ll_func(z0)
	assert(isfinite(f0), "Initial values out of model support, try other values")

	# find initial Lipschitz constant L0
	L0 = 1.0
	f1, grad1 = ll_func(z0 + grad0 / L0)
	while !isfinite(f1) # find defined point
		L0 = L0*2.
		f1, grad1 = ll_func(z0 + grad0 / L0)
	end
	tmp = grad1 - grad0
	L00 = sqrt(dot(tmp, tmp) / dot(grad0, grad0)) * L0  # second estimate
	L0 = max(L0, L00)

	# gradient descent loop
	zb0 = copy(z0)
	theta0 = 1.
	converged = false
	i = 0
	while i<maxiter && !converged   #for i in 0:maxiter # i=1
		y0 = (1-theta0).*z0 + theta0.*zb0
		fy1, grady = ll_func(y0)
		zb1 = zb0 + grady / (theta0 * L0)
		z1 = (1-theta0).*z0 + theta0.*zb1

		fz1, gradz = ll_func(zb1)
		h = 0
		while !isfinite(fz1) && h < 10 # grow L0 if we exited support
			L0 = L0 * 2. 
			zb1 = zb0 + grady / (theta0 * L0)
			z1 = (1-theta0).*z0 + theta0.*zb1

			fz1, gradz = ll_func(zb1)
			print("+")
			h += 1
		end
	 	assert(h<10, "Convergence failed, exited function support")

		# adapt L for convergence guarantee
		f1, gradz = ll_func(z1)
		L1 = 2*abs(dot(y0-z1, grady-gradz))/dot(y0-z1, y0-z1) > L0 ? L0*2. : L0*0.9

		theta1 = 2 / (1+sqrt(1+4*L1/(L0*theta0*theta0)))
	 	print(i, " : ", round(log10(abs(f1-f0)),2), " theta =", round(theta0,3), ", L =", L1, " f =", round(fy1,3))
	 	converged = abs(f1-f0) < precision

 		if dot(grady, z1-z0) < 0. #  restart needed
 			println(" === restart")
			z0, zb0, theta0, L0, f0 = z1, z1, 1., L1, f1
 		else
 			println("")
			z0, zb0, theta0, L0, f0 = z1, zb1, theta1, L1, f1
 		end
 		i += 1
	end

    res = SolverRun(toq(), i, f1, i==maxiter, Dict(), Dict())
	for p in pmap
		res.params[p.sym] = z0[ p.map]  #[1]:p.map[2]]
	end

	res
end

simpleAGD(model::Expr, init::Any, maxiter::Integer) = simpleAGD(model, init, maxiter, 1e-5)
simpleAGD(model::Expr, init::Any) = simpleAGD(model, init, 100, 1e-5)
simpleAGD(model::Expr) = simpleAGD(model, 1.0, 100, 1e-5)



