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
	x.converged ? nothing : print("convergence precision not reached, ")
	println("$(x.steps) iterations, $(round(x.time,1)) sec.")
end

##########################################################################################
#   Accelerated Gradient Descent solver
#
#	Ref : Adaptive Restart for Accelerated Gradient Schemes - Donoghue/Candes
#
##########################################################################################
# init = [1., 0.01, 1.0]
# maxiter = 50
# precision = 1e-10

function simpleAGD(model::Expr, init::Any, maxiter::Integer, precision::Float64)
	tic() # start timer

	ll_func, nparams, pmap, z0 = generateModelFunction(model, init, true, false) # build function, count the number of parameters

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
	while i<maxiter && !converged  
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

    res = SolverRun(toq(), i, f1, i<maxiter, Dict(), Dict())
	for p in pmap
		res.params[p.sym] = z0[ p.map]  #[1]:p.map[2]]
	end

	res
end

simpleAGD(model::Expr, init::Any, maxiter::Integer) = simpleAGD(model, init, maxiter, 1e-5)
simpleAGD(model::Expr, init::Any) = simpleAGD(model, init, 100, 1e-5)
simpleAGD(model::Expr) = simpleAGD(model, 1.0, 100, 1e-5)


##########################################################################################
#
#   Nelder-Mead optimization
#
#    translated and simplified from : http://people.sc.fsu.edu/~jburkardt/m_src/asa047/nelmin.m
#
#   convergence criterion = L1 norm of simplex < precision
#
##########################################################################################
# TODO : manage function support exit (using backtracking ?)

function simpleNM(model::Expr, init::Any, maxiter::Integer, precision::Float64)  
	tic() # start timer

	assert(precision>0., "precision should be > 0.")

	func, n, pmap, init = generateModelFunction(model, init, false, false) # build function, count the number of parameters
	assert(n>=1, "there should be at least one parameter")

	# first calc
	f0 = -func(init)
	assert(isfinite(f0), "Initial values out of model support, try other values")

	step = ones(length(init)) #  no scaling on parameters by default

	#  Nelder-Mead coefs
	ccoeff = 0.5
	ecoeff = 2.0
	rcoeff = 1.0

	stop_crit() = all( [max(p[i,:])-min(p[i,:]) for i in 1:n] .< precision) # precision criterion

	# loop inits
	p = Float64[ init[i] + step[i] * (i==j) for i in 1:n, j in 1:(n+1)]
	y = Float64[ -func(p[:,j]) for j in 1:(n+1)]
	ilo = indmin(y); ylo = y[ilo]

	it = 0
	while it < maxiter && !stop_crit()
 		progress(it, maxiter, 0)

		ihi = indmax(y)

		#  Calculate PBAR, the centroid of the simplex vertices
		#  excepting the vertex with Y value YNEWLO.
		pbar = Float64[sum(p[i,1:end .!= ihi]) for i in 1:n] / n 

		#  Reflection through the centroid.
		pstar = pbar + rcoeff * (pbar - p[:,ihi])
		ystar = -func(pstar)

	    if ystar < ylo #  Successful reflection, so extension.
	    	p2star = pbar + ecoeff * (pstar - pbar)
	    	y2star = -func(p2star)

			#  Check extension.
			p[:,ihi] = ystar < y2star ? pstar : p2star
			y[ihi] = ystar < y2star ? ystar : y2star

		else #  No extension.
			l = sum(ystar .< y)

			if l > 1
				p[:,ihi] = pstar
				y[ihi] = ystar

		    elseif l == 0 #  Contraction on the Y(IHI) side of the centroid.
			    p2star = pbar + ccoeff * ( p[:,ihi] - pbar )
			    y2star = -func(p2star)

				if y[ihi] < y2star #  Contract the whole simplex.
					p = Float64[ (p[i,j] + p[i, ilo]) * 0.5 for i in 1:n, j in 1:(n+1)]
					y = Float64[ -func(p[:,j]) for j in 1:(n+1)]

				else #  Retain contraction.
					p[:,ihi] = p2star
					y[ihi] = y2star
				end

	        elseif l == 1  #  Contraction on the reflection side of the centroid.
				p2star = pbar + ccoeff * ( pstar - pbar )
				y2star = -func(p2star)

				#  Retain reflection?
				p[:,ihi] = ystar < y2star ? pstar : p2star
				y[ihi] = ystar < y2star ? ystar : y2star
			end
		end

		ilo = indmin(y); ylo = y[ilo]
		it += 1

		# println("$it : $((ylo, p[:,ilo])) ; crit = $(max([max(p[i,:])-min(p[i,:]) for i in 1:n]))")
	end
	println()
	
    res = SolverRun(toq(), it, ylo, it<maxiter, Dict(), Dict())
	for par in pmap
		res.params[par.sym] = p[ilo, par.map]
	end

	res
end

simpleNM(model::Expr, init::Any, maxiter::Integer) = simpleNM(model, init, maxiter, 1e-3)
simpleNM(model::Expr, init::Any) = simpleNM(model, init, 100, 1e-3)
simpleNM(model::Expr) = simpleNM(model, 1.0, 100, 1e-3)
