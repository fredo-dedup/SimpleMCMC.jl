##########################################################################
#
#    Solving functions
#
##########################################################################

##########################################################################################
#   Nesterov accelerated gradient
##########################################################################################

function simpleNesterov(model::Expr, init::Any)
    local epsilon, u_slice
    local state0  # starting state of each loop
	
	tic() # start timer
	
	ll_func, nparams, pmap = buildFunctionWithGradient(model) # build function, count the number of parameters
	x0 = setInit(init, nparams) # build the initial values
	y0 = copy(x0)

	# first calc
    y0llik, grad = ll_func(s.beta)
    q = 0.
    theta0 = 1.

    x1 = x0
    y1 = y0
    for i in 1:10
    	x1 = y0 - tky
    end

	### main loop
 	for i in 1:steps  # i=1
 		local alpha, nalpha, n, s, j, n1, s1
 		local dummy, state_minus, state_plus, state, state1

 		state0.v = randn(nparams)
 		update!(state0)

 		u_slice  = log(rand()) + state0.H # use log ( != paper) to avoid underflow
 		
 		state = state_minus = state_plus = state0

 		# inner loop
 		j, n = 0, 1
 		s = true
 		while s && j < 12
 			dir = (rand() > 0.5) * 2. - 1.
 			if dir == -1
 				state_minus, dummy, state1, n1, s1, alpha, nalpha = buildTree(state_minus, dir, j, ll_func)
 			else
 				dummy, state_plus, state1, n1, s1, alpha, nalpha = buildTree(state_plus, dir, j, ll_func)
 			end
 			if s1 && rand() < n1/n  # accept 
 				state = state1
 			end
 			n += n1
 			j += 1
 			s = s1 && !uturn(state_minus, state_plus)
 		end
 		
 		# epsilon adjustment
 		if i <= nadapt  # warming up period
 			hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
			le = mu-sqrt(i)/gam*hbar
			lebar = i^(-kappa) * le + (1-i^(-kappa)) * lebar
			epsilon = exp(le)
		else # post warm up, keep dual epsilon
			epsilon = exp(lebar)
		end

		# println(llik, beta)
		i > burnin ? addToRes!(res, pmap, i-burnin, state.llik, state != state0, state.beta) : nothing

		res.misc[:epsilon][i] = epsilon
		res.misc[:jmax][i] = j

		state0 = state
	end

	calcStats!(res)
	res
end

simpleNesterov(model::Expr, steps::Integer) = simpleNUTS(model, steps, min(steps-1, div(steps,2)))
simpleNUTS(model::Expr, steps::Integer, burnin::Integer) = simpleNUTS(model, steps, burnin, 1.0)
