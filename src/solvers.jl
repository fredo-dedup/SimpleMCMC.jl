##########################################################################
#
#    Solving functions
#
##########################################################################




# simulate dataset
srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = rand(n) .< ( 1 ./ (1. + exp(X * beta0)))

# define model
model = quote
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, std 1.0 for predictors
	prob = 1 / (1. + exp(X * vars)) 
	Y ~ Bernoulli(prob)
end

# run random walk metropolis (10000 steps, 1000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000, 1000)
sol = Base.amap(mean, res.params[:vars], 1)

####

ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters

begin
	x0 = ones(nparams) # build the initial values
	y0 = copy(x0)
	lambda0 = 0.
    f0, grad = ll_func(y0)
	beta0 = 1.  # 
	
    for i in 1:100  # i=1
    	f1, gradx = ll_func(x0)

    	lambda1 = (1+sqrt(1+4*lambda0))/2.
    	fac = (1-lambda0)/lambda1

    	y1 = x0 - grad / beta0
    	x1 = (1-fac)*y1 + fac*y0

    	fy, grady = ll_func(y1)

    	beta1 = abs(dot(x0-y1, grady-gradx)) > beta0*dot(x0-y1, x0-y1)/2 ? beta0 : beta0*2.

 		println(i, " : ", round(log10(abs(f1-f0)),2), ",   beta =", beta1, ",   f =", f1)
 		# if dot(grad, x1-x0)<0. #  restart needed
 		if dot(x1-y0, x1-x0)<0. #  restart needed
 			println("=== restart")
    		x0, y0, f0, beta0, lambda0 = x1, x1, f1, beta1, 0.0
 		else
    		x0, y0, f0, beta0, lambda0 = x1, y1, f1, beta1, lambda1
 		end
    end
end

[x0 beta0]
ll_func(beta0)
ll_func(x0)














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

    for i in 1:10
    	dummy, grad = ll_func(y0)
    	x1 = y0 - tk .* grad
    	y1 = x1 + (i-1)/(i+2)(x1-x0)
    	x0, y0 = x1, y1
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
