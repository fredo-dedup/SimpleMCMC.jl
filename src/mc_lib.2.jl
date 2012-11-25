
####################################################################################

function expexp(ex::Expr, ident...)
	ident = (length(ident)==0 ? 0 : ident[1])::Integer
	println(rpad("", ident, " "), ex.head, " -> ")
	for e in ex.args
		typeof(e)==Expr ? expexp(e, ident+3) : println(rpad("", ident+3, " "), "[", typeof(e), "] : ", e)
	end
end

####################################################################

function mcmc(ex)
	if typeof(ex) == Expr
		if ex.args[1]== :~
			ex = :(__lp = __lp + $(Expr(:call, 
				{symbol(strcat("mlog_", ex.args[3].args[1])), ex.args[2], 
				#(ex.args[3].args[2:end])}, Any)))
				ex.args[3].args[2:end]...}, Any)))
		else 
			ex = Expr(ex.head, { mcmc(e) for e in ex.args}, Any)
		end
	end
	ex
end
#expexp(:(4 + 3), 0)
#mcmc(:(4 + 3))
#mcmc(:(yo ~ normal(mu, sigma)))

function mcmc2(ex)
	if typeof(ex) == Expr
		if ex.args[1]== :~
			ex = :(__lp = __lp + sum(logpdf($(ex.args[3]), $(ex.args[2]))))
		else 
			ex = Expr(ex.head, { mcmc2(e) for e in ex.args}, Any)
		end
	end
	ex
end

function mlog_normal(x::Float64, sigma::Float64) 
	tmp = x / sigma
	-tmp*tmp 
end

function mlog_normal(x::Vector{Float64}, sigma::Float64) 
	tmp = x / sigma
	- sum(tmp .* tmp)
end

##########################################################
	srand(1)
	n    = 10000
	nbeta = 40
	X    = [fill(1, (n,)) randn((n, nbeta-1))]
	beta0 = randn((nbeta,))
	Y = X * beta0 + randn((n,))


#################   model 2 #################
	model = quote
		resid = Y - X * vars
		sigma ~ Gamma(2, 1)

		vars ~ Normal(0.0, 1)
		resid ~ Normal(0.0, sigma)
	end

	modelExpr = mcmc2(model)
	setmap = :(sigma=beta[1] ; vars=beta[2:(nbeta+1)])

	loop = quote 
	 	for __i in 1:steps
			oldbeta, beta = beta, beta + randn(1+nbeta) * scale

			$setmap
			# eval(setmap)
	 		old__lp, __lp = __lp, 0.0

			$modelExpr
			# eval(modelExpr)
			if rand() > exp(__lp - old__lp)
				__lp, beta = old__lp, oldbeta
			end
			samples[__i, :] = vcat(__lp, beta)
		end
	end
loop
