##########################################################################################
#
#    Definition of distributions
#
##########################################################################################

########## locally defined distributions #############

function logpdfBernoulli(prob::Real, x::Real)
	prob > 1. || prob < 0. ? error("calling Bernoulli with prob > 1. or < 0.") : nothing
	if x == 0.
		prob == 1. ? throw("give up eval") : return(log(1. - prob))
	elseif x == 1.
		prob == 0. ? throw("give up eval") : return(log(prob))
	elseif
	 	error("calling Bernoulli with variable other than 0 or 1 (false or true)")
	end
end

function logpdfNormal(mu::Real, sigma::Real, x::Real)
	local const fac = -log(sqrt(2pi))
	if sigma <= 0.
		error(string("calling logpdfNormal with negative or null stdev"))
	end
	return -(x-mu)*(x-mu)/(2sigma*sigma)-log(sigma)+fac
end


########### distributions using libRmath ######### 

_jl_libRmath = dlopen("libRmath")

#          Name           libRmath Name    Arity       !!! empty lib name means local def
dists = { (:Weibull, 	  "dweibull",       3),
		  (:Uniform, 	  "dunif",          3),
		  (:Binomial, 	  "dbinom",         3),
		  (:Gamma,  	  "dgamma",         3),
		  (:Cauchy,  	  "dcauchy",        3),
		  (:logNormal,    "dlnorm",         3),
		  (:Beta, 	      "dbeta",          3),
		  (:Poisson,  	  "dpois",          2),
		  (:TDist,  	  "dt",             2),
		  (:Exponential,  "dexp",           2),
	      (:Normal,  	  "",               3),     #  "dnorm4" in libRmath
		  (:Bernoulli,    "",               2)}


for d in dists 
	if d[2] != "" 
		fsym = symbol("logpdf$(d[1])")

		npars = d[3]

		fex = quote
			function ($fsym)($([expr(:(::), symbol("x$i"), :Real) for i in 1:npars]...))
				local res = 0.

		        res = ccall(dlsym(_jl_libRmath, $(string(d[2]))), Float64,
		            	 	  $(expr(:tuple, [[:Float64 for i in 1:npars]..., :Int32 ]...)),
		            	 	  $([symbol("x$i") for i in 1:npars]...), 1	)

				if res == -Inf
					throw("give up eval")
				elseif res == NaN
					local args = $(expr(:tuple, [[symbol("x$i") for i in 1:npars]..., 1]...))
					error(string("calling ", $(string(fsym)), args, "returned an error"))
				end
				res 
			end
		end

		eval(fex)
	end
end

# TODO : factorize
# for d in {#(:Normal,  	"dnorm4"),
# 		  (:Weibull, 	"dweibull"),
# 		  (:Uniform, 	"dunif"),
# 		  (:Binomial, 	"dbinom"),
# 	      (:Gamma,  	"dgamma"),
# 	      (:Cauchy,  	"dcauchy"),
# 	      (:logNormal,  "dlnorm"),
# 		  (:Beta, 	    "dbeta")}

# 	fsym = symbol("logpdf$(d[1])")

# 	eval(quote

# 		function ($fsym)(a::Real, b::Real, x::Real)
# 			local res
# 			res = ccall(dlsym(_jl_libRmath, $(d[2])), Float64, 
# 				(Float64, Float64, Float64, Int32), 
# 					x, a, b, 1)
# 			if res == -Inf
# 				throw("give up eval")
# 			elseif res == NaN
# 				error(string("calling ", $fsym, "with $x, $a, $b returned an error"))
# 			else
# 				return(res)
# 			end
# 		end

# 	end) 

# end

# for d in {(:Poisson,  	  "dpois"),
# 	      (:TDist,  	  "dt"),
# 	      (:Exponential,  "dexp")}

# 	fsym = symbol("logpdf$(d[1])")

# 	eval(quote

# 		function ($fsym)(a::Real, x::Real)
# 			local res
# 			res = ccall(dlsym(_jl_libRmath, $(d[2])), Float64, 
# 				(Float64, Float64, Int32), 
# 					x, a, 1)
# 			if res == -Inf
# 				throw("give up eval")
# 			elseif res == NaN
# 				error(string("calling ", $fsym, "with $x, $a returned an error"))
# 			else
# 				return(res)
# 			end
# 		end

# 	end) 

# end


for d in dists # d = dists[1]
	fsym = symbol("logpdf$(d[1])")

	arity = d[3]
	ps = [ ifloor((i-1) / 2^(j-1)) % 2 for i=2:2^arity, j=1:arity]

	for l in 1:size(ps,1) # l = 3
		pars = Symbol[ps[l,j]==0 ? :Real : :AbstractArray for j in 1:arity]

		rv = symbol("x$(findfirst(pars .== :AbstractArray))")
		npars = length(pars)

		# mf = expr(:function, 
		#           expr(:call, fsym, 
		#                 [expr(:(::), symbol("x$i"), pars[i]) for i in 1:npars]...),
		#           expr(:block, 
		#                 :(local res = 0.),
		#                 expr(:for, 
		#                      expr(:(=), :i, expr(:(:), 1, :(length($rv)))),
		#                      expr(:block, 
		#                           expr(:(+=), 
		#                           	   :res, 
		#                                expr(:call, 
		#                                	    fsym, 
		#                                     [pars[i]==:Real ? symbol("x$i") : expr(:ref, symbol("x$i"), :i) for i in 1:npars]...))
		#                          )
		#                     ), 
		#                  :(res) 
		#                 ) 
		#            )

		fex = quote
			function ($fsym)($([expr(:(::), symbol("x$i"), pars[i]) for i in 1:npars]...)) 
		        local res = 0.
		        for i in 1:length($rv)
		        	res += ($fsym)($([pars[i]==:Real ? symbol("x$i") : expr(:ref, symbol("x$i"), :i) for i in 1:npars]...))
		        end
		        res
		    end
		end

		eval(fex)
	end

end



############# dummy distrib for testing ############

logpdfTestDiff(x) = sum([x]) 

