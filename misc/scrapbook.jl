################################

include("../src/SimpleMCMC.jl")

function recap(res)
    print("ess/sec. $(map(iround, res.essBySec)), ")
    print("mean : $(round(mean(res.params[:x]),3)), ")
    println("std : $(round(std(res.params[:x]),3))")
end

model = :(x::real ; x ~ Weibull(1, 1))  # mean 1.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 3.400 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 2, 0.8)) # 6.100 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 400 ess/s

model = :(x::real ; x ~ Weibull(3, 1)) # mean 0.89, std 0.325
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 6.900 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.6], 2, 0.3)) # 84.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 22.000 ess/s, correct

model = :(x::real ; x ~ Uniform(0, 2)) # mean 1.0, std 0.577
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 6.800 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 1, 0.9)) # 12.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 10000, 1000, [1.]))  # 400 ess/s, very slow due to gradient == 0 ?

model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 16.000 ess/s  7500
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 93.000 ess/s, 49-50.000
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 35.000 ess/s, correct, 22-24.0000

model = :(x::real ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 16.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 9.)) # 95.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 33.000 ess/s, correct

model = :(x::real(10) ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, 0.))  # 1.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, 0., 3, 7.)) # 30.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, 0.))  # 9.000 ess/s


z = [ rand(100) .< 0.5]
model = :(x::real ; x ~ Uniform(0,1); z ~ Bernoulli(x)) # mean 0.5, std ...
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.5]))  # 5.100 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.5], 2, 0.04)) # 10.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.5]))  # 6.600 ess/s, correct




###################### neldermead 


include("../src/SimpleMCMC.jl")
ex = expr(:., :SimpleMCMC, expr(:quote, symbol("##ll#56843")))
eval(ex)

res = SimpleMCMC.simpleNM(:(x::real ; -(x-2.6)^2))


reqmin = 0.1
n = length(init)
konvge = 5



######################################################
begin
	func(v) = ((v[2]-1)^2*5+(v[1]-2)^2)
	init = [0.1,10]

	reqmin = 0.1
	step = ones(length(init))

	##
	assert(reqmin>0.)


	n = length(init)
	assert(n>=1)

	ccoeff = 0.5
	ecoeff = 2.0
	rcoeff = 1.0


	maxit = 100

	p = [ init[i] + step[i] * (i==j) for i in 1:n, j in 1:(n+1)]
	y = Float64[ func(p[:,j]) for j in 1:(n+1)]

	#  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
	#  the vertex of the simplex to be replaced.
	ilo = indmin(y); ylo = y[ilo]

	it = 0
end

stop_crit() = all( [max(p[i,:])-min(p[i,:]) for i in 1:n] .< reqmin)

while it < maxit && !stop_crit()
	ihi = indmax(y)

	#  Calculate PBAR, the centroid of the simplex vertices
	#  excepting the vertex with Y value YNEWLO.
	pbar = [sum(p[i,1:end .!= ihi]) for i in 1:n] / n 

	#  Reflection through the centroid.
	pstar = pbar + rcoeff * (pbar - p[:,ihi])
	ystar = func(pstar)

    if ystar < ylo #  Successful reflection, so extension.
    	p2star = pbar + ecoeff * (pstar - pbar)
    	y2star = func(p2star)

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
		    y2star = func(p2star)

			if y[ihi] < y2star #  Contract the whole simplex.
				p = [ (p[i,j] + p[i, ilo]) * 0.5 for i in 1:n, j in 1:(n+1)]
				y = [ func(p[:,j]) for j in 1:(n+1)]

				ilo = indmin(y); ylo = y[ilo]

				# continue
			else #  Retain contraction.
				p[:,ihi] = p2star
				y[ihi] = y2star
			end

        elseif l == 1  #  Contraction on the reflection side of the centroid.
			p2star = pbar + ccoeff * ( pstar - pbar )
			y2star = func(p2star)

			#  Retain reflection?
			p[:,ihi] = ystar < y2star ? pstar : p2star
			y[ihi] = ystar < y2star ? ystar : y2star
		end
	end

	if y[ihi] < ylo #  Check if YLO improved.
		ylo = y[ihi]
		ilo = ihi
	end

	it += 1

	println("$it : $((ylo, p[:,ilo])) ; crit = $(max([max(p[i,:])-min(p[i,:]) for i in 1:n]))")
end


(y[ilo], p[:,ilo])

crit = max([max(p[i,:])-min(p[i,:]) for i in 1:n])



###########################

while true

	p[:,end] = init
	y[end] = func(init)
	icount += 1

	for j = 1 : n
		x = init[j]
		init[j] += step[j] * del
		p[:,j] = init
		y[j] = func(init)

		icount = icount + 1
		init[j] = x
	end
	#  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
	#  the vertex of the simplex to be replaced.
	ylo = min(y)
	ilo = findfirst(y .== ylo)

	#  Inner loop.
	while true

		if kcount <= icount; break; end

		ynewlo = max(y)
		ihi = findfirst(y .== ynewlo)

		#  Calculate PBAR, the centroid of the simplex vertices
		#  excepting the vertex with Y value YNEWLO.
		pbar = [sum(p[i,1:end .!= ihi]) / n for i in 1:n]

		#  Reflection through the centroid.
		pstar = pbar + rcoeff * (pbar - p[:,ihi])
		ystar = func(pstar)
		icount += 1

	    if ystar < ylo #  Successful reflection, so extension.
	    	p2star = pbar + ecoeff * (pstar - pbar)
	    	y2star = func(p2star)
	    	icount += 1

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
			    y2star = func(p2star)
			    icount += 1

				if y[ihi] < y2star #  Contract the whole simplex.
					for j in 1:(n+1)
						p[:,j] = ( p[:,j] + p[:,ilo] ) * 0.5
						xmin = p[:,j]
						y[j] = func(xmin)
						icount += 1
					end

					ylo = min(y)
					ilo = findfirst(ylo .== y)

					continue
				else #  Retain contraction.
					p[:,ihi] = p2star
					y[ihi] = y2star
				end

	        elseif l == 1  #  Contraction on the reflection side of the centroid.
				p2star = pbar + ccoeff * ( pstar - pbar )
				y2star = func(p2star)
				icount += 1
				#  Retain reflection?
				p[:,ihi] = ystar < y2star ? pstar : p2star
				y[ihi] = ystar < y2star ? ystar : y2star
			end
		end

		if y[ihi] < ylo #  Check if YLO improved.
			ylo = y[ihi]
			ilo = ihi
		end

		jcount -= 1

		if 0 < jcount; continue; end

		if icount <= kcount #  Check to see if minimum reached.
			jcount = konvge
			x = sum(y) / (n+1)
			z = dot(y-x, y-x)

			if z <= rq; break; end
		end

	end
	#
	#  Factorial tests to check that YNEWLO is a local minimum.
	#
	xmin = p[:,ilo]
	ynewlo = y[ilo]

	if kcount < icount; ifault = 2; break; end
	ifault = 0

	for i = 1 : n
		del = step[i] * eps

		xmin[i] += del
		z = func(xmin)
		icount += 1
		if z < ynewlo; ifault = 2; break; end

		xmin[i] -= 2*del
		z = func(xmin)
		icount += 1
		if z < ynewlo; ifault = 2; break; end

		xmin[i] = xmin[i] + del
	end

	if ifault == 0; break; end

    init = xmin #  Reinit the procedure.
    del = eps
    numres += 1

end

#########################

type interval
	l::Float64
	u::Float64
end

model = quote
	a::real
	b::real

	z = a+ b
	y = exp(z)
end

SimpleMCMC.generateModelFunction(model, 1., false, true)

                a = __beta[1]
                b = __beta[2]
                local __acc = 0.0
                z = +(a,b)
                __acc = y = exp(z)

+(a::interval, b::interval) = interval(a.l+b.l, a.u+b.u)

1.:5. + 6.:8.

sin(1.:10)
sin
typeof(1.:5.)


model = quote
	x::real

	z = exp(-x)
	y = x*z
end

SimpleMCMC.simpleNM(:(x::real ; x*exp(-x)), 3.1, 100, 1e-5)
SimpleMCMC.simpleNM(:(x::real ; x*exp(-x)), 3.1, 100)
SimpleMCMC.simpleAGD(:(x::real ; x*exp(-x)), 3.1, 100)
SimpleMCMC.simpleAGD(:(x::real ; x*exp(-x)), 10.1, 100)


function test(x,dx)
	t1 = -x
	t2 = exp(t1)
	t3 = x*t2

	dt1 = -dx
	dt2 = exp(t1+dt1)-t2
	dt3 = (x+dx)*(t2+dt2)-t3
	(min(t3,t3+dt3), max(t3,t3+dt3))
end

[ test(y,1) for y in 0.:10]
x = 0
dx = 1

test(0,1.5)  # FAUX !
test(1.5,1.5)

test(0,1)
test(1,1)

model(a::Real, b::Array) = a-log(b)






test(0.0, 0.5) 
test(0.5, 0.5)

type Inode
	l::Float64
	u::Float64
	fl::Float64
	fu::Float64



[test(0,1) ; test(1,1)]







SimpleMCMC.generateModelFunction(model, 1., true, true)

#######################################

dists = {#(:Normal,  	"dnorm4"),
			(:Weibull, 	  "dweibull",  3),
			# (:Uniform, 	  "dunif",     3),
			# (:Binomial, 	  "dbinom",    3),
			# (:Gamma,  	  "dgamma",    3),
			# (:Cauchy,  	  "dcauchy",   3),
			# (:logNormal,    "dlnorm",    3),
			# (:Beta, 	      "dbeta",     3),
			# (:Poisson,  	  "dpois",     2),
			# (:TDist,  	  "dt",        2),
			(:Exponential,  "dexp",      2)}

for d in dists # d = dists[2]

	# d = (:Binomial, 	  "dbinom",    3)
	fsym = symbol("logpdf$(d[1])")

	npars = d[3]
	myf = quote
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

	eval(myf)
end



expr(:function, 
	          expr(:call, fsym, 
	                [expr(:(::), symbol("x$i"), :Real) for i in 1:npars]...),
	          quote 
	          	local res = 0.
	            res = $(expr(:call, 
	                	 	  :ccall, 
	                	 	  :(dlsym(_jl_libRmath, $(string(d[2])))),
	                	 	  expr(:quote, Float64),
	                	 	  expr(:quote, tuple([[Float64 for i in 1:npars]..., Int32 ]...)),
	                	 	  [symbol("x$i") for i in 1:npars]...,
	                	 	  1) )
				if res == -Inf
					throw("give up eval")
				elseif res == NaN
	                local args = $(expr(:tuple, [[symbol("x$i") for i in 1:npars]..., 1]...))
					error(string("calling ", $fsym, args, "returned an error"))
			    else
			    	return(res)
			    end
			    res 
	           end)


myf = quote
	function ($fsym)($([expr(:(::), symbol("x$i"), :Real) for i in 1:npars]...))
		local res = 0.

        res = ccall(dlsym(_jl_libRmath, $(string(d[2]))), Float64,
            	 	  $(expr(:tuple, [[:Float64 for i in 1:npars]..., :Int32 ]...)),
            	 	  $([symbol("x$i") for i in 1:npars]...), 1	)

		if res == -Inf
			throw("give up eval")
		elseif res == NaN
			local args = $(expr(:tuple, [[symbol("x$i") for i in 1:npars]..., 1]...))
			error(string("calling ", $fsym, args, "returned an error"))
		end
		res 
	end
end


pars = [:Real, :AbstractArray, :Groumph]
myf = quote
	function ($fsym)($([pars[i]==:Real ? symbol("x$i") : expr(:ref, symbol("x$i"), :i) for i in 1:npars]...))
		local res = 0.
		45+x1+x2
	end
end



eval(myf)

logpdfBinomial(10,0.2,1)


dump(:(ccall(dlsym(_jl_libRmath, $(string(d[2]))), Float64, (Float64, Float64, Float64, Int32), x1, x2, x3, 1)))
dump(:(ccall(dlsym(_jl_libRmath, $(string(d[2]))), Float64,
            	 	  $(expr(:tuple, [[:Float64 for i in 1:npars]..., :Int32 ]...)),
            	 	  $([symbol("x$i") for i in 1:npars]...), 1	   ) ))

myf = quote
end

const Rmath = :libRmath

ccall(("dunif", Rmath), Float64,
                  (Float64, Float64, Float64, Int32),
                  x, d.($p1), d.($p2), d.($p3), 1)


ccall(dlsym(_jl_libRmath, "dunif"), Float64, (Float64, Float64, Float64, Int64), 0.5,0.,2.,1)

typeof(1.)
typeof(1)

