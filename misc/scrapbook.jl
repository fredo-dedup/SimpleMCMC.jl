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

