

######################################################
begin
	func(v) = ((v[2]-1)^2*5+(v[1]-2)^2)
	init = [0.1,5]

	reqmin = 0.1
	step = ones(length(init))

	##
	assert(reqmin>0.)


	n = length(init)
	assert(n>=1)

	ccoeff = 0.5
	ecoeff = 2.0
	rcoeff = 1.0


	maxit = 10

	p = [ init[i] + step[i] * (i==j) for i in 1:n, j in 1:(n+1)]
	y = Float64[ func(p[:,j]) for j in 1:(n+1)]

	#  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
	#  the vertex of the simplex to be replaced.
	ilo = indmin(y); ylo = y[ilo]


	it = 0
end

	it = 0

while (it < maxit) && ((max([max(p[i,:])-min(p[i,:]) for i in 1:n])) > reqmin) #var(y) > reqmin
# 	ihi = indmax(y)

# 	#  Calculate PBAR, the centroid of the simplex vertices
# 	#  excepting the vertex with Y value YNEWLO.
# 	pbar = [sum(p[i,1:end .!= ihi]) for i in 1:n] / n 

# 	#  Reflection through the centroid.
# 	pstar = pbar + rcoeff * (pbar - p[:,ihi])
# 	ystar = func(pstar)

#     if ystar < ylo #  Successful reflection, so extension.
#     	p2star = pbar + ecoeff * (pstar - pbar)
#     	y2star = func(p2star)

# 		#  Check extension.
# 		p[:,ihi] = ystar < y2star ? pstar : p2star
# 		y[ihi] = ystar < y2star ? ystar : y2star

# 	else #  No extension.
# 		l = sum(ystar .< y)

# 		if l > 1
# 			p[:,ihi] = pstar
# 			y[ihi] = ystar

# 	    elseif l == 0 #  Contraction on the Y(IHI) side of the centroid.
# 		    p2star = pbar + ccoeff * ( p[:,ihi] - pbar )
# 		    y2star = func(p2star)

# 			if y[ihi] < y2star #  Contract the whole simplex.
# 				p = [ (p[i,j] + p[i, ilo]) * 0.5 for i in 1:n, j in 1:(n+1)]
# 				y = [ func(p[:,j]) for j in 1:(n+1)]

# 				ilo = indmin(y); ylo = y[ilo]

# 				# continue
# 			else #  Retain contraction.
# 				p[:,ihi] = p2star
# 				y[ihi] = y2star
# 			end

#         elseif l == 1  #  Contraction on the reflection side of the centroid.
# 			p2star = pbar + ccoeff * ( pstar - pbar )
# 			y2star = func(p2star)

# 			#  Retain reflection?
# 			p[:,ihi] = ystar < y2star ? pstar : p2star
# 			y[ihi] = ystar < y2star ? ystar : y2star
# 		end
# 	end

# 	if y[ihi] < ylo #  Check if YLO improved.
# 		ylo = y[ihi]
# 		ilo = ihi
# 	end

	it += 1

	println("$it : $((ylo, p[:,ilo])) ; crit = $(max([max(p[i,:])-min(p[i,:]) for i in 1:n]))")
end


(y[ilo], p[:,ilo])
