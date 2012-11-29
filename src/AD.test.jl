
type ADVar
	x::Float64
	dx::Vector{Float64}
end
ADVar(x::Float64, dim::Integer, index::Integer) = ADVar(x, [i==index ? 1.0 : 0.0 for i in 1:dim])
ADVar(x::Int32, dim::Integer, index::Integer) = ADVar(convert(Float64,x), [i==index ? 1.0 : 0.0 for i in 1:dim])
ADVar(3.0, 4, 3)

ADVar(x::Number, dx::Vector{Number}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
ADVar(x::Int32, dx::Vector{Int32}) =  ADVar(convert(Float64,x), convert(Vector{Float64}, dx))
ADVar(x::Float64, dx::Vector{Int32}) =  ADVar(x, convert(Vector{Float64}, dx))
ADVar(x::Int32, dx::Vector{Float64}) =  ADVar(convert(Float64,x), dx)

ADVar(3.0, [1., 2])
ADVar(3.0, [1, 2])
ADVar(3, [1, 2])
ADVar(3, [1., 2])

###
	convert(::Type{ADVar}, x::Number) = ADVar(x, zeros(2))
	promote_rule(::Type{Float64}, ::Type{ADVar}) = ADVar
	promote_rule(::Type{Number}, ::Type{ADVar}) = ADVar

	ADVar(4., [3.])

	convert(Number, 4.5)
	typeof(4) <: Number
	is(4, Integer)

	a = ADVar(3, 3, 1)  # error
	a = ADVar(3.0, 3, 1)
	a.x
	a.dx
	a
	x

######### addition  ######################
	+(x::ADVar, y::ADVar) = ADVar(x.x + y.x, x.dx + y.dx)
	+(x::ADVar, y::Any) = ADVar(x.x + y, x.dx)
	+(x::Any, y::ADVar) = +(y, x)

	@assert  (ADVar(4., [1., 0]) + ADVar(4., [1., 0])).x == 8.
	@assert  (ADVar(4., [1., 0]) + ADVar(4., [1., 0])).dx == [2., 0]
	@assert  (ADVar(4., [1., 0]) + 1.).x == 5.
	@assert  (ADVar(4., [1., 0]) + 4.).dx == [1., 0]
	@assert  (1. + ADVar(4., [1., 0])).x == 5.
	@assert  (4. + ADVar(4., [1., 0])).dx == [1., 0]

######### substraction ######################
	-(x::ADVar) = ADVar(-x.x, -x.dx)  # unary
	-(x::ADVar, y::ADVar) = ADVar(x.x - y.x, x.dx - y.dx)
	-(x::ADVar, y::Any) = ADVar(x.x - y, x.dx)
	-(x::Any, y::ADVar) = -(y - x)

	@assert  (ADVar(4., [1., 0]) - ADVar(4., [1., 0])).x == 0.
	@assert  (ADVar(4., [1., 0]) - ADVar(4., [1., 0])).dx == [0., 0]
	@assert  (ADVar(4., [1., 0]) - 1.).x == 3.
	@assert  (ADVar(4., [1., 0]) - 4.).dx == [1., 0]
	@assert  (1. - ADVar(4., [1., 0])).x == -3.
	@assert  (4. - ADVar(4., [1., 0])).dx == [-1., 0]

######### multiplication  ######################
	*(x::ADVar, y::ADVar) = ADVar(x.x * y.x, x.dx*y.x + y.dx*x.x)
	*(x::ADVar, y::Any) = ADVar(x.x*y, x.dx*y)  # not correct for matrices !
	*(x::Any, y::ADVar) = *(y, x)

	@assert  (ADVar(4., [1., 0]) * ADVar(4., [1., 0])).x == 16.
	@assert  (ADVar(4., [1., 0]) * ADVar(4., [0., 1])).dx == [4., 4]
	@assert  (ADVar(4., [1., 0]) * ADVar(4., [1., 0])).dx == [8., 0]
	@assert  (ADVar(4., [1., 0]) * 1.).x == 4.
	@assert  (ADVar(4., [1., 0]) * 4.).dx == [4., 0]
	@assert  (1. * ADVar(4., [1., 0])).x == 4.
	@assert  (4. * ADVar(4., [1., 0])).dx == [4., 0]

	#  x  + y^2 avec (x=4 et y=2)
	(ADVar(4., [1., 0]) + (ADVar(2., [0., 1]) * ADVar(2., [0., 1]))).x  # 8
	(ADVar(4., [1., 0]) + (ADVar(2., [0., 1]) * ADVar(2., [0., 1]))).dx #  [1.  4]

	#  x * y^2 avec (x=2 et y=3)
	(ADVar(2., [1., 0]) * (ADVar(3., [0., 1]) * ADVar(3., [0., 1]))).x  # 18
	(ADVar(2., [1., 0]) * (ADVar(3., [0., 1]) * ADVar(3., [0., 1]))).dx #  [9.  12.]

######### division  ######################
	/(x::ADVar, y::ADVar) = ADVar(x.x / y.x, (x.dx*y.x - y.dx*x.x) / (y.x * y.x))
	/(x::ADVar, y::Any) = ADVar(x.x / y, x.dx / y) 
	/(x::Any, y::ADVar) = *(x / y.x, - y.dx * (x / (y.x * y.x)))

	@assert  (ADVar(4., [1., 0]) / ADVar(2., [1., 0])).x == 2.
	@assert  (ADVar(4., [1., 0]) / ADVar(2., [0., 1])).dx == [.5, -1]
	@assert  (ADVar(4., [1., 0]) / 2.).x == 2.
	@assert  (ADVar(4., [1., 0]) / 2.).dx == [.5, 0]
	@assert  (1. / ADVar(4., [1., 0])).x == 0.25
	@assert  (4. * ADVar(4., [1., 0])).dx == [4., 0]

######### power  ######################
	^(x::ADVar, y::ADVar) = ADVar(x.x ^ y.x, (y.x * x.x^(y.x-1.)) * x.dx + (log(x.x) * x.x^y.x) * y.dx)
	^(x::ADVar, y::Any) = x ^ ADVar(y, zeros(size(x.dx))) 
	^(x::Any, y::ADVar) = ADVar(x, zeros(size(y.dx))) ^ y

	@assert (ADVar(3.0, [1., 0]) ^ 3).x == 27.
	@assert (ADVar(3.0, [1., 0]) ^ 3).dx == [27., 0]
	@assert (2.0 ^ ADVar(3.0, [1., 0])).x == 8.
	@assert norm((2.0 ^ ADVar(3.0, [1., 0])).dx - [5.54518, 0] ) < 1e-5


########## scrapbook  #######################

test(x) = (x+2) * (x+4)
test(ADVar(-2., [1., 0])).x
test(ADVar(-4., [1., 0])).dx

test(x) = (x+2) * (x+4) - 3
test(ADVar(-2., [1., 0])).x


[ [test(x)::Any, test(ADVar(float(x), [1., 0]))] for x in -10:1:10]
[ test(x).x for x in -10:1:10]

x = 4
test


end


3. * [ADVar(1., [1., 2]), ADVar(2., [0., 2])]
3. + [ADVar(1., [1., 2]), ADVar(2., [0., 2])]

ADVar(4.2, [1.0, 20.2, 356.])
4 + 5 
b = a + 6
b.x
(6 + a).x

b
(a+b).dx


type abcd
	x::Float64

	abcd(x0::Number) = new(x0)
end

abcd(13.2)
abcd(13)

