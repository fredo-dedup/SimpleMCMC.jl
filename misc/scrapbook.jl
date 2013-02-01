load("pkg")
Pkg.init()
Pkg.add("Distributions")
Pkg.update("Distributions")
Pkg.add("DataFrames")

################################
Y = [1., 2, 3, 4]
X = [0. 1; 0 1; 1 1; 1 2]

model = quote
	vars::real(2)

	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

require("../src/SimpleMCMC.jl")
include("../src/SimpleMCMC.jl")
myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

res = SimpleMCMC.simpleHMC(model,100, 5, 1.5, 10, 0.1)

l0, grad = __loglik(ones(2))
[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]


################################################

model = :(x::real; y = z-x; y ~ TestDiff())

    myf, np = SimpleMCMC.buildFunctionWithGradient(model)
    ex2 = myf.args[2].args

    cex = expr(:block, vcat({:(__beta = [1.0])}, ex2))


#####################################################


