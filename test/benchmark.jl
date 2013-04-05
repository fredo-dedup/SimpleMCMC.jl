
include("../src/SimpleMCMC.jl")


macro timeit(ex)
    @gensym t i
    quote
        $t = Inf
        for $i=1:5
            $t = min($t, @elapsed $ex)
        end
        $t
    end
end


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


ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
init = SimpleMCMC.setInit(1.0, nparams)
g = N -> for i in 1:N;ll_func(init);end

t = @timeit g(1000)

require("base/Git.jl")
using Base.Git
import Git

head()

cd("..")
pwd()
`git status`





