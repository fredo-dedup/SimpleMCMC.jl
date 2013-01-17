simple-mcmc
===========

Basic mcmc samplers written in Julia

Implements :
1. a DSL for specifying models
2. a set of sampling functions

## The model DSL
For sampling without gradient (simpleRWM), basically the full Julia language can be used with the following limitations :
- the `::` operator is redefined to declare model parameters being sampled, this makes the usual meaning of the operator inaccessible within the model definition.
- the `~` operator is redefined to associate model variables with a distribution, with the same consequences.
- as the model parsing creates a new function in the 'Main' namespace (__loglik), and introduces new variables within the model (__acc and __beta) there are potential variables collisions : don't use those names !

For sampling using gradient (simpleHMC), the reverse mode automated derivation implemented in this library adds significant limits : 
- No control flow operators (if, for loops, ..) are currently possible. These will necessitate a bit more work on the parsing functions. Note that very often for loops can be replaced by matrix/vector operations (see examples) so this may not be such a big limitation.
- Only a subset of julia's operators and functions can be derived (an error message will appear otherwise). The list can easily be extended though.
- If reference within vector/matrix depends directly or indirectly on a model parameter (for example  `shift = X[ round(sigma)]` ) , the gradient will be false.
- Only a few of the continuous distributions of the 'Distributions' library are implemented : Normal, Uniform and Weibull. The list can easily be extended here too.


An example model spec should be enough to illustrate the DSL : 

```
model = quote
	b::real
	k::real(5)
	
	a = b+6
	x = sin(dot(k, z))

	x ~ Weibull(a, 2.0)
end
```

- `::` indicates variables to be sampled : `b::real` declares a scalar model parameter b, `k::real(5)` declares a vector k of size 5. The size can be also be an expression as long as it is evaluates to a stricly positive integer (an error will be thrown otherwise).
- Statements with the operator ~ (`x ~ Weibull(a, 2.0)`) declare how to build the model likelihood, here this says that x should have a Weibull distributions (any continuous distribution of the "Distribution.jl" module can be used for Random Walk Metropolis, sampling methods using the gradient are limited to those with partial derivatives defined) of shape `a` and scale `2`
- other statements are evaluated normaly and can either use and define model-local variables (`a`, `x`) or use variables defined in the calling environment (`z`)

## The sampling functions
Currently, we have `simpleRWM` running a random walk metropolis and `simpleHMC` running an Hamiltonian Monte-Carlo with a reverse mode gradient calculation (for the curious you can run SimpleMCMC.buildFunctionWithGradient(model) to get the generated code).

###Calling syntax
- `simpleRWM(model, steps, burnin, init)` : `init` can either be a vector (same size as the number of parameters) or a real that will be assigned to all parameters
- `simpleRWM(model, steps, burnin)` : with inital values set to 1.0
- `simpleRWM(model, steps)` : with burnin equal to half of steps

- `simpleHMC(model, steps, burnin, init, length, stepsize)`
- `simpleHMC(model, steps, burnin, length, stepsize)` : with inital values set to 1.0
- `simpleHMC(model, steps, length, stepsize)` : with burnin equal to half of steps

###Return value
A Float64 Matrix with : 
- the first column containing the log-likelihoog
- the second column containing a flag indicating acceptance or rejection
- and 1 column for each model parameter

by (steps - burnin) rows.


## Examples

### Linear regression

```
# Generate values

srand(1)
n = 1000
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = X * beta0 + randn((n,))

# define model
model = quote
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, variance 1.0 for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  
end

# run random walk metropolis (10000 steps, 500 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000)
# or a Hamiltonian Monte-Carlo (with 5 inner steps of size 0.1)
res = SimpleMCMC.simpleHMC(model, 1000, 5, 1e-1)
```

## Issues
- Hangs a lot when apparently calling logpdf(_Distributions_) outside the distribution support, have to look into that...
- Gradient generated code is not optimized, could be better...

## Ideas for improvements
- Build a serious test directory
- Add adaptative algorithms for simpleHMC for inner steps and stepsize parameters (using NUTS ?)
- Convert the set of jags/bugs examples, run them and compare timings
- See if optimizing the generated gradient code is possible
- Specify partial derivatives of other continuous distributions (currently only Normal, Uniform and Weibull)
- Add truncation ?
- Allow the declaration of validity domains for model parameters ?
- Add discrete distributions ?

