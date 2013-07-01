SimpleMCMC.jl
=============

Small framework for MCMC sampling and maximization on user-defined models

Implements :
- a DSL for specifying models
- an automatic derivation of models (using reverse accumulation) which can be called independently to produce a stand-alone function
- a set of sampling methods : Randow Walk Metropolis, Hamiltonian Monte-Carlo and NUTS Hamiltonian Monte-Carlo (the last two using automatic derivation)
- a set of solving methods : Nelder-Mead and accelerated gradient descent (this last one using automatic derivation)


In a nutshell this allows quickly specifying a model and launch MCMC sampling and/or optimizing, letting the library take care of gradient code generation :

```jl
# Y is a vector of N outcomes
# X is a N x M matrix of predictors

# the model
model = quote
	resid = Y - X * coefs    # linear model to explain Y (vectorwith predictors X
	resid ~ Normal(0, 1.0)   # Normal prior, zero mean and unit standard deviation on residuals
end

simpleAGD(model, coefs=zeros(M))           # find MLE by accelerated gradient descent

simpleRMW(model, steps=10000, coefs=zeros(M))  # Random Walk Metropolis

```


## The model DSL
It simply is a Julia Expression enclosed `:(...)` or `quote .. end` that follows the language syntax except for the `~` operator used here to associate a variable with a distribution (same as the BUGS/JAGS/STAN syntax). The model expression may completely omit `~` operator in which case the last evaluated statement will be considered the variable to be sampled or optimized.

Valid model expressions : 

```jl
model = quote
	a = b+6
	x = exp(-dot(k, z))

	x ~ Normal(a, 2)
end

model2 = :(y = A * z ; dot(y,y))
```

### Variables
Model variables fall in three categories :
- Model parameters : these are inferred by the method calls (function generation, sampling, solving) to be the keyword arguments remaining after taking away those that have a specific meaning to the method ( steps / burnin / precision, etc). _Note that the use of keyword arguments to pass model parameters names prohibits parameter names in the model such as 'steps' / 'burnin', etc._
- Variables defined within the model : such as `a`, `x`, `y` in the examples above
- And the variables that are neither model parameters or defined variables : these are considered as external variables and will be regarded as constants for model evaluation and differentiation. The model function produced will look for them in the Main module (they have to be top level variables to be visible from the point of view of the SimpleMCMC module).

_The generated function will create many temporary variables with two of them having a fixed denomination (`__beta` for the argument of the model function, and `__acc` for the log-likelihood accumulator). If you use those names in your model definition the results might become unpredictable._

### Distributions
Statements with the operator `~` declare how to build the model likelihood, e.g. `x ~ Normal(a, 2)` says that x should have a Normal distribution.

Currently, the available distributions are : 

Distribution  |   Notes
--------------|-----------
`Normal(mu, sigma)`		|  sigma > 0
`Uniform(min, max)`		|  min < max
`Weibull(shape, scale)`		|  shape and scale > 0
`Beta(a, b)`			|  a and b > 0
`TDist(df)`			|  df > 0
`Exponential(scale)`		|  scale > 0
`Gamma(shape, scale)`		|  shape and scale > 0	
`Cauchy(mu, scale)`	       	|  scale > 0
`logNormal(logmu, logscale)`   	|
`Bernoulli(prob)`		|  0 <= prob <= 1, sampled var is an integer and cannot depend on model parameters
`Binomial(size, prob)`		|  0 <= prob <= 1, sampled var is an integer and cannot depend on model parameters
`Poisson(lambda)`		|  sampled var is an integer and cannot depend on model parameter

All follow the "Distributions" package conventions for naming and arguments.

### Allowed functions in a model expression
Besides the usual Julia meaning of `~` that becomes unavailable within a model definition (since it now links a variable with a distribution), only a small subset of functions can be used if the gradient calculations steps are generated (even if no gradient is required there are still limitations). Notably, if statements, for/while loops, comprehensions, are not currently possible (you will have to use matrix/vector algebra to replace for loops, or max/min/abs functions to replace if-then-else ).

Supported functions are : 

operator       |   arguments
-------------|----------
`+`  			| with operands scalar, vector or matrix (of compatible size)
`-` (unary)  	| with operand scalar, vector or matrix 
`-` (binary) 	| with operands scalar, vector or matrix (of compatible size)
`sum()`  		| with operand scalar, vector or matrix 
`dot(,)` 		| with operands scalar or vector  (not matrix)
`log()`  		| with operand scalar, vector or matrix 
`exp()`  		| with operand scalar, vector or matrix 
`sin()`  		| with operand scalar, vector or matrix 
`cos()`  		| with operand scalar, vector or matrix 
`abs()`  		| with operand scalar, vector or matrix 
`*`      		| with operands scalar, vector or matrix (of compatible size)
`.*`      		| with operands scalar, vector or matrix (of compatible size)
`^`      		| with operands scalar only
`.^`      		| with operands scalar, vector or matrix (of compatible size)
`/`      		| with operands scalar, vector or matrix, with at least one operand scalar
`./`      		| with operands scalar, vector or matrix (of compatible size)
`max(,)` (binary only) | with operands scalar, vector or matrix (of compatible size)
`min(,)` (binary only) | with operands scalar, vector or matrix (of compatible size)
`transpose()` or `'` | with operand scalar, vector or matrix
`+=`, `-=` and `*=` | -

## The model function
Calling the solving and MCMC sampling tools will generate the model function transparently, so you normally do not need to go through this step. However, if you do need to get the model function directly, with or without gradient code, you can call the `generateModelFunction` method directly :

`mf, nparams, map, init = generateModelFunction(model, gradient=false, debug=false, x=., y=., etc.)`

or alternatively : 

`mf, nparams, map, init = generateModelFunction(model, gradient=false, debug=false; init...)`

- `model` is the model expression, 
- `gradient` specifies if the gradient code is to be calculated or not
- `debug` = true, dumps the function code without generating anything, useful for debugging
- other arguments are assumed to be model parameters. They can be passed separately or in a Dict (`init`)

Returned values are :
- `mf` : the model function returning a single scalar (`gradient=false`), or a scalar + vector for the gradient (`gradient=true`). This function requires the model parameters values to be passed in a single vector.
- `nparams` : the length of the parameter vector
- `map` : a mapping structure specifying how to go from the parameter vector to the parameters as indicated in the function call
- `init` : inital values in vector form

example : 

```jl
A = rand(10,10) # external variable
model = :( y=A*x; dot(y,y)) # model

# generate function, with gradient, x being the model parameter (a vector of length 10) with initial values = zero
testf, n, map, init = generateModelFunction(model, gradient=true, x=zeros(10))
testf(rand(10)) # value and gradient for a random value of x

# or just check out the generated code
generateModelFunction(model, gradient=true, debug=true, x=zeros(10))
```

## Solving tools
Two solving algorithms are implemented. Note that they do not play well with functions close to the limit of their support (such as models with distributions Uniform, Gamma, ..) especially if the optimum is close to the support frontier.
Additionally, the accelerated gradient descent supposedly needs a convex function (concave in fact since we are maximizing the model) and may not behave properly if you do not have this property around the initial values and the optimum.

`simpleNM(model, maxiter=100, precision=1e-3; init...)` for Nelder-Mead optimization
and
`simpleAGD(model, maxiter=100, precision=1e-5; init...)` for Accelerated Gradient Descent

## MCMC sampling tools
Three sampling methods are provided by the SimpleMCMC package : plain Random Walk Metropolis (with an automated scaling), Standard and NUTS variant Hamiltonian Monte-Carlo : 

- `simpleRWM(model, steps=1000, burnin=100; init...)` for Random Walk Metropolis (isteps is the number of samples, burnin the number of initial samples to ignore in the result)
- `simpleHMC(model, steps=1000, burnin=100, isteps=2, stepsize=1e-3; init...)` for Hamiltonian Monte-Carlo (isteps is the number of inner steps, stepssize their scale).
- `simpleNUTS(model, steps=1000, burnin=100; init...)` for NUTS type Hamiltonian Monte-Carlo.

Each function prints out basic info at the end of the run such as the running time, the min and max of effective sample size accross parameters and a few other stats. The details are contained in the returned structure.


## Examples

### Linear regression

```jl
# Generate values
srand(1)
n = 1000 # number of observations
nbeta = 10 # number of predictors, including intercept
X = [ones(n) randn((n, nbeta-1))]
beta0 = randn((nbeta,))
Y = X * beta0 + randn((n,))

# define model
model = quote
	vars ~ Normal(0, 1.0)  # Normal prior, zero mean and unit standard deviation for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  # Normal prior, zero mean and unit standard deviation for residuals
end


res = simpleRWM(model, steps=10000, vars=zeros(nbeta)) # random walk metropolis (10000 steps, 100 burnin)

res = simpleHMC(model, 1000, 100, 5, 1e-1, vars=zeros(nbeta)) # HMC (with 5 inner steps of size 0.1)

res = simpleNUTS(model, 1000, vars=zeros(nbeta)) # or a NUTS flavoured HMC
```

_______________________________________________________________
## Issues
- May be some bugs left in the NUTS implementation as it sometimes seem to go into a huge amount of doubling steps and converges toward tiny epsilons
- The automated derivation will not look into refs, if somehow a ref depends directly or indirectly on a model parameter (for example  `x = Y[ round(sigma)]` ), the gradient will be false.
- Setting a subset of values in a vector or array may cause a false gradient
- simpleNM could handle better support exit
- Code of simpleAGD needs cleaning up

_______________________________________________________________
## (Possible) future work
- Enable other functions for automated derivation : vcat, hcat, one, zero, comprehensions ...  ?
- Add for loops and ifs ?
- Add truncation, censoring
- Compare timings with other sampling tools (JAGS, STAN)
- _ideas ?_

_______________________________________________________________
## History

Date         |   Changes
-------------|----------
june 14th | removed model parameters definition from model DSL, they are now within the methods arguments, thanks to keyword args
	  | optimized generated function, for a x2-x3 speedup
	  | added solving functions (for maximization to be consistent with log-likelihood functions) using Nelder-Mead and Accelerated Gradient Methods
	  | changed derivation rules format (in diff.jl) allowing different formulas depending on parameter type
april 11th   | added major missing distributions (Gamma, TDist, Exponential, Cauchy, logNormal, Poisson, Binomial and Beta) 
             | added some functions that can be derived (min, max, abs, transpose, +=, *=)
             | simplified unit testing of derivation and distributions that will make future improvements much easier to test

_______________________________________________________________

