SimpleMCMC.jl
=============

Small framework for MCMC sampling and maximization on user-defined models

Implements :
- a DSL for specifying models
- an automatic derivation of models (using reverse accumulation), producing a stand-alone function
- a set of sampling methods : Randow Walk Metropolis, Hamiltonian Monte-Carlo and NUTS Hamiltonian Monte-Carlo (the last two using automatic derivation)
- a set of solving methods : Nelder-Mead and accelerated gradient descent (this last one using automatic derivation)

## History

Date         |   Changes
-------------|----------
june 14th	 | removed model parameters definition from model DSL, they are now within the methods arguments, thanks to keyword args
			 | optimized generated function, for a x2-x3 speedup
			 | added solving functions (for maximization to be consistent with log-likelihood functions) using Nelder-Mead and Accelerated Gradient Methods
			 | changed derivation rules format (in diff.jl) for simplicity
-------------|----------
april 11th   | added major missing distributions (Gamma, TDist, Exponential, Cauchy, logNormal, Poisson, Binomial and Beta) 
             | added some functions that can be derived (min, max, abs, transpose, +=, *=)
             | simplified unit testing of derivation and distributions that will make future improvements much easier to test

______________________________________________
__Update (april 11th) :__
- __added most missing distributions (Gamma, TDist, Exponential, Cauchy, logNormal, Poisson, Binomial and Beta)__
- __...and a few functions (min, max, abs, transpose, +=, *=)__ 
- __and simplified unit testing of derivation and distributions that will make future improvements much easier to test__

_______________________________________________________________

## The model DSL
Inspired from the BUGS/JAGS syntax, it uses the `~` operator to associate a variable with a distribution. Model parameters are defined with the `::` operator followed by the keyword real and optional dimensions in parentheses (up to 2 dimensions). The rest is standard Julia operators and syntax.

An example model spec should make it very clear : 

```jl
model = quote
	b::real
	k::real(5)
	
	a = b+6
	x = exp(-dot(k, z))

	x ~ Normal(a, 2)
end
```
- `::` indicates variables to be sampled (i.e model parameters): `b::real` declares a scalar parameter b, `k::real(5)` declares a vector k of size 5. The size can be also be an expression as long as it is evaluates to a stricly positive integer (an error will be thrown otherwise).
- Statements with the operator ~ (`x ~ Normal(a, 2)`) declare how to build the model likelihood, here this says that x should have a Normal distributions (SimpleMCMC.jl uses the same naming conventions as the "Distribution.jl" module).
- other statements are evaluated normaly and can either use and define model-local variables (`a`, `x`) or use variables defined in the calling environment (`z`) _note that variables of the calling environment have to be visible to the generated log-likelihood function and should be therefore globals !_.

The DSL design decisions make the usual meaning of `~` and `::` in Julia inacessible within a model definition. Note too that the transformation of the model expression by the parser creates additionnal variables whose name you should avoid in your model : `__acc` and `__beta`. The model parsing won't check for naming collisions.

The samplers using gradients (simpleHMC and simpleNUTS) require an additionnal parsing step that will generate the gradient code by automated derivation. This marginally increases the calculation time (by O(1), not by O(# of parameter) ) but it imposes limits of what the model can contain : there are the 'nice' limitations : functions and distributions whose partial derivatives are not defined (see src/diff.jl), these can be added easily (file an issue for the functions that you miss the most). And then there are the other limitations : control flow operators (if, for loops, ..) are not handled yet because they will necessitate a bit more work on the parsing functions. Note that very often for loops can be replaced by matrix/vector algebra (see examples) so this may not be such a big limitation.

A last note : the automated derivation will not look into refs, if somehow a ref depends directly or indirectly on a model parameter (for example  `x = Y[ round(sigma)]` ) , the gradient will be false.

###Available operators

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

###Available distributions
All distributions follow the "Distributions" library conventions for naming and arguments.

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


## The sampling functions
Currently, we have `simpleRWM` running a random walk metropolis, `simpleHMC` running an Hamiltonian Monte-Carlo and `simpleNUTS` using the NUTS algo to automatically find the proper number of inner steps and stepsize.

###Calling syntax
- `simpleRWM(model, steps, burnin, init)` : `init` can either be a vector (same size as the number of parameters) or a real that will be assigned to all parameters
- or `simpleRWM(model, steps, burnin)` : with inital values set to 1.0
- or `simpleRWM(model, steps)` : with burnin equal to half of steps
- `simpleHMC(model, steps, burnin, init, length, stepsize)` : length is # sub-steps and `stepsize` their size
- or `simpleHMC(model, steps, burnin, length, stepsize)` : with inital values set to 1.0
- or `simpleHMC(model, steps, length, stepsize)` : with burnin equal to half of steps
- `simpleNUTS(model, steps, burnin, init)` : _same as Random Walk Metropolis_
- or `simpleNUTS(model, steps, burnin)` : with inital values set to 1.0
- or `simpleNUTS(model, steps)` : with burnin equal to half of steps


###Return value
A structure containing the samples and a few stats (use dump to see what's inside).

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
	vars::real(nbeta)

	vars ~ Normal(0, 1.0)  # Normal prior, zero mean and unit standard deviation for predictors
	resid = Y - X * vars
	resid ~ Normal(0, 1.0)  # Normal prior, zero mean and unit standard deviation for residuals
end

# run random walk metropolis (10000 steps, 5000 for burnin)
res = SimpleMCMC.simpleRWM(model, 10000)
# or a Hamiltonian Monte-Carlo (with 5 inner steps of size 0.1)
res = SimpleMCMC.simpleHMC(model, 1000, 5, 1e-1)
# or a NUTS flavoured HMC
res = SimpleMCMC.simpleNUTS(model, 1000)
```

## Issues
- May be some bugs left in the NUTS implementation as it sometimes seem to go into a huge amount of doubling steps and converges toward tiny epsilons
- Gradient generated code is not optimized, could be better... though most of the computation time seems to be spent in Rmath calls for distributions.

## (Possible) future work
- Compare timings with other sampling tools (JAGS, STAN)
- Enable other functions for automated derivation : vcat, hcat, one, zero, comprehensions ...  ?
- Add for loops and ifs ?
- Add truncation, censoring
- Add a gradient descent function (CG, Nesterov) to find mode of models ?
- _ideas ?_

