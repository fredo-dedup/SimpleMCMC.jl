simple-mcmc
===========

Basic mcmc samplers written in Julia

Implements :
1. a DSL for specifying models
2. a set of sampling functions

## The model DSL
Largely inspired from the BUGS/JAGS syntax, it uses the `~` operator to associate a variable with a distribution. Model parameters are defined with the `::` operator followed by the keyword real and optional dimensions in parentheses (up to 2 dimensions).

An example model spec should make it very clear : 

```
model = quote
	b::real
	k::real(5)
	
	a = b+6
	x = exp(-dot(k, z))

	x ~ Weibull(a, 2)
end
```
- `::` indicates variables to be sampled (i.e model parameters): `b::real` declares a scalar parameter b, `k::real(5)` declares a vector k of size 5. The size can be also be an expression as long as it is evaluates to a stricly positive integer (an error will be thrown otherwise).
- Statements with the operator ~ (`x ~ Weibull(a, 2)`) declare how to build the model likelihood, here this says that x should have a Weibull distributions (SimpleMCMC.jl uses the same naming conventions as the "Distribution.jl" module).
- other statements are evaluated normaly and can either use and define model-local variables (`a`, `x`) or use variables defined in the calling environment (`z`) __calling environment actually means in `Main` !__.

This usage make the usual meaning of `~` and `::` in Julia inacessible within a model definition. Note too that the transformation of the model expression by the parser creates 2 additionnal variables (__acc and __beta) => don't use those names !

The samplers using gradients (simpleHMC and simpleNUTS) require an additionnal parsing step that will generate the gradient code by automated derivation. This marginally increases the calculation time (by O(1), not by O(# of parameter) ) but it imposes limits of what the model can contain : there are the 'nice' limitations : functions and distributions whose partial derivatives are not handled (see src/diff.jl), these can be added easily (you can contribute !). And then there are the other limitations : control flow operators (if, for loops, ..) are not handled yet because they will necessitate a bit more work on the parsing functions (`if` may not be too hard though). Note that very often for loops can be replaced by matrix/vector algebra (see examples) so this may not be such a big limitation.

A last note : the automated derivation will not look into refs, if somehow a ref depends directly or indirectly on a model parameter (for example  `x = Y[ round(sigma)]` ) , the gradient will be false.

## The sampling functions
Currently, we have `simpleRWM` running a random walk metropolis, `simpleHMC` running an Hamiltonian Monte-Carlo and `simpleNUTS` using the NUTS algo to automatically find the proper number of inner steps and stepsize.

###Calling syntax
- `simpleRWM(model, steps, burnin, init)` : `init` can either be a vector (same size as the number of parameters) or a real that will be assigned to all parameters
- `simpleRWM(model, steps, burnin)` : with inital values set to 1.0
- `simpleRWM(model, steps)` : with burnin equal to half of steps

- `simpleHMC(model, steps, burnin, init, length, stepsize)`
- `simpleHMC(model, steps, burnin, length, stepsize)` : with inital values set to 1.0
- `simpleHMC(model, steps, length, stepsize)` : with burnin equal to half of steps

- `simpleNUTS(model, steps, burnin, init)` 
- `simpleNUTS(model, steps, burnin)` : with inital values set to 1.0
- `simpleNUTS(model, steps)` : with burnin equal to half of steps


###Return value
A structure containing the samples and a few stats (use dump to see what's inside it is self explanatory).

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
# or a NUTS flavoured HMC
res = SimpleMCMC.simpleNUTS(model, 1000)
```

## Issues
- May be some bugs left in the NUTS implementation as it sometimes seem to go into a huge amount of doubling steps and converges toward tiny epsilons
- Gradient generated code is not optimized, could be better...

## Ideas for improvements
- Compare timings with other sampling tools (JAGS, STAN)
- Specify partial derivatives of other continuous distributions (currently only Normal, Uniform, Bernoulli and Weibull)
- Enable other functions for automated derivation : max, min, transpose, ? :, ...
- Add truncation, censoring

