simple-mcmc
===========

Basic mcmc samplers implemented in Julia

Implements :
1. a DSL for specifying models
2. a set of sampling functions

## The model DSL
Basically the full Julia language can be used with the following limitations :
- the `::` operator is redefined to declare model parameters being sampled, this makes the usual meaning of the operator inaccessible within the model definition
- the '~' operator is redefined to associate model variables with a distribution, with the same consequences
- as the model parsing creates temporary variables and a new function in the 'Main' namespace, there are potential variables collisions (unlikely though)
- sampling functions that calculate a gradient (HMC) are limited to the functions for which partial derivatives are specified. The list is easily extendable (see function 'derive'). Note that control flow operators (for, if, while) are not yet implemented and will necessitate a bit more work..

An example model spec should be enough to illustrate the DSL : 

`model = quote
	b::scalar
	k::vector(5)
	
	a = b+6
	x = sin(dot(k, z))

	x ~ Weibull(a, 2.0)
end`

- `b::scalar` declares a model parameter, to be sampled, here `b` is defined as a simple scalar
- `k::vector(5)` declares a vector model parameter of length 5, the size can be an expression as long as it is evaluable to a stricly positive integer, otherwise an error will be thrown
- `x ~ Weibull(a, 2.0)` statements with the operator ~ declare how to build the model likelihood, here this says that x should have a Weibull distributions (any continuous distribution of the "Distribution.jl" can be used) of shape `a` and scale 2
- other statements are evaluated normally and can either use or define model-local variables (`a`, `x`) or use variables defined in the calling environment (`z`)

## The sampling functions
Currently, we have `simpleRWM` running a random walk metropolis and `simpleHMC` running an Hamiltonian Monte-Carlo with a reverse mode gradient calculation.

Calling syntax
- simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
- simpleRWM(model::Expr, steps::Integer, burnin::Integer) : with inital values set to 1.0
- simpleRWM(model::Expr, steps::Integer) : with burnin equal to half of steps

- simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, length::Integer, stepsize::Float64)
- simpleHMC(model::Expr, steps::Integer, burnin::Integer, length::Integer, stepsize::Float64) : with inital values set to 1.0
- simpleHMC(model::Expr, steps::Integer, length::Integer, stepsize::Float64) : with burnin equal to half of steps

## Examples


## TODO
- Add adaptative algorithms for simpleHMC for length and stepsize parameters
- Add a NUTS type HMC sampler ?
- Convert the set of jags/bugs examples, run them and compare timings
- Specify partial derivatives of other continuous distributions
- Add discrete distributions ?

